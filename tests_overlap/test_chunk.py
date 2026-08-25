"""Chunk-wise MoE FFN: correctness of task claiming, spin-wait and token completion.

Usage:
    python tests_overlap/test_chunk.py                        # all chunks ready up-front
    python tests_overlap/test_chunk.py --arrival cpu          # CPU feeds `ready`
    python tests_overlap/test_chunk.py --arrival gpu          # a producer kernel feeds `ready`
    python tests_overlap/test_chunk.py --check-signal         # verify `token_done` chunk by chunk
"""

import argparse
import ctypes
import time

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.cuda.set_device(1)
paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

import deep_gemm
print("deep_gemm:", deep_gemm.__path__)

E = 16
H = 4096
I = 2048
CHUNK = 4096
NUM_SMS = 96
ALIGNMENT = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrival", choices=["ready", "cpu", "gpu"], default="ready",
                        help="who marks the chunks as arrived: nobody (all ready), the CPU, or a producer kernel")
    parser.add_argument("--interval-ms", type=float, default=0.5,
                        help="delay between two arrivals, for the `cpu` and `gpu` modes")
    parser.add_argument("--check-signal", action="store_true",
                        help="verify `token_done` after every chunk, which forces a sync per chunk")
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=8)
    return parser.parse_args()


def calc_diff(x, y):
    x, y = x.astype("float32"), y.astype("float32")
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def make_routed_layout(num_tokens, topk, seed=0):
    """Build the unzipped layout of a real routing, i.e. what dispatch + unzip produce.

    Every unduplicated token appears once under each of its top-k experts, and each
    expert's region is padded to `ALIGNMENT` rows as the GEMM requires.
    """
    rng = np.random.default_rng(seed)
    route = np.argsort(rng.random((num_tokens, E)), axis=1)[:, :topk]

    expert_tokens = [np.sort(np.where((route == e).any(axis=1))[0]) for e in range(E)]
    counts = [len(tokens) for tokens in expert_tokens]
    aligned = [(c + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT for c in counts]
    m_start = np.cumsum([0] + aligned[:-1]).tolist()

    # `-1` marks the padded rows, which no chunk ever covers
    row_to_token = np.full(sum(aligned), -1, dtype=np.int32)
    m_indices = np.zeros(sum(aligned), dtype=np.int32)
    for expert_idx in range(E):
        start, count = m_start[expert_idx], counts[expert_idx]
        row_to_token[start: start + count] = expert_tokens[expert_idx]
        m_indices[start: start + aligned[expert_idx]] = expert_idx

    return counts, m_start, row_to_token, m_indices


def make_task_queue(counts, m_start, ready, seed=0):
    """The chunk arrival queue: one `[expert_idx, m_start, m_size, ready]` row per task.

    A real DeepEP dispatch appends a row when a chunk lands, so the expert order is
    random while chunks of one expert stay ordered.
    """
    per_expert_tasks = []
    for expert_idx, (count, start) in enumerate(zip(counts, m_start)):
        per_expert_tasks.append([
            [expert_idx, start + offset, min(CHUNK, count - offset), ready]
            for offset in range(0, count, CHUNK)
        ])

    # Interleave experts randomly, keeping each expert's chunks in order
    rng = np.random.default_rng(seed)
    cursors, remaining, queue = [0] * E, [len(tasks) for tasks in per_expert_tasks], []
    while sum(remaining) > 0:
        candidates = [i for i in range(E) if remaining[i] > 0]
        i = candidates[rng.integers(len(candidates))]
        queue.append(per_expert_tasks[i][cursors[i]])
        cursors[i] += 1
        remaining[i] -= 1

    assert len(queue[0]) == deep_gemm.num_chunk_task_fields
    return paddle.to_tensor(queue, dtype="int32")


def reference(x, w_gateup, w_down, probs, m_indices, correctness=True):
    o1 = paddle.empty([x.shape[0], 2 * I], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x, w_gateup, o1, m_indices)

    if correctness:
        gate, up = o1[:, :I].float(), o1[:, I:].float()
        o2 = (F.silu(gate) * up * probs.unsqueeze(-1)).astype("bfloat16")
    else:
        # 仅模拟 swiglu 融合算子的访存带宽，用于测试性能
        gate, up = o1[:x.shape[0] // 2], o1[x.shape[0] // 2:]
        o2 = (gate + up).reshape([x.shape[0], I])

    o3 = paddle.empty([x.shape[0], H], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2, w_down, o3, m_indices)
    return o1, o2, o3


def reference_gateup_chunk(buffers, tasks):
    x, w_gateup, o1 = buffers[0], buffers[1], buffers[4]
    for expert_idx, m_start, m_size, _ in tasks:
        deep_gemm.bf16_gemm_nn(
            x[m_start : m_start + m_size],
            w_gateup[expert_idx],
            o1[m_start : m_start + m_size],
        )


def compute_chunk(task_idx, buffers, task_queue):
    """The (gateup, swiglu, down, signal) group of one chunk, all claiming the same task."""
    x, w_gateup, w_down, probs, o1, o2, o3, row_to_token, token_done = buffers
    deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx)
    deep_gemm.chunk_weighted_swiglu(o1, probs, o2, task_queue, task_idx)
    deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
    deep_gemm.chunk_signal_token_done(row_to_token, token_done, task_queue, task_idx)


def main():
    args = parse_args()
    assert not (args.check_signal and args.arrival != "ready"), (
        "--check-signal syncs per chunk, which defeats the point of an async arrival"
    )

    counts, m_start, row_to_token_np, m_indices_np = make_routed_layout(args.num_tokens, args.topk)
    m_total = len(row_to_token_np)
    print("arrival:", args.arrival, "| interval:", args.interval_ms,
          "ms | check_signal:", args.check_signal)
    print("tokens_per_expert:", counts)
    print("num_unzipped_tokens:", m_total, "num_tokens:", args.num_tokens, "topk:", args.topk)

    x = paddle.randn([m_total, H], "bfloat16")
    w_gateup = paddle.randn([E, H, 2 * I], "bfloat16") * 0.02
    w_down = paddle.randn([E, I, H], "bfloat16") * 0.02
    probs = paddle.rand([m_total], "float32")
    row_to_token = paddle.to_tensor(row_to_token_np)
    m_indices = paddle.to_tensor(m_indices_np)

    deep_gemm.set_num_sms(NUM_SMS)
    o1_ref, o2_ref, o3_ref = reference(x, w_gateup, w_down, probs, m_indices)

    def make_buffers():
        # NaN, so that any row left untouched by the task queue shows up
        return (x, w_gateup, w_down, probs,
                paddle.full([m_total, 2 * I], float("nan"), "bfloat16"),
                paddle.full([m_total, I], float("nan"), "bfloat16"),
                paddle.full([m_total, H], float("nan"), "bfloat16"),
                row_to_token, paddle.zeros([args.num_tokens], "int32"))

    ################################# BASELINE #################################
    ready_queue = make_task_queue(counts, m_start, ready=1)
    num_tasks = ready_queue.shape[0]
    print("num_tasks:", num_tasks)
    buffers = make_buffers()

    # 直接预填充所有 ready=1，测试无等待情况下的性能
    for i in range(4):
        if i == 1:
            paddle.base.core.nvprof_start()
        paddle.base.core.nvprof_nvtx_push("ready")
        for task_idx in range(num_tasks):
            compute_chunk(task_idx, buffers, ready_queue)
        paddle.base.core.nvprof_nvtx_pop()

    # 测试 baseline group_gemm 的性能
    for i in range(3):
        paddle.base.core.nvprof_nvtx_push("baseline_group_gemm")
        reference(x, w_gateup, w_down, probs, m_indices, correctness=False)
        paddle.base.core.nvprof_nvtx_pop()

    # 测试分别调用 baseline gemm 的性能
    tasks = ready_queue.tolist()
    for i in range(3):
        paddle.base.core.nvprof_nvtx_push("baseline_gemm")
        reference_gateup_chunk(buffers, tasks)
        paddle.base.core.nvprof_nvtx_pop()

    ################################# OVERLAP ##################################
    # The queue under test, plus the CPU-side view of it for the `cpu` mode
    interval_s = args.interval_ms / 1e3
    task_queue = (ready_queue if args.arrival == "ready"
                  else make_task_queue(counts, m_start, ready=0))
    queue_rows = task_queue.tolist()

    buffers = make_buffers()
    o1, o2, o3, token_done = buffers[4], buffers[5], buffers[6], buffers[8]
    expected = np.zeros([args.num_tokens], dtype=np.int32)

    # 模拟独立的通信流和计算流，与默认流分开，否则后面写 task_queue 会死锁
    comm_stream, compute_stream = paddle.cuda.Stream(), paddle.cuda.Stream()
    event = paddle.device.Event()
    event.record()
    comm_stream.wait_event(event)
    compute_stream.wait_event(event)

    # 发射计算流算子
    with paddle.device.stream_guard(compute_stream):
        paddle.base.core.nvprof_nvtx_push("compute")
        for task_idx in range(num_tasks):
            paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
            compute_chunk(task_idx, buffers, task_queue)
            paddle.base.core.nvprof_nvtx_pop()

            if args.check_signal:
                # Verify the counters grow exactly as the chunks complete
                paddle.device.synchronize()
                _, chunk_start, chunk_size, _ = queue_rows[task_idx]
                expected[row_to_token_np[chunk_start: chunk_start + chunk_size]] += 1
                got = token_done.numpy()
                assert np.array_equal(got, expected), (
                    f"token_done mismatch after task {task_idx}: {np.flatnonzero(got != expected)[:8]}"
                )
        paddle.base.core.nvprof_nvtx_pop()

    # 模拟通信流给 task_queue 异步写信号
    if args.arrival == "gpu":
        with paddle.device.stream_guard(comm_stream):
            deep_gemm.simulate_chunk_arrival(task_queue, int(interval_s * 1e9))
    elif args.arrival == "cpu":
        one = paddle.ones([1], "int32")
        paddle.base.core.nvprof_nvtx_push("cpu_arrival")
        for task_idx in range(num_tasks):
            paddle.base.core.nvprof_nvtx_push("sleep")
            time.sleep(interval_s)
            paddle.base.core.nvprof_nvtx_pop()
            task_queue[task_idx, 3] = one
        paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()

    ################################# VALIDATE #################################
    # Only the real rows are compared: the padded rows are computed by the reference but
    # zeroed (o2) or left as the down GEMM's garbage (o3) by the chunk path
    real_rows = paddle.to_tensor(np.flatnonzero(row_to_token_np >= 0).astype(np.int32))
    for name, out, ref, tol in (("o1", o1, o1_ref, 0.0), ("o2", o2, o2_ref, 1e-6), ("o3", o3, o3_ref, 1e-6)):
        diff = calc_diff(out.index_select(real_rows), ref.index_select(real_rows))
        print(f"{name}: diff={diff:.3e}")
        assert diff <= tol, f"{name} mismatch: {diff}"

    # Every token must have been counted by exactly its own number of experts
    final = token_done.numpy()
    print("token_done: min:", final.min(), "max:", final.max(),
          "| padded rows:", int((row_to_token_np < 0).sum()), "(never signalled)")
    assert np.array_equal(final, np.full([args.num_tokens], args.topk, dtype=np.int32))

    print("PASSED")


if __name__ == "__main__":
    main()
