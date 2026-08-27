"""Chunk-wise MoE FFN: correctness of task claiming, spin-wait and token completion.

Usage:
    python tests_overlap/test_chunk.py                        # all chunks ready up-front
    python tests_overlap/test_chunk.py --arrival cpu          # CPU feeds `ready`
    python tests_overlap/test_chunk.py --check-signal         # verify the signals chunk by chunk
"""

import argparse
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
EP = 16
SEQLEN = 16384
TOPK = 8

CHUNK = 4096
NUM_SMS = 96
ALIGNMENT = 128
GROUP = 64          # the gate/up interleave granularity required by the fused epilogue
FUSE_SWIGLU = True  # fuse the SwiGLU into the gate-up epilogue instead of a second kernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrival", choices=["ready", "cpu"], default="ready",
                        help="who marks the chunks as arrived: nobody (all ready) or the CPU")
    parser.add_argument("--interval-ms", type=float, default=0.5,
                        help="delay between two arrivals, for the `cpu` mode")
    parser.add_argument("--check-signal", action="store_true",
                        help="verify `token_done` and `zip_task_queue` after every chunk, "
                             "which forces a sync per chunk")
    return parser.parse_args()


def calc_diff(x, y):
    x, y = x.astype("float32"), y.astype("float32")
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def make_routed_layout():
    """
    模拟在一个 EP 组内, 每个 rank 有 E 个专家的情况下, rank 0 收到的 dispatch+unzip 后的 token.

    Every unduplicated token appears once under each of its top-k experts, and each
    expert's region is padded to `ALIGNMENT` rows as the GEMM requires.
    """
    # 模拟全局 token 打分
    scores = paddle.randn([EP * SEQLEN, EP * E])
    scores += paddle.randn([EP * E]) * 0.1  # add some system bias to experts
    _, topk_indices = scores.topk(TOPK)
    topk_indices = topk_indices.cast("int32").sort()

    # 只保留命中 rank0 的 token
    topk_hit = topk_indices < E
    token_hit = topk_hit.any(axis=1).nonzero().squeeze(1)

    topk_indices[~topk_hit] = -1
    topk_indices = topk_indices.gather(token_hit, axis=0)

    # Note:
    # 1) topk_indices[num_recv_tokens, topk] 是通信 kernel 内部维护的一个状态,
    #    在 dispatch 完成前处于离散、不完整状态, 因此对于计算 kernel 来说不可见,
    #    计算这边只能靠通信冗余更新的 num_valid_topk 得知一个 token 有几个本地专家;
    # 2) topk_indices 的顺序与 unzipped_tokens 中的顺序没有必然关系, 虽然实际上
    #    unzipped_tokens 中的 token 会相对有序, 但通信的不确定性让其顺序无法保证,
    #    因此下面 atomic_to_zip 测试的也是最极端的随机打乱的情况.

    tokens_per_expert = paddle.sum(
        paddle.arange(E, dtype="int32")[:, None] == topk_indices.flatten(),
        axis=1,
    ).tolist()

    # 构造计算可见的相关 meta
    m_start, atomic_to_zip, m_indices = [0], [], []

    for expert_idx, n in enumerate(tokens_per_expert):
        n_aligned = (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        m_start.append(m_start[-1] + n_aligned)
        m_indices.append(paddle.full([n_aligned], expert_idx, dtype="int32"))

        # 选出属于 expert_idx 的 token 并打乱顺序
        token_idxs = (topk_indices == expert_idx).any(axis=1).nonzero().squeeze(1)
        assert len(token_idxs) == n
        perm = paddle.randperm(token_idxs.shape[0])
        atomic_to_zip.append(token_idxs.astype("int32")[perm])
        if n % ALIGNMENT > 0:
            atomic_to_zip.append(paddle.full([ALIGNMENT - n % ALIGNMENT], -1, dtype="int32"))

    m_start = m_start[:-1]
    atomic_to_zip = paddle.concat(atomic_to_zip)
    m_indices = paddle.concat(m_indices)

    return tokens_per_expert, topk_indices, m_start, atomic_to_zip, m_indices


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


def interleave_gateup(w_gateup):
    """`[gate | up]` -> `[gate[0:64], up[0:64], ...]` per expert, what the fused epilogue needs."""
    perm = []
    for start in range(0, I, GROUP):
        perm.extend(range(start, start + GROUP))
        perm.extend(range(I + start, I + start + GROUP))
    perm = paddle.to_tensor(np.asarray(perm, dtype=np.int32))
    return w_gateup.index_select(perm, axis=2).contiguous()


def split_gate_up(o1):
    """The gate/up halves of `o1`, whatever column order the weight was in."""
    if FUSE_SWIGLU:
        blocks = o1.reshape([o1.shape[0], I // GROUP, 2, GROUP])
        return blocks[:, :, 0].reshape([-1, I]), blocks[:, :, 1].reshape([-1, I])
    return o1[:, :I], o1[:, I:]


def reference(x, w_gateup, w_down, probs, m_indices, perf=None):
    if perf is not None:
        # 模拟调用 zip/unzip, 实际结果无用
        topk_indices, tokens_per_expert = perf
        _, rowmap, unzipped_probs, _ = paddle.nn.functional.moe_permute(
            x[:len(topk_indices)],
            None,  # scale
            topk_indices,
            paddle.empty(topk_indices.shape, dtype="float32"),  # topk_probs
            padding_alignment=ALIGNMENT,
            num_experts=E,
            tokens_per_expert=tokens_per_expert,
        )
        paddle.nn.functional.moe_unpermute(
            x,
            rowmap,
            topk_indices,
            unzipped_probs,
            total_zipped_tokens=len(topk_indices),
            num_experts=E,
        )

    o1 = paddle.empty([x.shape[0], 2 * I], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x, w_gateup, o1, m_indices)

    if perf is not None:
        # 仅模拟 swiglu 融合算子的访存带宽，用于测试性能
        gate, up = o1[:x.shape[0] // 2], o1[x.shape[0] // 2:]
        o2 = (gate + up).reshape([x.shape[0], I])
    else:
        gate, up = split_gate_up(o1)
        gate, up = gate.float(), up.float()
        o2 = (F.silu(gate) * up * probs.unsqueeze(-1)).astype("bfloat16")

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
    (x, w_gateup, w_down, probs, o1, o2, o3,
     atomic_to_zip, num_valid_topk, token_done, zip_task_queue, zip_queue_tail) = buffers
    if FUSE_SWIGLU:
        deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx, o2=o2, probs=probs)
    else:
        deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx)
        deep_gemm.chunk_weighted_swiglu(o1, probs, o2, task_queue, task_idx)
    deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
    deep_gemm.chunk_signal_token_done(atomic_to_zip, num_valid_topk, token_done,
                                      zip_task_queue, zip_queue_tail, task_queue, task_idx)


def reset_signals(buffers):
    """Clear the completion state, as the done kernels only ever count up."""
    token_done, zip_task_queue, zip_queue_tail = buffers[9], buffers[10], buffers[11]
    token_done.zero_()
    zip_task_queue.fill_(-1)
    zip_queue_tail.zero_()


def main():
    args = parse_args()
    assert not (args.check_signal and args.arrival != "ready"), (
        "--check-signal syncs per chunk, which defeats the point of an async arrival"
    )

    tokens_per_expert, topk_indices, m_start, atomic_to_zip, m_indices = make_routed_layout()
    num_recv_tokens = len(topk_indices)
    m_total = len(atomic_to_zip)
    num_valid_topk = paddle.sum(topk_indices >= 0, axis=1).astype("int32")
    print("arrival:", args.arrival, "| interval:", args.interval_ms,
          "ms | check_signal:", args.check_signal)
    print("num_recv_tokens:", num_recv_tokens)
    print("tokens_per_expert:", tokens_per_expert)
    print("num_unzipped_tokens:", m_total, "seq_len:", SEQLEN, "topk:", TOPK)

    x = paddle.randn([m_total, H], "bfloat16")
    w_gateup = paddle.randn([E, H, 2 * I], "bfloat16") * 0.02
    w_down = paddle.randn([E, I, H], "bfloat16") * 0.02
    probs = paddle.rand([m_total], "float32")

    if FUSE_SWIGLU:
        w_gateup = interleave_gateup(w_gateup)

    deep_gemm.set_num_sms(NUM_SMS)
    o1_ref, o2_ref, o3_ref = reference(x, w_gateup, w_down, probs, m_indices)

    def make_buffers():
        # NaN, so that any row left untouched by the task queue shows up
        return (x, w_gateup, w_down, probs,
                paddle.full([m_total, 2 * I], float("nan"), "bfloat16"),
                paddle.full([m_total, I], float("nan"), "bfloat16"),
                paddle.full([m_total, H], float("nan"), "bfloat16"),
                atomic_to_zip, num_valid_topk,
                paddle.zeros([num_recv_tokens], "int32"),
                paddle.full([num_recv_tokens], -1, "int32"),
                paddle.zeros([1], "int32"))

    ################################# BASELINE #################################
    ready_queue = make_task_queue(tokens_per_expert, m_start, ready=1)
    num_tasks = ready_queue.shape[0]
    print("num_tasks:", num_tasks)
    buffers = make_buffers()
    tasks = ready_queue.tolist()

    # warmup
    compute_chunk(0, buffers, ready_queue)
    reference(x, w_gateup, w_down, probs, m_indices, perf=(topk_indices, tokens_per_expert))
    reference_gateup_chunk(buffers, tasks)
    paddle.base.core.nvprof_start()

    # 直接预填充所有 ready=1，测试无等待情况下的性能
    for i in range(3):
        reset_signals(buffers)
        paddle.base.core.nvprof_nvtx_push("chunk_all_ready")
        for task_idx in range(num_tasks):
            compute_chunk(task_idx, buffers, ready_queue)
        paddle.base.core.nvprof_nvtx_pop()

    # 测试 baseline group_gemm 的性能
    for i in range(3):
        paddle.base.core.nvprof_nvtx_push("baseline_group_gemm")
        reference(x, w_gateup, w_down, probs, m_indices,
                  perf=(topk_indices, tokens_per_expert))
        paddle.base.core.nvprof_nvtx_pop()

    # 测试分别调用 baseline gateup 的性能
    for i in range(3):
        paddle.base.core.nvprof_nvtx_push("baseline_gateup_chunk")
        reference_gateup_chunk(buffers, tasks)
        paddle.base.core.nvprof_nvtx_pop()

    ################################# OVERLAP ##################################
    # The queue under test, plus the CPU-side view of it for the `cpu` mode
    interval_s = args.interval_ms / 1e3
    task_queue = (ready_queue if args.arrival == "ready"
                  else make_task_queue(tokens_per_expert, m_start, ready=0))
    queue_rows = task_queue.tolist()

    buffers = make_buffers()
    o1, o2, o3 = buffers[4], buffers[5], buffers[6]
    token_done, zip_task_queue, zip_queue_tail = buffers[9], buffers[10], buffers[11]
    expected = np.zeros([num_recv_tokens], dtype=np.int32)
    expected_topk = num_valid_topk.numpy()

    # 计算流与默认流分开，否则后面写 task_queue 会死锁
    compute_stream = paddle.cuda.Stream()
    event = paddle.device.Event()
    event.record()
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
                expected[atomic_to_zip[chunk_start : chunk_start + chunk_size]] += 1
                got = token_done.numpy()
                assert np.array_equal(got, expected), (
                    f"token_done mismatch after task {task_idx}: {np.flatnonzero(got != expected)[:8]}"
                )

                # The queue must hold exactly the tokens whose experts are all done
                done_tokens = np.flatnonzero(expected == expected_topk)
                tail = int(zip_queue_tail.numpy()[0])
                pushed = zip_task_queue.numpy()[:tail]
                assert tail == len(done_tokens), (
                    f"zip_queue_tail {tail} != {len(done_tokens)} after task {task_idx}"
                )
                assert np.array_equal(np.sort(pushed), done_tokens), (
                    f"zip_task_queue mismatch after task {task_idx}"
                )
        paddle.base.core.nvprof_nvtx_pop()

    # 模拟通信流给 task_queue 异步写信号
    if args.arrival == "cpu":
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
    valid_rows = atomic_to_zip.nonzero()
    for name, out, ref, tol in (("o1", o1, o1_ref, 0.0), ("o2", o2, o2_ref, 1e-6), ("o3", o3, o3_ref, 1e-6)):
        diff = calc_diff(out.index_select(valid_rows), ref.index_select(valid_rows))
        print(f"{name}: diff={diff:.3e}")
        assert diff <= tol, f"{name} mismatch: {diff}"

    # Every token must have been counted by exactly its own number of experts
    final = token_done.numpy()
    print("token_done: min:", final.min(), "max:", final.max(),
          "| padded rows:", int((atomic_to_zip < 0).sum()), "(never signalled)")
    assert np.array_equal(final, expected_topk)

    # And every token must have been pushed to `zip` exactly once, with no hole left
    tail = int(zip_queue_tail.numpy()[0])
    pushed = zip_task_queue.numpy()
    print("zip_task_queue: pushed:", tail, "of", num_recv_tokens)
    assert tail == num_recv_tokens, f"zip_queue_tail {tail} != {num_recv_tokens}"
    assert np.array_equal(np.sort(pushed), np.arange(num_recv_tokens)), "zip_task_queue mismatch"

    print("PASSED")


if __name__ == "__main__":
    main()
