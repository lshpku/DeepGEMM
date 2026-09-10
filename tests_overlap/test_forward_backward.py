import time
import argparse
import numpy as np
from typing import NamedTuple

import paddle
from paddle import Tensor
import paddle.nn.functional as F

paddle.cuda.set_device(1)
paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

import deep_gemm
print("deep_gemm:", deep_gemm.__path__)

from utils import make_deepep_layout, make_atomic_layout

# 使用特别编译的注释掉 deep_gemm 的版本, 不然会和我们的头文件冲突
import paddlefleet_ops
assert not paddlefleet_ops._DEEP_GEMM_AVAILABLE

E = 16
H = 4096
I = 2048
EP = 16
SEQLEN = 16384
TOPK = 8

CHUNK = 4096
NUM_SMS = 100
ALIGNMENT = 128
PRECISE_SWIGLU = False
INTERLEAVED = False


class Result(NamedTuple):
    x: Tensor
    o1: Tensor
    o2: Tensor
    o3: Tensor
    out: Tensor
    do2: Tensor
    do1: Tensor
    dx: Tensor
    drecv_x: Tensor
    drecv_probs: Tensor
    o2_bwd: Tensor = None

    def __getitem__(self, key: str):
        return getattr(self, key)


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
    """`[gate | up]` -> `[gate[0], up[0], gate[1], up[1], ...]` per expert, fully interleaved."""
    perm = np.empty([2 * I], dtype=np.int32)
    perm[0::2] = np.arange(I)
    perm[1::2] = np.arange(I) + I
    return w_gateup.index_select(paddle.to_tensor(perm), axis=2).contiguous()


def deinterleave_gateup(w_gateup):
    return paddle.concat([w_gateup[..., 0::2], w_gateup[..., 1::2]], axis=-1)


def reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices, w_gateup, w_down, dout):
    topk_indices = topk_indices.cast("int32")

    ################################# Forward ##################################

    x, rowmap, unzipped_probs, _ = paddle.nn.functional.moe_permute(
        recv_x,
        None,  # scale
        topk_indices,
        recv_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    o1 = paddle.empty([len(x), 2 * I], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x, w_gateup, o1, m_indices)

    # gate, up = o1.chunk(2, axis=-1)
    # gate, up = gate.float(), up.float()
    # o2 = ((gate * F.sigmoid(gate)) * up * unzipped_probs.unsqueeze(-1)).cast("bfloat16")
    o2 = paddlefleet_ops.fused_swiglu_scale(o1, unzipped_probs)

    o3 = paddle.empty([len(x), H], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2, w_down, o3, m_indices)

    out, _ = paddle.nn.functional.moe_unpermute(
        o3,
        rowmap,
        topk_indices,
        unzipped_probs,
        total_zipped_tokens=len(topk_indices),
        num_experts=E,
    )

    ################################# Backward #################################

    do3, rowmap, _, _ = paddle.nn.functional.moe_permute(
        dout,
        None,  # scale
        topk_indices,
        recv_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    do2 = paddle.empty_like(o2)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do3, w_down, do2, m_indices)

    do1, dprobs = paddlefleet_ops.fused_swiglu_scale_bwd(o1, unzipped_probs, do2)

    dx = paddle.empty_like(x)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do1, w_gateup, dx, m_indices)

    drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
        dx,
        rowmap,
        topk_indices,
        dprobs,
        total_zipped_tokens=len(topk_indices),
        num_experts=E,
    )

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs)


def compute_chunk(recv_x, recv_probs, topk_indices, w_gateup, w_down, dout,
                  atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                  task_queue, task_queue_bwd):
    num_valid_topk = (topk_indices != -1).sum(axis=-1, dtype="int32")

    ################################# Forward ##################################

    # 模拟通信已经给出 unzip 的结果
    x = deep_gemm.token_gather(recv_x, atomic_to_zip)

    # paddle scatter 会将 -1 的下标映射到最后一格, 需要多分配一格来接住这些无效值
    probs_pad = paddle.zeros([len(x) + 1], dtype="float32")
    probs_pad.scatter_(zip_to_atomic.flatten(), recv_probs.flatten())
    probs = probs_pad[:-1]

    o1 = paddle.full([len(x), 2 * I], float("nan"), dtype="bfloat16")
    o2 = paddle.full([len(x), I], float("nan"), dtype="bfloat16")
    o3 = paddle.full([len(x), H], float("nan"), dtype="bfloat16")
    out = paddle.full([len(recv_probs), H], float("nan"), dtype="bfloat16")

    token_done = paddle.zeros([len(recv_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_probs)], dtype="int32")

    paddle.base.core.nvprof_nvtx_push("forward")
    for task_idx in range(len(task_queue)):
        deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx)
        deep_gemm.chunk_weighted_swiglu(o1, probs, o2, task_queue, task_idx,
                                        precise=PRECISE_SWIGLU, interleaved=INTERLEAVED)
        deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
        deep_gemm.chunk_zip(o3, out, atomic_to_zip, zip_to_atomic, topk_indices, num_valid_topk,
                            token_done, zip_done, task_queue, task_idx, CHUNK)
    paddle.base.core.nvprof_nvtx_pop()

    ################################# Backward #################################

    do3 = deep_gemm.token_gather(dout, atomic_to_zip_bwd)

    dx = paddle.full_like(x, float("nan"))
    do1 = paddle.full_like(o1, float("nan"))
    do2 = paddle.full_like(o2, float("nan"))
    drecv_x = paddle.full_like(out, float("nan"))
    drecv_probs = paddle.zeros_like(recv_probs)  # 无效位预先填0
    o2_bwd = paddle.full_like(o2, float("nan"))

    token_done = paddle.zeros([len(recv_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_probs)], dtype="int32")

    paddle.base.core.nvprof_nvtx_push("backward")
    for task_idx in range(len(task_queue_bwd)):
        paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
        deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, task_queue_bwd, task_idx)
        deep_gemm.chunk_weighted_swiglu_grad(
            o1, probs, do2, o2_bwd, do1, drecv_probs, atomic_to_zip_bwd, zip_to_atomic,
            topk_indices, task_queue_bwd, task_idx, CHUNK, precise=PRECISE_SWIGLU,
            interleaved=INTERLEAVED)
        deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, task_queue_bwd, task_idx)
        deep_gemm.chunk_zip(dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, topk_indices,
                            num_valid_topk, token_done, zip_done, task_queue_bwd, task_idx, CHUNK)
        paddle.base.core.nvprof_nvtx_pop()
    paddle.base.core.nvprof_nvtx_pop()

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs, o2_bwd)


def get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip):
    """将 atomic 序的 o1/o2/o3 等转换为参考序的映射表, padding 保留原位."""
    perm = paddle.arange(m_start[-1])
    for n, offset in zip(tokens_per_expert, m_start):
        perm[offset : offset + n] = atomic_to_zip[offset : offset + n].argsort() + offset
    return perm


def check(x, y):
    diff = (x.float() - y.float()).abs()
    avg, max = float(diff.mean()), float(diff.max())
    banner = (" " + "-" * 40) if (avg or max) else ""
    avg = "0" if avg == 0 else f"{avg:e}"
    max = "0" if max == 0 else f"{max:e}"
    return f"avg: {avg} max: {max}" + banner


def main():
    deep_gemm.set_num_sms(NUM_SMS)

    topk_indices, tokens_per_expert, m_start, m_indices = make_deepep_layout(
        EP, SEQLEN, E, TOPK, ALIGNMENT)

    recv_x = paddle.randn([len(topk_indices), H], dtype="bfloat16")
    dout = paddle.randn_like(recv_x)

    recv_probs = paddle.randn(topk_indices.shape)
    w_gateup = paddle.randn([E, H, 2 * I], dtype="bfloat16") * 0.02
    w_down = paddle.randn([E, I, H], dtype="bfloat16") * 0.02
    w_gateup_ref = deinterleave_gateup(w_gateup) if INTERLEAVED else w_gateup

    ################################# Baseline #################################

    refs = reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices,
                     w_gateup_ref, w_down, dout)

    ################################## Chunk ###################################

    atomic_to_zip, zip_to_atomic = make_atomic_layout(topk_indices, tokens_per_expert, m_start)
    atomic_to_zip_bwd, zip_to_atomic_bwd = make_atomic_layout(
        topk_indices, tokens_per_expert, m_start)

    task_queue = make_task_queue(tokens_per_expert, m_start, ready=True, seed=0)
    task_queue_bwd = make_task_queue(tokens_per_expert, m_start, ready=True, seed=1)

    paddle.base.core.nvprof_start()
    outs = compute_chunk(recv_x, recv_probs, topk_indices, w_gateup, w_down, dout,
                         atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                         task_queue, task_queue_bwd)
    paddle.base.core.nvprof_stop()

    ################################# Validate #################################

    fwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip)
    bwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip_bwd)

    checks = [
        (("x", "o1", "o2", "o3"), fwd_perm),
        (("out",), None),
        (("do2", "do1", "dx"), bwd_perm),
        (("drecv_x", "drecv_probs"), None),
    ]

    for names, perm in checks:
        for name in names:
            out, ref = outs[name], refs[name]
            out = out[perm] if perm is not None else out
            out = deinterleave_gateup(out) if (INTERLEAVED and "o1" in name) else out
            print(f"{name}:", check(out, ref))

    print("o2_bwd vs ref:", check(outs["o2_bwd"][bwd_perm], refs["o2"]))
    print("o2_bwd vs fwd:", check(outs["o2_bwd"][bwd_perm], outs["o2"][fwd_perm]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precise-swiglu", action="store_true",
                        help="Use precise swiglu, only applies for unfused swiglu")
    parser.add_argument("--interleaved", action="store_true",
                        help="Use fully-interleaved w_gateup, only applies for unfused swiglu")
    args = parser.parse_args()

    PRECISE_SWIGLU = args.precise_swiglu
    INTERLEAVED = args.interleaved

    main()
