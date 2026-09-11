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

from utils import (
    make_deepep_layout, make_atomic_layout, make_task_queue, interleave_gateup,
    deinterleave_gateup, get_atomic_perm,
)

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
ORDERED_WGRAD = False


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
    w_gateup_grad: Tensor
    w_down_grad: Tensor
    o2_bwd: Tensor

    def __getitem__(self, key: str):
        return getattr(self, key)


def run_wgrad(tokens_per_expert, x, do1, w_gateup_grad, o2, do3, w_down_grad):
    ks_cpu = [(n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT for n in tokens_per_expert]
    grouped_layout = paddle.to_tensor(ks_cpu, dtype="int32")

    paddle.base.core.nvprof_nvtx_push("wgrad")
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        x, do1, w_gateup_grad, ks_cpu, grouped_layout, w_gateup_grad)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        o2, do3, w_down_grad, ks_cpu, grouped_layout, w_down_grad)
    paddle.base.core.nvprof_nvtx_pop()


def reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices,
              w_gateup, w_down, dout):
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

    ################################## Wgrad ###################################

    # wgrad 使用累加语义, 输入需要置 0, 实际执行的时候是预处理, 无成本
    w_gateup_grad = paddle.zeros(w_gateup.shape, dtype="float32")
    w_down_grad = paddle.zeros(w_down.shape, dtype="float32")

    run_wgrad(tokens_per_expert, x, do1, w_gateup_grad, o2, do3, w_down_grad)

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs,
                  w_gateup_grad, w_down_grad, o2)


def compute_chunk(recv_x, recv_probs, topk_indices, tokens_per_expert, m_start, w_gateup, w_down,
                  dout, atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
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

    ################################## Wgrad ###################################

    w_gateup_grad = paddle.zeros(w_gateup.shape, dtype="float32")
    w_down_grad = paddle.zeros(w_down.shape, dtype="float32")

    if ORDERED_WGRAD:
        m_start_gpu = paddle.to_tensor(m_start, dtype="int32")

        # x 从 recv_x 中解压, 这里使用 gather 并非最优性能, 因为重复读了 recv_x 的某些行
        ordered_to_zip = deep_gemm.sort_unzip_map(zip_to_atomic_bwd, m_start_gpu, len(x))
        x_wgrad = deep_gemm.token_gather(recv_x, ordered_to_zip)

        # do1/o2_bwd/do3 从反向 atomic 序的输入重排序为标准 unzip 序
        ordered_to_atomic = deep_gemm.sort_atomic_map(zip_to_atomic_bwd, m_start_gpu, len(x))
        do1_wgrad = deep_gemm.token_gather(do1, ordered_to_atomic)
        o2_wgrad = deep_gemm.token_gather(o2_bwd, ordered_to_atomic)
        do3_wgrad = deep_gemm.token_gather(do3, ordered_to_atomic)
    else:
        x_wgrad = deep_gemm.token_gather(recv_x, atomic_to_zip_bwd)
        do1_wgrad, o2_wgrad, do3_wgrad = do1, o2_bwd, do3

    run_wgrad(tokens_per_expert, x_wgrad, do1_wgrad, w_gateup_grad,
              o2_wgrad, do3_wgrad, w_down_grad)

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs,
                  w_gateup_grad, w_down_grad, o2_bwd)


def check(x, y):
    diff = (x.float() - y.float()).abs()
    avg, max = float(diff.mean()), float(diff.max())
    banner = (" " + ("-" if avg < 1e-3 else "X") * 40) if (avg or max) else ""
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

    task_queue = make_task_queue(tokens_per_expert, m_start, CHUNK, ready=True, seed=0)
    task_queue_bwd = make_task_queue(tokens_per_expert, m_start, CHUNK, ready=True, seed=1)

    paddle.base.core.nvprof_start()
    outs = compute_chunk(recv_x, recv_probs, topk_indices, tokens_per_expert, m_start,
                         w_gateup, w_down, dout, atomic_to_zip, zip_to_atomic,
                         atomic_to_zip_bwd, zip_to_atomic_bwd, task_queue, task_queue_bwd)
    paddle.base.core.nvprof_stop()

    ################################# Validate #################################

    fwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip)
    bwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip_bwd)

    checks = [
        (("x", "o1", "o2", "o3"), fwd_perm),
        (("out",), None),
        (("do2", "do1", "dx", "o2_bwd"), bwd_perm),
        (("drecv_x", "drecv_probs", "w_gateup_grad", "w_down_grad"), None),
    ]

    for names, perm in checks:
        for name in names:
            out, ref = outs[name], refs[name]
            out = out[perm] if perm is not None else out
            if INTERLEAVED and ("o1" in name or "gateup" in name):
                out = deinterleave_gateup(out)
            print(f"{name}:", check(out, ref))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precise-swiglu", action="store_true",
                        help="Use precise swiglu, only applies for unfused swiglu")
    parser.add_argument("--interleaved", action="store_true",
                        help="Use fully-interleaved w_gateup, only applies for unfused swiglu")
    parser.add_argument("--ordered-wgrad", action="store_true",
                        help="Use standard deterministic order for wgrad")
    args = parser.parse_args()

    PRECISE_SWIGLU = args.precise_swiglu
    INTERLEAVED = args.interleaved
    ORDERED_WGRAD = args.ordered_wgrad

    main()
