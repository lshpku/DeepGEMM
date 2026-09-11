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

USE_UE8M0 = True  # 当前只支持 ue8m0
QUANT_BLOCK_SIZE = 512


class Result(NamedTuple):
    x: tuple[Tensor, Tensor]
    o1: Tensor
    o2: tuple[Tensor, Tensor]
    o3: Tensor
    out: Tensor
    do2: Tensor
    do1: Tensor
    dx: Tensor
    drecv_x: Tensor
    drecv_probs: Tensor
    o2_bwd: Tensor

    def __getitem__(self, key: str):
        return getattr(self, key)


def reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices,
              w_gateup, w_down, w_gateup_t, w_down_t, dout):
    topk_indices = topk_indices.cast("int32")

    ################################# Forward ##################################

    # 前向通信发过来的已经是 fp8 的 x
    recv_x_fp8, recv_scale = recv_x

    x_fp8, rowmap, unzipped_probs, x_scale = paddle.nn.functional.moe_permute(
        recv_x_fp8,
        recv_scale,
        topk_indices,
        recv_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
        using_ue8m0_scale=USE_UE8M0,
    )
    x = (x_fp8, x_scale)

    # NOTES: baseline 里面的 scale 是通信完才 transpose, 因为 baseline 的 DeepEP 不支持发送
    #        transpose 的 scale, 不需要把这里的 transpose 套到 chunk 方案上
    # NOTES: paddle 的一些 permute/quant 函数输出底层已经是 transpose 的, 但是为了可读性和安全性,
    #        这里显式调用一下 transpose, 如果底层已经是 transpose 的下面就是空操作
    x_scale = x_scale.T.contiguous().T

    o1 = paddle.empty([len(x_fp8), 2 * I], dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((x_fp8, x_scale), w_gateup_t, o1, m_indices)

    o2_fp8, o2_scale = paddlefleet_ops.fuse_weighted_swiglu_fp8_quant(
        o1, unzipped_probs, using_pow2_scaling=True, use_ue8m0=USE_UE8M0)

    o2 = (o2_fp8, o2_scale)
    o2_scale = o2_scale.T.contiguous().T

    o3 = paddle.empty([len(x_fp8), H], dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((o2_fp8, o2_scale), w_down_t, o3, m_indices)

    out, _ = paddle.nn.functional.moe_unpermute(
        o3,
        rowmap,
        topk_indices,
        unzipped_probs,
        total_zipped_tokens=len(topk_indices),
        num_experts=E,
    )

    ################################# Backward #################################

    do3, _, _, _ = paddle.nn.functional.moe_permute(
        dout,
        None,  # scale
        topk_indices,
        recv_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    # 反向通信发的 dout 是 bf16, 需要本地自己 quant, 因为 wgrad 需要精确的 bf16
    do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        do3,
        output_scale_transpose=True,
        quant_method="1x128",
        input_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )

    do2 = paddle.empty(o2_fp8.shape, dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((do3_fp8, do3_scale.T), w_down, do2, m_indices)

    # o2_bwd 的精度和前向的 o2 不同，但这不影响主干梯度，只影响 wgrad
    do1, dprobs, o2_bwd = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
        o1, do2, unzipped_probs.unsqueeze(-1))

    do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        do1,
        output_scale_transpose=True,
        quant_method="1x128",
        input_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )

    dx = paddle.empty(x_fp8.shape, dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((do1_fp8, do1_scale.T), w_gateup, dx, m_indices)

    drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
        dx,
        rowmap,
        topk_indices,
        dprobs,
        total_zipped_tokens=len(topk_indices),
        num_experts=E,
    )

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs, o2_bwd)


def compute_chunk(recv_x, recv_probs, topk_indices, w_gateup, w_down, dout,
                  atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                  task_queue, task_queue_bwd):
    num_valid_topk = (topk_indices != -1).sum(axis=-1, dtype="int32")

    ################################# Forward ##################################

    ################################# Backward #################################

    return None


def quant_input(x):
    """对于 hidden_states, 在 hidden 维上使用 128 分块量化."""
    x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        quant_method="1x128",
        output_scale_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )
    # quant 算子因为历史原因会将长度向 4 对齐, 这里截断不影响连续性
    scale = scale[:x.shape[0]]
    assert x_fp8.shape == x.shape
    assert scale.shape == [x.shape[0], x.shape[1] // QUANT_BLOCK_SIZE]
    assert x_fp8.is_contiguous()
    assert scale.is_contiguous()
    return x_fp8, scale


def quant_weight(w, transpose=False):
    """对于 weight, 在 n 维上使用 128 分块量化."""
    # quant 算子只接受 list 输入，这里手动切成列表
    expert_weight_list = list(w)
    if transpose:
        w_fp8, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=USE_UE8M0,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[2], w.shape[1]]
        assert scale.shape == [w.shape[0] * w.shape[2], w.shape[1] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    else:
        w_fp8, scale = paddlefleet_ops.fuse_stack_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=USE_UE8M0,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[1], w.shape[2]]
        assert scale.shape == [w.shape[0] * w.shape[1], w.shape[2] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    # quant 算子输出把专家维铺平了，需要重新展开
    w_fp8 = w_fp8.reshape([w.shape[0], -1, w_fp8.shape[1]])
    scale = scale.reshape([w.shape[0], -1, scale.shape[1]])
    # ue8m0 要求 scale 最后两维 transpose
    if USE_UE8M0:
        scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
    return w_fp8, scale


def dequant(x, scale):
    if USE_UE8M0:
        scale = 2.0 ** (scale.contiguous().view("int8").cast("int32") - 127)
    return x.float() * scale.repeat_interleave(128, axis=-1)


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
    # w_gateup_ref = deinterleave_gateup(w_gateup) if INTERLEAVED else w_gateup

    recv_x_quant = quant_input(recv_x)

    # fp8 的性能对 layout 敏感, 因此 fp8 需要维护两套 layout 的权重, 不像 bf16 只需要改算子参数
    w_gateup_quant = quant_weight(w_gateup, transpose=False)
    w_down_quant = quant_weight(w_down, transpose=False)
    w_gateup_t_quant = quant_weight(w_gateup, transpose=True)
    w_down_t_quant = quant_weight(w_down, transpose=True)

    ################################# Baseline #################################

    refs = reference(recv_x_quant, recv_probs, topk_indices, tokens_per_expert, m_indices,
                     w_gateup_quant, w_down_quant, w_gateup_t_quant, w_down_t_quant, dout)

    for name, tensor in refs._asdict().items():
        if isinstance(tensor, tuple):
            tensor = dequant(*tensor)
        print(name, ":", tensor)

    ################################## Chunk ###################################


    ################################# Validate #################################


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
