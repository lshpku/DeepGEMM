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


def quant_wgrad_input(x):
    """沿 token 维 128 分块量化, 供 fp8 wgrad 使用.

    k_grouped_fp8_gemm_tn 的收缩维是 token 维 (a 的 shape 为 [sum_k, m]), 而 1D1D recipe 要求
    scale 沿收缩维分块, 即一个 scale 覆盖 (128 token, 1 channel), 和前向沿 hidden 维的 1x128
    量化不是同一份数据, 所以 wgrad 必须重新量化一次.
    """
    x_fp8_t, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        quant_method="1x128",
        input_transpose=True,
        return_transpose_only=True,
        output_scale_transpose=True,
        # NOTES: 这里只能给 fp32 scale, 由 deep_gemm 自己 pack 成 ue8m0 (kernel:
        #        pack_fp32_into_ue8m0), 因为 packed ue8m0 是按专家 4 个 K-block 一组打包的,
        #        要求每个专家的 k 向 gran_k*4=512 对齐, 当前 ALIGNMENT=128 不满足
        using_pow2_scale=True,
        using_ue8m0_scale=False,
    )
    assert x_fp8_t.shape == [x.shape[1], x.shape[0]]
    assert scale.shape == [x.shape[0] // 128, x.shape[1]]
    return x_fp8_t, scale


def to_mn_major(quant_pair):
    """[hidden, token] -> [token, hidden], 仅用于适配 k_grouped tn 的 MN-major 要求.

    这一步不属于方案成本: paddle 的量化算子沿最后一维分块, 所以沿 token 分块就必然把 token 放在
    最后一维输出; 真实方案里由自己的融合算子直接写出 [token, hidden] 的数据 + 沿 token 分块的
    scale, 单遍读写即可, 不存在这次转置.
    """
    x_fp8_t, scale = quant_pair
    return x_fp8_t.view("int8").T.contiguous().view("float8_e4m3fn"), scale


def reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices,
              w_gateup, w_down, w_gateup_t, w_down_t, dout):
    topk_indices = topk_indices.cast("int32")

    ################################# Forward ##################################

    paddle.base.core.nvprof_nvtx_push("forward")

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

    paddle.base.core.nvprof_nvtx_pop()

    ################################# Backward #################################

    paddle.base.core.nvprof_nvtx_push("backward")

    do3, _, _, _ = paddle.nn.functional.moe_permute(
        dout,
        None,  # scale
        topk_indices,
        recv_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    # 反向通信发的 dout 是 bf16, 需要本地自己 quant, 因为 wgrad 要用 bf16;
    # 这导致通信带宽翻倍, 而且在这里 quant 的成本比在通信前 quant 高, 因为有重复 token
    do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        do3,
        output_scale_transpose=True,
        quant_method="1x128",
        input_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )

    do2 = paddle.empty(o2_fp8.shape, dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((do3_fp8, do3_scale.T), w_down, do2, m_indices)

    # o2_bwd 的精度和前向的 o2 不同，但这与主干梯度无关，o2_bwd 只供 wgrad 使用
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

    paddle.base.core.nvprof_nvtx_pop()

    ################################## Wgrad ###################################

    ks_cpu = [(n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT for n in tokens_per_expert]
    grouped_layout = paddle.to_tensor(ks_cpu, dtype="int32")

    # 实际为预处理, 无成本
    w_gateup_grad = paddle.zeros(w_gateup[0].shape, dtype="float32")
    w_down_grad = paddle.zeros(w_down[0].shape, dtype="float32")

    paddle.base.core.nvprof_nvtx_push("wgrad")

    # 为了使用 bf16 的 wgrad，需要对 x_fp8 激活进行 dequant, 这是 baseline 最违和的地方,
    # 既没有真正提升精度还增加了计算量
    x_dequant = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale)

    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        x_dequant, do1, w_gateup_grad, ks_cpu, grouped_layout, w_gateup_grad)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        o2_bwd, do3, w_down_grad, ks_cpu, grouped_layout, w_down_grad)

    paddle.base.core.nvprof_nvtx_pop()

    ################################ FP8 Wgrad #################################

    # 用 paddle 的量化算子把 wgrad 的四个输入沿 token 维量化, 再喂给 fp8 的 k_grouped_gemm,
    # 用于对比 "多出来的量化" 和 "fp8 wgrad 省下来的时间"
    paddle.base.core.nvprof_nvtx_push("wgrad_fp8_quant")
    quants = [quant_wgrad_input(t) for t in (x_dequant, do1, o2_bwd, do3)]
    paddle.base.core.nvprof_nvtx_pop()

    x_q, do1_q, o2_q, do3_q = [to_mn_major(q) for q in quants]
    w_gateup_grad_fp8 = paddle.zeros(w_gateup[0].shape, dtype="float32")
    w_down_grad_fp8 = paddle.zeros(w_down[0].shape, dtype="float32")

    paddle.base.core.nvprof_nvtx_push("wgrad_fp8")
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        x_q, do1_q, w_gateup_grad_fp8, ks_cpu, grouped_layout, w_gateup_grad_fp8)
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        o2_q, do3_q, w_down_grad_fp8, ks_cpu, grouped_layout, w_down_grad_fp8)
    paddle.base.core.nvprof_nvtx_pop()

    print("w_gateup_grad bf16:", check(w_gateup_grad, paddle.zeros_like(w_gateup_grad)))
    print("w_gateup_grad fp8 :", check(w_gateup_grad_fp8, w_gateup_grad))
    print("w_down_grad   bf16:", check(w_down_grad, paddle.zeros_like(w_down_grad)))
    print("w_down_grad   fp8 :", check(w_down_grad_fp8, w_down_grad))

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
    dout = paddle.randn_like(recv_x) * 0.02
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
