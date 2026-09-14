import time
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

from utils import make_deepep_layout, make_atomic_layout, make_task_queue, get_atomic_perm

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

USE_UE8M0 = True  # 当前只支持 ue8m0
QUANT_BLOCK_SIZE = 512
SCALE_1E0 = 0x7f7f7f7f


class Result(NamedTuple):
    x: tuple[Tensor, Tensor]
    o1: Tensor
    o2: tuple[Tensor, Tensor]
    o3: Tensor
    out: Tensor
    do2: Tensor
    do1: tuple[Tensor, Tensor]
    dx: Tensor
    drecv_x: Tensor
    drecv_probs: Tensor
    o2_bwd: Tensor | tuple[Tensor, Tensor]
    w_gateup_grad: Tensor
    w_down_grad: Tensor

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
    """[hidden, token] -> [token, hidden], 仅用于适配 k_grouped tn 的 MN-major 要求."""
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
    do1_quant = (do1_fp8, do1_scale.T)

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

    # 基于 baseline 接入 fp8 k_grouped_gemm, 用于对比 quant 的开销和 fp8 wgrad 的收益
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

    return Result(x, o1, o2, o3, out, do2, do1_quant, dx, drecv_x, drecv_probs, o2_bwd,
                  w_gateup_grad, w_down_grad)


def compute_chunk(recv_x, recv_probs, topk_indices, tokens_per_expert, m_start,
                  w_gateup, w_down, w_gateup_t, w_down_t, dout,
                  atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                  task_queue, task_queue_bwd):
    num_valid_topk = (topk_indices != -1).sum(axis=-1, dtype="int32")

    ################################# Forward ##################################

    # 模拟通信已经给出 unzip 的结果; 通信输出的 scale 已经整体转置
    recv_x_fp8, recv_scale = recv_x
    x_fp8 = deep_gemm.token_gather(recv_x_fp8, atomic_to_zip)
    x_scale = deep_gemm.token_gather(recv_scale, atomic_to_zip)
    x_scale = x_scale.T.contiguous().T
    x = (x_fp8, x_scale)

    # paddle scatter 会将 -1 的下标映射到最后一格, 需要多分配一格来接住这些无效值
    probs_pad = paddle.zeros([len(x_fp8) + 1], dtype="float32")
    probs_pad.scatter_(zip_to_atomic.flatten(), recv_probs.flatten())
    probs = probs_pad[:-1]

    o1 = paddle.full([len(x_fp8), 2 * I], float("nan"), dtype="bfloat16")
    o2_fp8 = paddle.full([len(x_fp8), I], float("nan"), dtype="float8_e4m3fn")
    # 注意即使是中间变量的 o2_scale 也要转置
    o2_scale = paddle.full([I // QUANT_BLOCK_SIZE, len(x_fp8)], SCALE_1E0, dtype="int32").T
    o2 = (o2_fp8, o2_scale)
    o3 = paddle.full([len(x_fp8), H], float("nan"), dtype="bfloat16")
    out = paddle.full([len(recv_x_fp8), H], float("nan"), dtype="bfloat16")

    token_done = paddle.zeros([len(recv_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_probs)], dtype="int32")

    paddle.base.core.nvprof_nvtx_push("forward")
    for task_idx in range(len(task_queue)):
        deep_gemm.fp8_chunk_gemm_nt((x_fp8, x_scale), w_gateup_t, o1, task_queue, task_idx)
        deep_gemm.chunk_weighted_swiglu(o1, probs, o2_fp8, task_queue, task_idx, CHUNK,
                                        o2_scales=o2_scale)
        deep_gemm.fp8_chunk_gemm_nt((o2_fp8, o2_scale), w_down_t, o3, task_queue, task_idx)
        deep_gemm.chunk_zip(o3, out, atomic_to_zip, zip_to_atomic, topk_indices, num_valid_topk,
                            token_done, zip_done, task_queue, task_idx, CHUNK)
    paddle.base.core.nvprof_nvtx_pop()

    ################################# Backward #################################

    dout_fp8, dout_scale = dout
    do3_fp8 = deep_gemm.token_gather(dout_fp8, atomic_to_zip_bwd)
    do3_scale = deep_gemm.token_gather(dout_scale, atomic_to_zip_bwd)
    do3_scale = do3_scale.T.contiguous().T
    do3 = (do3_fp8, do3_scale)

    do2 = paddle.full([len(x_fp8), I], float("nan"), dtype="bfloat16")
    dx = paddle.full([len(x_fp8), H], float("nan"), dtype="bfloat16")
    do1_fp8 = paddle.empty([len(x_fp8), 2 * I], dtype="float8_e4m3fn")
    do1_scale = paddle.full([2 * I // QUANT_BLOCK_SIZE, len(x_fp8)], SCALE_1E0, dtype="int32").T
    do1 = (do1_fp8, do1_scale)
    o2_bwd_fp8 = paddle.empty([len(x_fp8), I], dtype="float8_e4m3fn")
    o2_bwd_scale = paddle.full([I // QUANT_BLOCK_SIZE, len(x_fp8)], SCALE_1E0, dtype="int32").T
    o2_bwd = (o2_bwd_fp8, o2_bwd_scale)
    drecv_x = paddle.full_like(out, float("nan"))
    drecv_probs = paddle.zeros_like(recv_probs)  # 无效位预先填 0

    token_done = paddle.zeros([len(recv_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_probs)], dtype="int32")

    paddle.base.core.nvprof_nvtx_push("backward")
    for task_idx in range(len(task_queue_bwd)):
        deep_gemm.fp8_chunk_gemm_nt((do3_fp8, do3_scale), w_down, do2, task_queue_bwd, task_idx)
        deep_gemm.chunk_weighted_swiglu_grad(
            o1, probs, do2, o2_bwd_fp8, do1_fp8, drecv_probs, atomic_to_zip_bwd, zip_to_atomic,
            topk_indices, task_queue_bwd, task_idx, CHUNK,
            o2_bwd_scales=o2_bwd_scale, do1_scales=do1_scale)
        deep_gemm.fp8_chunk_gemm_nt((do1_fp8, do1_scale), w_gateup, dx, task_queue_bwd, task_idx)
        deep_gemm.chunk_zip(dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, topk_indices,
                            num_valid_topk, token_done, zip_done, task_queue_bwd, task_idx, CHUNK)
    paddle.base.core.nvprof_nvtx_pop()

    ################################## Wgrad ###################################

    # 各专家 seq 维重新向 512 对齐
    ks_512, m_start_512 = [], [0]
    for n in tokens_per_expert:
        n_512 = (n + 511) // 512 * 512
        m_start_512.append(m_start_512[-1] + n_512)
        ks_512.append(n_512)
    m_start_gpu = paddle.to_tensor(m_start, dtype="int32")
    m_start_512_gpu = paddle.to_tensor(m_start_512, dtype="int32")
    grouped_layout = paddle.to_tensor(ks_512, dtype="int32")

    paddle.base.core.nvprof_nvtx_push("wgrad_map")
    ordered_to_zip, ordered_to_atomic = deep_gemm.sort_map(
        zip_to_atomic_bwd, m_start_gpu, m_start_512[-1], m_start_512_gpu)
    paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_nvtx_push("requant")
    # 只有 x 是从 zipped 的向量解压，其他都是原样大小重排
    x_w = deep_gemm.requant_wgrad_input(recv_x[0], recv_x[1].T.contiguous().T, ordered_to_zip)
    do1_w = deep_gemm.requant_wgrad_input(*do1, ordered_to_atomic)
    o2_w = deep_gemm.requant_wgrad_input(*o2_bwd, ordered_to_atomic)
    do3_w = deep_gemm.requant_wgrad_input(*do3, ordered_to_atomic)
    paddle.base.core.nvprof_nvtx_pop()

    w_gateup_grad = paddle.zeros([E, H, 2 * I], dtype="float32")
    w_down_grad = paddle.zeros([E, I, H], dtype="float32")

    paddle.base.core.nvprof_nvtx_push("wgrad")
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        x_w, do1_w, w_gateup_grad, ks_512, grouped_layout, w_gateup_grad)
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        o2_w, do3_w, w_down_grad, ks_512, grouped_layout, w_down_grad)
    paddle.base.core.nvprof_nvtx_pop()

    return Result(x, o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs, o2_bwd,
                  w_gateup_grad, w_down_grad)


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
    return f"avg: {avg} max: {max}{banner}"


def main():
    deep_gemm.set_num_sms(NUM_SMS)

    topk_indices, tokens_per_expert, m_start, m_indices = make_deepep_layout(
        EP, SEQLEN, E, TOPK, ALIGNMENT)

    recv_x = paddle.randn([len(topk_indices), H], dtype="bfloat16")
    dout = paddle.randn_like(recv_x) * 0.02
    recv_probs = paddle.randn(topk_indices.shape)
    w_gateup = paddle.randn([E, H, 2 * I], dtype="bfloat16") * 0.02
    w_down = paddle.randn([E, I, H], dtype="bfloat16") * 0.02

    recv_x_quant = quant_input(recv_x)
    dout_quant = quant_input(dout)

    # fp8 的性能对 layout 敏感, 因此 fp8 需要维护两套 layout 的权重, 不像 bf16 只需要改算子参数
    w_gateup_quant = quant_weight(w_gateup, transpose=False)
    w_down_quant = quant_weight(w_down, transpose=False)
    w_gateup_t_quant = quant_weight(w_gateup, transpose=True)
    w_down_t_quant = quant_weight(w_down, transpose=True)

    ################################# Baseline #################################

    refs = reference(recv_x_quant, recv_probs, topk_indices, tokens_per_expert, m_indices,
                     w_gateup_quant, w_down_quant, w_gateup_t_quant, w_down_t_quant, dout)

    ################################## Chunk ###################################

    atomic_to_zip, zip_to_atomic = make_atomic_layout(topk_indices, tokens_per_expert, m_start)
    atomic_to_zip_bwd, zip_to_atomic_bwd = make_atomic_layout(
        topk_indices, tokens_per_expert, m_start)
    task_queue = make_task_queue(tokens_per_expert, m_start, CHUNK, ready=True, seed=0)
    task_queue_bwd = make_task_queue(tokens_per_expert, m_start, CHUNK, ready=True, seed=1)

    outs = compute_chunk(recv_x_quant, recv_probs, topk_indices, tokens_per_expert, m_start,
                         w_gateup_quant, w_down_quant, w_gateup_t_quant, w_down_t_quant,
                         dout_quant, atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd,
                         zip_to_atomic_bwd, task_queue, task_queue_bwd)

    ################################# Validate #################################

    fwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip)
    bwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip_bwd)

    checks = [
        (("x", "o1", "o2", "o3"), fwd_perm),
        (("out",), None),
        (("do2", "do1", "o2_bwd", "dx"), bwd_perm),
        (("drecv_x", "drecv_probs", "w_gateup_grad", "w_down_grad"), None),
    ]
    for names, perm in checks:
        for name in names:
            out, ref = outs[name], refs[name]
            if isinstance(out, tuple) and isinstance(ref, tuple):
                # 两边都是 fp8, 直接比较字节
                out = (out[0].view("int8")[perm], out[1][perm]) if perm is not None else out
                fp8_diff = int((out[0].view("int8") != ref[0].view("int8")).sum())
                scale_diff = int((out[1] != ref[1]).sum())
                banner = (" " + "-" * 40) if (fp8_diff or scale_diff) else ""
                print(f"{name}: fp8: {fp8_diff} scale: {scale_diff}{banner}")
            else:
                # 否则将 fp8 一方先 dequant 再比较
                out = dequant(*out) if isinstance(out, tuple) else out
                out = out[perm] if perm is not None else out
                print(f"{name}:", check(out, ref))

    ################################# Profile ##################################

    del refs, outs

    paddle.base.core.nvprof_start()

    paddle.base.core.nvprof_nvtx_push("baseline")
    reference(recv_x_quant, recv_probs, topk_indices, tokens_per_expert, m_indices,
              w_gateup_quant, w_down_quant, w_gateup_t_quant, w_down_t_quant, dout)
    paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_nvtx_push("chunk")
    compute_chunk(recv_x_quant, recv_probs, topk_indices, tokens_per_expert, m_start,
                  w_gateup_quant, w_down_quant, w_gateup_t_quant, w_down_t_quant,
                  dout_quant, atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                  task_queue, task_queue_bwd)
    paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
