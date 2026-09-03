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

E = 16
H = 4096
I = 2048
EP = 16
SEQLEN = 16384
TOPK = 8

CHUNK = 4096
NUM_SMS = 96
ALIGNMENT = 128
FUSE_SWIGLU = True
PRECISE_SWIGLU = True


class Result(NamedTuple):
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
    x_bwd: Tensor = None

    def __getitem__(self, key: str):
        return getattr(self, key)


def make_deepep_layout():
    """
    模拟在一个 EP 组内, 每个 rank 有 E 个专家的情况下, rank 0 收到的 dispatch+unzip 后的
    (recv_token_indices, tokens_per_expert).
    """
    # 模拟全局 token 打分
    scores = paddle.randn([EP * SEQLEN, EP * E])
    scores += paddle.randn([EP * E]) * 0.1  # add some system bias to experts
    _, topk_indices = scores.topk(TOPK)

    # 只保留命中 rank0 的 token, 故 topk_indices 里每一行至少有一个非 -1 值
    topk_hit = topk_indices < E
    token_hit = topk_hit.any(axis=1).nonzero().squeeze(1)
    topk_indices[~topk_hit] = -1
    topk_indices = topk_indices[token_hit]

    # DeepEP 并未对每行内容进行排序, 甚至有效值和 -1 是交错排列的, 这里对每行进行乱序
    row_perm = paddle.randn(topk_indices.shape).argsort(axis=1)
    topk_indices = topk_indices.index_sample(row_perm)

    tokens_per_expert = paddle.sum(
        paddle.arange(E)[:, None] == topk_indices.flatten(), axis=1).tolist()

    m_start, m_indices = [0], []

    for expert_idx, n in enumerate(tokens_per_expert):
        n_aligned = (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        m_start.append(m_start[-1] + n_aligned)
        m_indices.append(paddle.full([n_aligned], expert_idx, dtype="int32"))

    m_indices = paddle.concat(m_indices)

    return topk_indices, tokens_per_expert, m_start, m_indices


def make_atomic_layout(topk_indices, tokens_per_expert, m_start):
    """
    根据 DeepEP 序构造一份随机顺序的 atomic 序.
    DeepEP 序前反向是相同的, 但是 atomic 序不同, 可以调用两次本函数模拟不同的 atomic 序.
    """
    atomic_to_zip = paddle.full([m_start[-1]], -1, dtype="int32")
    zip_to_atomic = paddle.full(topk_indices.shape, -1, dtype="int32")

    for expert_idx, (n, offset) in enumerate(zip(tokens_per_expert, m_start)):
        # 选出属于 expert_idx 的 token 并打乱顺序
        slot_hit = topk_indices == expert_idx 
        token_idxs = slot_hit.any(axis=1).nonzero().squeeze(1).cast("int32")
        assert len(token_idxs) == n
        perm = paddle.randperm(n)

        atomic_to_zip[offset : offset + n] = token_idxs[perm]

        zip_to_atomic[slot_hit] = paddle.empty([n], dtype="int32").scatter_(
            perm, paddle.arange(n, dtype="int32") + offset
        )

    return atomic_to_zip, zip_to_atomic


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


def split_gate_up(o1):
    """The gate/up halves of `o1`, which is in the fully interleaved column order."""
    blocks = o1.reshape([o1.shape[0], I, 2])
    return blocks[:, :, 0], blocks[:, :, 1]


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

    o1.stop_gradient = False
    unzipped_probs.stop_gradient = False
    gate, up = split_gate_up(o1)
    gate, up = gate.float(), up.float()
    o2 = ((gate * F.sigmoid(gate)) * up * unzipped_probs.unsqueeze(-1)).cast("bfloat16")

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

    o2.backward(do2)
    do1 = o1.grad
    dprobs = unzipped_probs.grad

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

    return Result(o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs)


def compute_chunk(recv_x_pad, recv_probs, topk_indices, w_gateup, w_down, dout_pad,
                  atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                  task_queue, task_queue_bwd):
    num_valid_topk = (topk_indices != -1).sum(axis=-1, dtype="int32")

    ################################# Forward ##################################

    # paddle gather 会将 -1 的下标映射到最后一行, 故 padding token 会得到全 0
    x = recv_x_pad[atomic_to_zip]

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
        if FUSE_SWIGLU:
            deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx, o2=o2, probs=probs)
        else:
            deep_gemm.bf16_chunk_gemm_nn(x, w_gateup, o1, task_queue, task_idx)
            deep_gemm.chunk_weighted_swiglu(
                o1, probs, o2, task_queue, task_idx, precise=PRECISE_SWIGLU)
        deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
        deep_gemm.chunk_zip(o3, out, atomic_to_zip, zip_to_atomic, topk_indices, num_valid_topk,
                            token_done, zip_done, task_queue, task_idx, CHUNK)
    paddle.base.core.nvprof_nvtx_pop()

    ################################# Backward #################################

    do3 = dout_pad[atomic_to_zip_bwd]

    dx = paddle.full_like(x, float("nan"))
    do1 = paddle.full_like(o1, float("nan"))
    do2 = paddle.full_like(o2, float("nan"))
    o2_bwd = paddle.full_like(o2, float("nan"))
    x_bwd = paddle.full_like(x, float("nan"))

    token_done = paddle.zeros([len(recv_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_probs)], dtype="int32")

    paddle.base.core.nvprof_nvtx_push("backward")
    for task_idx in range(len(task_queue_bwd)):
        deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, task_queue_bwd, task_idx)
    paddle.base.core.nvprof_nvtx_pop()

    drecv_x = drecv_probs = None

    return Result(o1, o2, o3, out, do2, do1, dx, drecv_x, drecv_probs, o2_bwd, x_bwd)


def get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip):
    """将 atomic 序的 o1/o2/o3 等转换为参考序的映射表, padding 保留原位."""
    perm = paddle.arange(m_start[-1])
    for n, offset in zip(tokens_per_expert, m_start):
        perm[offset : offset + n] = atomic_to_zip[offset : offset + n].argsort() + offset
    return perm


def check(x, y):
    diff = (x.float() - y.float()).abs()
    return f"avg: {diff.mean():e} max: {diff.max():e}"


def main():
    deep_gemm.set_num_sms(NUM_SMS)

    topk_indices, tokens_per_expert, m_start, m_indices = make_deepep_layout()

    # 给最后 padding 一行全 0, 方便 gather 的时候将 -1 的下标映射到全 0
    recv_x_pad = paddle.randn([len(topk_indices) + 1, H], dtype="bfloat16")
    recv_x_pad[-1].zero_()
    recv_x = recv_x_pad[:-1]
    dout_pad = paddle.randn_like(recv_x_pad)
    dout_pad[-1].zero_()
    dout = dout_pad[:-1]

    recv_probs = paddle.randn(topk_indices.shape)
    w_gateup = paddle.randn([E, H, 2 * I], dtype="bfloat16") * 0.02
    w_down = paddle.randn([E, I, H], dtype="bfloat16") * 0.02

    ################################# Baseline #################################

    refs = reference(recv_x, recv_probs, topk_indices, tokens_per_expert, m_indices,
                     w_gateup, w_down, dout)

    ################################## Chunk ###################################

    atomic_to_zip, zip_to_atomic = make_atomic_layout(topk_indices, tokens_per_expert, m_start)
    atomic_to_zip_bwd, zip_to_atomic_bwd = make_atomic_layout(
        topk_indices, tokens_per_expert, m_start)

    task_queue = make_task_queue(tokens_per_expert, m_start, ready=True, seed=0)
    task_queue_bwd = make_task_queue(tokens_per_expert, m_start, ready=True, seed=1)

    paddle.base.core.nvprof_start()
    outs = compute_chunk(recv_x_pad, recv_probs, topk_indices, w_gateup, w_down, dout_pad,
                         atomic_to_zip, zip_to_atomic, atomic_to_zip_bwd, zip_to_atomic_bwd,
                         task_queue, task_queue_bwd)
    paddle.base.core.nvprof_stop()

    ################################# Validate #################################

    fwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip)
    bwd_perm = get_atomic_perm(tokens_per_expert, m_start, atomic_to_zip_bwd)

    for name in ("o1", "o2", "o3"):
        print(f"{name}:", check(outs[name][fwd_perm], refs[name]))
    for name in ("out",):
        print(f"{name}:", check(outs[name], refs[name]))
    for name in ("do2",):
        print(f"{name}:", check(outs[name][bwd_perm], refs[name]))


if __name__ == "__main__":
    main()
