"""Fused token-done + zip: correctness against `paddle.nn.functional.moe_unpermute`.

Usage:
    python tests_overlap/test_zip.py
"""

import numpy as np
import paddle

paddle.cuda.set_device(1)
paddle.empty([16, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200, edgeitems=5)

import deep_gemm
print("deep_gemm:", deep_gemm.__path__)


N = 16384  # num_recv_tokens
TOPK = 8
HIDDEN = 4096
E = 16  # 本地专家数
ALIGNMENT = 128

# (num_sms, chunk): 小 chunk 制造每专家多个 chunk 与余数 chunk, 小 num_sms 压 CTA 本地队列
CASES = ((96, 4096), (96, 512), (8, 1024))


def make_case():
    """
    构造一份 zip 的输入:
      - 每个 token 随机命中 1..TOPK 个本地专家, 在行内的顺序是乱的 (和 DeepEP 一致)
      - zip_to_atomic 指向 o3 里互不相同的行, 且 o3 里的行也是乱的 (模拟 atomic 序)
      - recv_topk_idx 的无效槽位故意填垃圾值, 用来验证 zip 只信 zip_to_atomic 的有效性判断
    """
    paddle.seed(0)

    # 随机命中 1..TOPK 个本地专家, 其余位置写 -1
    num_valid_topk = paddle.randint(1, TOPK + 1, [N])
    topk_idx_perm = paddle.randn([N, E]).argsort()[..., :TOPK]
    topk_idx_mask = paddle.arange(TOPK) < num_valid_topk.unsqueeze(1)
    topk_idx = paddle.where(
        topk_idx_mask, topk_idx_perm, paddle.full([1], -1, dtype="int64")
    )

    # 再次打乱行内顺序, 让有效位和 -1 交替出现
    topk_idx_perm = paddle.randn([N, TOPK]).argsort()
    topk_idx = topk_idx.index_sample(topk_idx_perm)

    tokens_per_expert = paddle.sum(
        paddle.arange(E).unsqueeze(1) == topk_idx.flatten(), axis=1
    ).tolist()

    topk_probs = paddle.randn([N, TOPK])

    num_unzipped_tokens = sum(
        (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT for n in tokens_per_expert
    )

    # o3 使用 atomic 序, o3_ref 使用 paddle unzip 的递增序
    o3 = paddle.randn([num_unzipped_tokens, HIDDEN], "bfloat16")
    o3_ref = paddle.empty_like(o3)

    # 逐专家赋值其 zip_to_atomic, 打乱在 o3 中的顺序
    offset = 0
    zip_to_atomic = paddle.full([N, TOPK], -1, dtype="int32")
    atomic_to_zip = paddle.full([num_unzipped_tokens], -1, dtype="int32")
    for i, n in enumerate(tokens_per_expert):
        atomic_perm = paddle.randn(n).argsort().cast("int32")
        zip_to_atomic[topk_idx == i] = atomic_perm + offset
        zip_pos = (topk_idx == i).any(axis=1).nonzero().flatten()
        atomic_to_zip[atomic_perm + offset] = zip_pos
        o3_ref[offset : offset + n] = o3[atomic_perm + offset]
        offset += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    # 调用 unzip 获取 zipped_expertwise_rowmap, 这是 zip 依赖的输入, 其余返回值无用
    hidden_states = paddle.empty([N, HIDDEN], dtype="bfloat16")
    scale = None
    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        unzipped_scale,
    ) = paddle.nn.functional.moe_permute(
        hidden_states,
        scale,
        topk_idx.astype("int32"),
        topk_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    return o3, topk_idx, zip_to_atomic, atomic_to_zip, o3_ref, zipped_expertwise_rowmap


def reference(o3, zipped_expertwise_rowmap, topk_idx):
    unzipped_probs = paddle.empty([o3.shape[0]], dtype="float32")
    zipped_out, zipped_probs_topk = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        topk_idx,
        unzipped_probs,
        total_zipped_tokens=N,
        num_experts=E,
    )
    return zipped_out


def make_task_queue(tokens_per_expert, chunk, seed=0):
    """The chunk arrival queue: one `[expert_idx, m_start, m_size, ready]` row per task.

    A real DeepEP dispatch appends a row when a chunk lands, so the expert order is
    random while chunks of one expert stay ordered. Everything is ready up-front here,
    as `zip` only cares about what the queue says, not about when it says it.
    """
    per_expert_tasks, offset = [], 0
    for expert_idx, count in enumerate(tokens_per_expert):
        per_expert_tasks.append([
            [expert_idx, offset + start, min(chunk, count - start), 1]
            for start in range(0, count, chunk)
        ])
        offset += (count + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

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


def check(name, out, zip_done, out_ref):
    diff = paddle.abs(out.float() - out_ref.float())
    count = int((diff != 0).sum())
    unfinished = int((zip_done != 1).sum())
    print(name, "diff:", count, "avg:", float(diff.mean()), "max:", float(diff.max()),
          "unfinished:", unfinished)
    return count == 0 and unfinished == 0


def main():
    o3, topk_idx, zip_to_atomic, atomic_to_zip, o3_ref, rowmap = make_case()

    topk_idx_i32 = topk_idx.cast("int32")
    out_ref = reference(o3_ref, rowmap, topk_idx_i32)

    tokens_per_expert = [int((topk_idx == i).sum()) for i in range(E)]
    num_valid_topk = paddle.sum(topk_idx >= 0, axis=1).astype("int32")
    print("num_recv_tokens:", N, "num_unzipped_tokens:", len(atomic_to_zip))
    print("tokens_per_expert:", tokens_per_expert)

    for num_sms, chunk in CASES:
        deep_gemm.set_num_sms(num_sms)
        task_queue = make_task_queue(tokens_per_expert, chunk)

        # NaN, so that any token left untouched by the queue shows up
        combine_input = paddle.full([N, HIDDEN], float("nan"), "bfloat16")
        token_done = paddle.zeros([N], "int32")
        zip_done = paddle.zeros([N], "int32")

        for task_idx in range(task_queue.shape[0]):
            deep_gemm.chunk_zip(o3, combine_input, atomic_to_zip, zip_to_atomic,
                                topk_idx, num_valid_topk, token_done, zip_done,
                                task_queue, task_idx, chunk)
        paddle.device.synchronize()

        name = f"sms={num_sms} chunk={chunk} tasks={task_queue.shape[0]}"
        assert check(name, combine_input, zip_done, out_ref), f"{name} mismatch"

        # Every token must have been counted by exactly its own number of experts
        assert paddle.equal_all(token_done, num_valid_topk), f"{name} token_done mismatch"

    print("PASSED")

    # Profile
    paddle.base.core.nvprof_start()
    deep_gemm.set_num_sms(96)
    chunk = 4096
    task_queue = make_task_queue(tokens_per_expert, chunk)
    prev_done_tokens = 0

    for i in range(10):
        token_done.zero_()
        paddle.base.core.nvprof_nvtx_push(f"trial_{i}")
        for task_idx in range(len(task_queue)):
            paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
            deep_gemm.chunk_zip(o3, combine_input, atomic_to_zip, zip_to_atomic,
                                topk_idx, num_valid_topk, token_done, zip_done,
                                task_queue, task_idx, chunk)
            paddle.base.core.nvprof_nvtx_pop()
            if i == 0:
                done_tokens = (token_done == num_valid_topk).sum().item()
                print("new_done_tokens:", done_tokens - prev_done_tokens)
                prev_done_tokens = done_tokens
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(10):
        paddle.base.core.nvprof_nvtx_push(f"ref_{i}")
        reference(o3_ref, rowmap, topk_idx_i32)
        paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
