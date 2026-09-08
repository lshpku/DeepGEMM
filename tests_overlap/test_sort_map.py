"""离线顺序转换函数 (sort_unzip_map / sort_atomic_map / token_gather) 的正确性测试."""

import paddle

paddle.cuda.set_device(1)
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

import deep_gemm
print("deep_gemm:", deep_gemm.__path__)

from utils import make_deepep_layout, make_atomic_layout

E = 16
H = 4096
EP = 16
SEQLEN = 4096
TOPK = 8
ALIGNMENT = 128


def reference_ordered_to_zip(tokens_per_expert, m_start, atomic_to_zip):
    """每个专家区域内按 token 在 DeepEP 序中的下标排序, padding 位填 -1."""
    ordered_to_zip = paddle.full([m_start[-1]], -1, dtype="int32")
    for n, offset in zip(tokens_per_expert, m_start):
        ordered_to_zip[offset : offset + n] = atomic_to_zip[offset : offset + n].sort()
    return ordered_to_zip


def reference_ordered_to_atomic(tokens_per_expert, m_start, atomic_to_zip):
    """标准 unzip 序 -> atomic 序, 即 argsort; padding 位填 -1."""
    ordered_to_atomic = paddle.full([m_start[-1]], -1, dtype="int32")
    for n, offset in zip(tokens_per_expert, m_start):
        perm = atomic_to_zip[offset : offset + n].argsort().cast("int32")
        ordered_to_atomic[offset : offset + n] = perm + offset
    return ordered_to_atomic


def check(name, out, ref):
    diff = int((out != ref).sum())
    print(f"{name}: diff {diff}" + ("" if diff == 0 else " " + "-" * 40))
    assert diff == 0


def main():
    topk_indices, tokens_per_expert, m_start, _ = make_deepep_layout(
        EP, SEQLEN, E, TOPK, ALIGNMENT)
    num_recv_tokens, num_unzipped_tokens = len(topk_indices), m_start[-1]
    print(f"num_recv_tokens: {num_recv_tokens}, num_unzipped_tokens: {num_unzipped_tokens}, "
          f"padding rows: {num_unzipped_tokens - sum(tokens_per_expert)}")

    atomic_to_zip, zip_to_atomic = make_atomic_layout(
        topk_indices, tokens_per_expert, m_start)
    atomic_to_zip_bwd, zip_to_atomic_bwd = make_atomic_layout(
        topk_indices, tokens_per_expert, m_start)
    m_start_gpu = paddle.to_tensor(m_start, dtype="int32")

    ############################### sort_unzip_map ###############################

    ordered_to_zip = deep_gemm.sort_unzip_map(zip_to_atomic, m_start_gpu, num_unzipped_tokens)
    check("ordered_to_zip",
          ordered_to_zip,
          reference_ordered_to_zip(tokens_per_expert, m_start, atomic_to_zip))

    # 反向 atomic 序的 zip_to_atomic 必须给出同一张表 (标准序只由 DeepEP 序决定)
    check("ordered_to_zip (bwd table)",
          deep_gemm.sort_unzip_map(zip_to_atomic_bwd, m_start_gpu, num_unzipped_tokens),
          ordered_to_zip)

    ############################### sort_atomic_map ##############################

    check("ordered_to_atomic",
          deep_gemm.sort_atomic_map(zip_to_atomic_bwd, m_start_gpu, num_unzipped_tokens),
          reference_ordered_to_atomic(tokens_per_expert, m_start, atomic_to_zip_bwd))

    ################################ token_gather ###############################

    # 前向: 从未 unzip 的 recv_x 解压出标准序的 unzipped_tokens
    recv_x = paddle.randn([num_recv_tokens, H], dtype="bfloat16")
    x_ref = paddle.zeros([num_unzipped_tokens, H], dtype="bfloat16")
    valid = (ordered_to_zip >= 0).nonzero().squeeze(1)
    x_ref[valid] = recv_x[ordered_to_zip[valid]]
    check("token_gather (unzip)", deep_gemm.token_gather(recv_x, ordered_to_zip), x_ref)

    # 反向: 从 atomic 序的 do3 得到标准序的 do3
    ordered_to_atomic = deep_gemm.sort_atomic_map(
        zip_to_atomic_bwd, m_start_gpu, num_unzipped_tokens)
    do3 = paddle.randn([num_unzipped_tokens, H], dtype="bfloat16")
    do3_ref = paddle.zeros_like(do3)
    valid = (ordered_to_atomic >= 0).nonzero().squeeze(1)
    do3_ref[valid] = do3[ordered_to_atomic[valid]]
    check("token_gather (reorder)", deep_gemm.token_gather(do3, ordered_to_atomic), do3_ref)

    # 两次 gather 串起来: 标准序的 x 也可以由 atomic 序的 x 重排得到
    x_atomic = deep_gemm.token_gather(recv_x, atomic_to_zip_bwd)
    check("token_gather (chained)", deep_gemm.token_gather(x_atomic, ordered_to_atomic), x_ref)

    # 非 bf16 / 非幂次行长也要能跑
    probs = paddle.randn([num_recv_tokens, 12], dtype="float32")
    probs_ref = paddle.zeros([num_unzipped_tokens, 12], dtype="float32")
    probs_ref[valid] = probs[ordered_to_zip[valid]]
    check("token_gather (fp32, n=12)", deep_gemm.token_gather(probs, ordered_to_zip), probs_ref)


if __name__ == "__main__":
    main()
