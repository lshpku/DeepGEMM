"""Gate-up GEMM with the SwiGLU fused into its epilogue: correctness and cost of the fusion.

Usage:
    python tests_overlap/test_fused_swiglu.py
    python tests_overlap/test_fused_swiglu.py --num-sms 148 --m 4096
"""

import argparse

import numpy as np
import paddle
import paddle.nn.functional as F

paddle.cuda.set_device(1)
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

import deep_gemm

H = 4096
I = 2048
GROUP = 64  # the interleave granularity, i.e. half of the epilogue's store block N


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096, help="one chunk of tokens")
    parser.add_argument("--num-sms", type=int, default=96, help="the overlap scheme's compute budget")
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def calc_diff(x, y):
    x, y = x.astype("float32"), y.astype("float32")
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def interleave_columns(num_cols):
    """`[gate | up]` -> `[gate[0:64], up[0:64], gate[64:128], up[64:128], ...]`, a free relayout."""
    half = num_cols // 2
    perm = []
    for start in range(0, half, GROUP):
        perm.extend(range(start, start + GROUP))
        perm.extend(range(half + start, half + start + GROUP))
    return np.asarray(perm, dtype=np.int64)


def bench(fn, name, iters):
    fn()
    paddle.device.synchronize()
    start, end = paddle.device.Event(True), paddle.device.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    paddle.device.synchronize()
    t = start.elapsed_time(end) / iters * 1e3
    print(f"{name:24s}: {t:7.1f} us")
    return t


def main():
    args = parse_args()
    m = args.m
    print(f"m: {m} | H: {H} | I: {I} | num_sms: {args.num_sms}")

    x = paddle.randn([m, H], "bfloat16")
    w = paddle.randn([H, 2 * I], "bfloat16") * 0.02
    probs = paddle.rand([m], "float32")

    # The interleaved weight, and the same permutation applied to the reference output
    perm = paddle.to_tensor(interleave_columns(2 * I).astype(np.int32))
    w_inter = w.index_select(perm, axis=1).contiguous()

    # The chunk API takes `[G, K, N]` weights and one task per launch
    w_chunk = w_inter.unsqueeze(0).contiguous()
    queue = paddle.to_tensor([[0, 0, m, 1]], dtype="int32")

    deep_gemm.set_num_sms(args.num_sms)

    # Reference: a plain GEMM on the interleaved weight, then SwiGLU on the de-interleaved halves
    o1_ref = paddle.empty([m, 2 * I], "bfloat16")
    deep_gemm.bf16_gemm_nn(x, w_inter, o1_ref)
    blocks = o1_ref.reshape([m, 2 * I // (2 * GROUP), 2, GROUP])
    gate = blocks[:, :, 0].reshape([m, I]).astype("float32")
    up = blocks[:, :, 1].reshape([m, I]).astype("float32")
    o2_ref = (F.silu(gate) * up * probs.unsqueeze(-1)).astype("bfloat16")

    # The fused kernel: `o1` in the interleaved order, `o2` straight out of the epilogue
    o1 = paddle.full([m, 2 * I], float("nan"), "bfloat16")
    o2 = paddle.full([m, I], float("nan"), "bfloat16")
    deep_gemm.bf16_chunk_gemm_nn(x, w_chunk, o1, queue, 0, o2=o2, probs=probs)
    paddle.device.synchronize()

    for name, out, ref, tol in (("o1", o1, o1_ref, 0.0), ("o2", o2, o2_ref, 1e-6)):
        diff = calc_diff(out, ref)
        print(f"{name}: diff={diff:.3e}")
        assert diff <= tol, f"{name} mismatch: {diff}"

    # Cost of the fusion: against the bare chunk GEMM, and against today's two-kernel path
    o2_sep = paddle.empty([m, I], "bfloat16")
    paddle.base.core.nvprof_start()
    t_gemm = bench(lambda: deep_gemm.bf16_chunk_gemm_nn(x, w_chunk, o1, queue, 0),
                   "gemm only", args.iters)
    t_fused = bench(lambda: deep_gemm.bf16_chunk_gemm_nn(x, w_chunk, o1, queue, 0, o2=o2, probs=probs),
                    "gemm + fused swiglu", args.iters)
    t_swiglu = bench(lambda: deep_gemm.chunk_weighted_swiglu(o1, probs, o2_sep, queue, 0),
                     "standalone swiglu", args.iters)
    paddle.base.core.nvprof_stop()
    print(f"fusion overhead: {t_fused - t_gemm:.1f} us | separate kernel: {t_swiglu:.1f} us")

    # The standalone kernel is bound by the `o1` round trip, the fused one is not
    bytes_sep = m * 2 * I * 2 + m * I * 2
    print(f"standalone swiglu traffic: {bytes_sep / 1e6:.1f} MB"
          f" -> {bytes_sep / t_swiglu / 1e6:.2f} TB/s")

    print("PASSED")


if __name__ == "__main__":
    main()
