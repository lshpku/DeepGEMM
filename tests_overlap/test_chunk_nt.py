"""Chunk GEMM with a K-major weight (`nt`), which the backward pass needs.

The forward calls `A[M, K] @ B[G, K, N]`; the backward needs `dA[M, K] = dD[M, N] @ B[G, K, N].mT`,
i.e. the same weight contracted along `N` instead of `K`. Both are the same kernel template with
a different `B` major type, so `bf16_chunk_gemm_nt` takes a `[G, N, K]` weight -- the forward's
stored buffer as is -- and `bf16_chunk_gemm_nn` is a transposed view of it. This test checks that
the two agree, and that the `nt` layout costs nothing.

Usage:
    python tests_overlap/test_chunk_nt.py
    python tests_overlap/test_chunk_nt.py --device 0 --m 4096
"""

import argparse

import paddle

paddle.set_printoptions(linewidth=200)

import deep_gemm

G = 16  # num_experts
NUM_SMS = 96
BLOCK_M = 128  # the contiguous layout's M alignment, which `m_start` must respect


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=4096, help="one chunk of tokens")
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def calc_diff(x, y):
    x, y = x.astype("float32"), y.astype("float32")
    denominator = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denominator).item()


def bench(fn, iters):
    fn()
    paddle.device.synchronize()
    start, end = paddle.device.Event(True), paddle.device.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    paddle.device.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def run_case(m, n, k, expert_idx, m_start, iters):
    """One chunk task of `[m, k] @ [k, n]`, with the weight stored both ways."""
    a = paddle.randn([m_start + m, k], "bfloat16")
    # `b_kn[g]` is `[K, N]` (the forward layout), `b_nk[g]` is its transpose, stored contiguously
    b_kn = paddle.randn([G, k, n], "bfloat16") * 0.02
    b_nk = b_kn.transpose([0, 2, 1]).contiguous()
    nan = paddle.full([1], float("nan"), "bfloat16")
    d = paddle.broadcast_to(nan, [m_start + m, n]).contiguous()

    # The task descriptor the kernel claims: the row range and the expert
    queue = paddle.to_tensor([[expert_idx, m_start, m, 1]], dtype="int32")

    # `nn` takes `[G, K, N]`, `nt` takes `[G, N, K]`; both describe the very same matrix
    ref = paddle.matmul(a[m_start:].astype("float32"),
                        b_kn[expert_idx].astype("float32")).astype("bfloat16")

    results = {}
    for name, fn, b in (("nn", deep_gemm.bf16_chunk_gemm_nn, b_kn),
                        ("nt", deep_gemm.bf16_chunk_gemm_nt, b_nk)):
        d.copy_(paddle.broadcast_to(nan, d.shape), False)
        fn(a, b, d, queue, 0)
        paddle.device.synchronize()

        diff = calc_diff(d[m_start:], ref)
        assert diff < 1e-4, f"{name} m={m} n={n} k={k} mismatch: {diff}"
        # The rows outside the task must be untouched
        assert bool(paddle.isnan(d[:m_start]).all()), f"{name} wrote outside the task"

        t = bench(lambda fn=fn, b=b: fn(a, b, d, queue, 0), iters)
        results[name] = (t, diff)

    return results


def check_backward_call(m, h, i, iters):
    """The real backward call: the weights stay in their forward storage layout.

    `w_gateup` is stored as `[G, H, 2I]`, which the forward passes to `nn` as `[G, K, N]`. The
    backward contracts along `2I` instead, i.e. it wants `[G, N, K] = [G, H, 2I]` -- the exact
    same buffer, handed to `nt` with no transpose, no copy and no offline relayout.
    """
    w_gateup = paddle.randn([G, h, 2 * i], "bfloat16") * 0.02
    w_down = paddle.randn([G, i, h], "bfloat16") * 0.02
    do1 = paddle.randn([m, 2 * i], "bfloat16")
    do3 = paddle.randn([m, h], "bfloat16")
    dx = paddle.zeros([m, h], "bfloat16")
    do2 = paddle.zeros([m, i], "bfloat16")

    for expert_idx in (0, G - 1):
        queue = paddle.to_tensor([[expert_idx, 0, m, 1]], dtype="int32")

        # dx = do1 @ w_gateup^T
        deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, queue, 0)
        # do2 = do3 @ w_down^T
        deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, queue, 0)
        paddle.device.synchronize()

        dx_ref = paddle.matmul(do1.astype("float32"),
                               w_gateup[expert_idx].astype("float32"), transpose_y=True)
        do2_ref = paddle.matmul(do3.astype("float32"),
                                w_down[expert_idx].astype("float32"), transpose_y=True)
        for name, out, ref in (("dx", dx, dx_ref), ("do2", do2, do2_ref)):
            diff = calc_diff(out, ref.astype("bfloat16"))
            assert diff < 1e-4, f"expert {expert_idx} {name} mismatch: {diff}"

    t_dx = bench(lambda: deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, queue, 0), iters)
    t_do2 = bench(lambda: deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, queue, 0), iters)
    print(f"\nbackward call on the forward weights: gateup_grad {t_dx:.1f} us, "
          f"down_grad {t_do2:.1f} us")


def main():
    args = parse_args()
    paddle.cuda.set_device(args.device)
    paddle.seed(0)
    deep_gemm.set_num_sms(NUM_SMS)
    print(f"device: {args.device} | num_sms: {NUM_SMS} | m: {args.m}")

    H, I = 4096, 2048
    # The four GEMMs of one chunk: `nn` is what the forward calls, `nt` what the backward needs
    cases = [
        ("fwd gateup   x  @ w_gateup ", args.m, 2 * I, H),
        ("fwd down     o2 @ w_down   ", args.m, H, I),
        ("bwd gateup   do1@ w_gateup^T", args.m, H, 2 * I),
        ("bwd down     do3@ w_down^T ", args.m, I, H),
        # A short remainder chunk, and a task that does not start at row 0
        ("remainder m=1000           ", 1000, H, 2 * I),
        ("offset m_start=4096        ", args.m, H, 2 * I),
    ]

    print(f"\n{'case':30s} {'M':>6s} {'N':>6s} {'K':>6s} {'nn (us)':>9s} {'nt (us)':>9s} {'nt/nn':>7s}")
    for i, (name, m, n, k) in enumerate(cases):
        m_start = 4096 if "m_start" in name else 0
        expert_idx = i % G
        r = run_case(m, n, k, expert_idx, m_start, args.iters)
        t_nn, t_nt = r["nn"][0], r["nt"][0]
        print(f"{name:30s} {m:6d} {n:6d} {k:6d} {t_nn:9.1f} {t_nt:9.1f} {t_nt / t_nn:7.3f}")

    check_backward_call(args.m, H, I, args.iters)

    print("\nPASSED")


if __name__ == "__main__":
    main()
