import paddle
import numpy as np
paddle.cuda.set_device(1)
paddle.empty([10 * 2**30])
paddle.set_printoptions(linewidth=200)
paddle.seed(0)

# from paddlefleet_ops import deep_gemm  # this is OK
import deep_gemm
print("deep_gemm:", deep_gemm.__path__)


def fwd_gate_up_bf16(x, expert_w1, m_indices):
    o1 = paddle.empty(
        [x.shape[0], expert_w1.shape[2]], dtype="bfloat16"
    )
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
        x,
        expert_w1,
        o1,
        m_indices,
    )
    return o1


E = 16
H = 4096
I = 2048

probs = paddle.randn([E]) * 0.2 + 1.0
tokens_per_expert = (((probs * 8192).int() + 127) // 128 * 128).tolist()
print("tokens_per_expert:", tokens_per_expert)

m_indices = paddle.concat(
    [paddle.full([n], i, "int32") for i, n in enumerate(tokens_per_expert)]
)
print("m_indices:", m_indices)

chunk = 4096
print("group_gemm_tiles:", sum(tokens_per_expert) * 2 * I / (128 * 128))
print("chunk_tiles:")
for i, n in enumerate(tokens_per_expert):
    print(" ", i, ":", n * 2 * I / (128 * 128))
print("chunk_kernels:", sum((n + chunk - 1) // chunk for n in tokens_per_expert))

x = paddle.randn([sum(tokens_per_expert), H], "bfloat16")
w_gateup = paddle.randn([E, H, 2 * I], "bfloat16")

deep_gemm.set_num_sms(96)
print("num_sms:", deep_gemm.get_num_sms())

n_streams = 1
streams = [paddle.cuda.Stream() for _ in range(n_streams)]


def compute_chunks():
    start = 0
    workloads = [0] * n_streams

    # child streams wait main stream
    paddle.randn([1024, 1024, 1024])
    event = paddle.device.Event()
    event.record()
    for stream in streams:
        stream.wait_event(event)

    # issue tasks
    for expert_idx, n in enumerate(tokens_per_expert):
        end = start + n
        paddle.base.core.nvprof_nvtx_push(f"{expert_idx}_{n//128}")

        for i in range(start, end, chunk):
            x_chunk = x[i : min(i + chunk, end)]

            # pick the stream with least workload
            stream_idx = workloads.index(min(workloads))
            workloads[stream_idx] += len(x_chunk)

            with paddle.device.stream_guard(streams[stream_idx]):
                out = paddle.empty([x_chunk.shape[0], 2 * I], "bfloat16")
                deep_gemm.bf16_gemm_nn(x_chunk, w_gateup[expert_idx], out)

        paddle.base.core.nvprof_nvtx_pop()
        start = end

    # main stream waits child streams
    for stream in streams:
        event.record(stream)
        paddle.device.current_stream().wait_event(event)
    paddle.randn([1024, 1024, 1024])


for i in range(10):
    paddle.base.core.nvprof_start() if i == 1 else ()
    paddle.base.core.nvprof_nvtx_push(f"trial_{i}")

    compute_chunks()
    # fwd_gate_up_bf16(x, w_gateup, m_indices)

    paddle.base.core.nvprof_nvtx_pop()

paddle.device.synchronize()
paddle.base.core.nvprof_stop()
