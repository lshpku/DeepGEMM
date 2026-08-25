#pragma once

#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_chunk_arrival_sim.hpp"
#include "../jit_kernels/impls/smxx_chunk_token_done.hpp"
#include "../jit_kernels/impls/smxx_chunk_weighted_swiglu.hpp"
#endif

namespace deep_gemm::overlap {

#if DG_TENSORMAP_COMPATIBLE

// Chunk-wise BF16 GEMM for the fine-grained compute-communication overlap
// Shape must be `[M, K] @ [G, K, N]`, where the task queue holds
// `[expert_idx, m_start, m_size, ready]` per task, and one call computes one task
// NOTES: passing `o2` and `probs` fuses the weighted SwiGLU into the epilogue, which needs
//        `b`'s columns interleaved in groups of 64, i.e. `[gate[0:64], up[0:64], gate[64:128], ...]`,
//        so that one `BLOCK_N` tile holds both halves of the same 64 activation channels;
//        `d` then keeps the linear output (in that interleaved order) for the backward pass
static void bf16_chunk_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d,
                               const torch::Tensor& task_queue,
                               const int& task_idx,
                               const std::optional<torch::Tensor>& o2,
                               const std::optional<torch::Tensor>& probs,
                               const std::string& compiled_dims) {
    // Transpose into `[G, N, K].mT`
    const auto& b_t = b.transpose(1, 2);
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b_t);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b_t);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    // D must be N-major
    check_major_type_cd(d);

    // The fused SwiGLU halves N, and its output shares the row indexing with `d`
    if (o2.has_value()) {
        const auto [m__, n_half] = get_shape<2>(o2.value());
        DG_HOST_ASSERT(m == m__ and n == n_half * 2 and n % 128 == 0);
        DG_HOST_ASSERT(o2.value().scalar_type() == torch::kBFloat16);
        check_major_type_cd(o2.value());
        DG_HOST_ASSERT(probs.has_value());
        DG_HOST_ASSERT(probs.value().scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(static_cast<int>(probs.value().numel()) == m and probs.value().is_contiguous());
    }

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 and "Chunk GEMM only supports SM100 for now");
    sm100_bf16_chunk_gemm(a, b_t, d, task_queue, task_idx,
                          num_groups, m, n, k, major_a, major_b, compiled_dims, o2, probs);
}

// Weighted SwiGLU for one chunk task: `o2 = silu(o1[:, :N]) * o1[:, N:] * probs`
static void chunk_weighted_swiglu(const torch::Tensor& o1, const torch::Tensor& probs,
                                  const torch::Tensor& o2,
                                  const torch::Tensor& task_queue,
                                  const int& task_idx) {
    // Type and shape checks
    const auto [m, n2] = get_shape<2>(o1);
    const auto [m_, n] = get_shape<2>(o2);
    DG_HOST_ASSERT(m == m_ and n2 == n * 2);
    DG_HOST_ASSERT(o1.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(o2.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(probs.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(static_cast<int>(probs.numel()) == m);
    DG_HOST_ASSERT(o1.is_contiguous() and o2.is_contiguous() and probs.is_contiguous());

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    smxx_chunk_weighted_swiglu(o1, probs, o2, task_queue, task_idx);
}

// Bump the token completion counters of one chunk, to be issued right after its down GEMM
// NOTES: `row_to_token` maps an unzipped row to the unduplicated token (`-1` for padding),
//        and `zip` sums a token once its counter reaches the token's number of experts
static void chunk_signal_token_done(const torch::Tensor& row_to_token,
                                    const torch::Tensor& token_done,
                                    const torch::Tensor& task_queue,
                                    const int& task_idx) {
    DG_HOST_ASSERT(row_to_token.is_contiguous() and token_done.is_contiguous());
    DG_HOST_ASSERT(row_to_token.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(token_done.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(row_to_token.dim() == 1 and token_done.dim() == 1);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    smxx_chunk_token_done(row_to_token, token_done, task_queue, task_idx);
}

// Mark the chunks as arrived one by one, standing in for the communication kernel
// NOTES: for single-card debugging only, and it must run on another stream to overlap
static void simulate_chunk_arrival(const torch::Tensor& task_queue, const int& interval_ns) {
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(interval_ns >= 0);

    smxx_chunk_arrival_sim(task_queue, interval_ns);
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("bf16_chunk_gemm_nn", &bf16_chunk_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("o2") = std::nullopt, py::arg("probs") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("chunk_weighted_swiglu", &chunk_weighted_swiglu,
          py::arg("o1"), py::arg("probs"), py::arg("o2"),
          py::arg("task_queue"), py::arg("task_idx"));
    m.def("simulate_chunk_arrival", &simulate_chunk_arrival,
          py::arg("task_queue"), py::arg("interval_ns"));
    m.def("chunk_signal_token_done", &chunk_signal_token_done,
          py::arg("row_to_token"), py::arg("token_done"),
          py::arg("task_queue"), py::arg("task_idx"));
    m.attr("num_chunk_task_fields") = static_cast<int>(kNumChunkTaskFields);
#endif
}

} // namespace deep_gemm::overlap
