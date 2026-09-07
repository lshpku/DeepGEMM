#pragma once

#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_chunk_token_done.hpp"
#include "../jit_kernels/impls/smxx_chunk_weighted_swiglu.hpp"
#include "../jit_kernels/impls/smxx_chunk_weighted_swiglu_grad.hpp"
#include "../jit_kernels/impls/smxx_chunk_zip.hpp"
#endif

namespace deep_gemm::overlap {

#if DG_TENSORMAP_COMPATIBLE

// Chunk-wise BF16 GEMM for the fine-grained compute-communication overlap
// Shape must be `[M, K] @ [G, N, K].mT`, where the task queue holds
// `[expert_idx, m_start, m_size, ready]` per task, and one call computes one task
// NOTES: passing `o2` and `probs` fuses the weighted SwiGLU into the epilogue, which needs
//        `b`'s gate/up rows fully interleaved, i.e. `[gate[0], up[0], gate[1], up[1], ...]`
//        (the Paddle MoE convention), so that a channel's gate and up land side by side;
//        `d` then keeps the linear output (in that interleaved order) for the backward pass
static void bf16_chunk_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d,
                               const torch::Tensor& task_queue,
                               const int& task_idx,
                               const std::optional<torch::Tensor>& o2,
                               const std::optional<torch::Tensor>& probs,
                               const std::string& compiled_dims) {
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
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
    sm100_bf16_chunk_gemm(a, b, d, task_queue, task_idx,
                          num_groups, m, n, k, major_a, major_b, compiled_dims, o2, probs);
}

// The same GEMM with a `[G, K, N]` weight, which the forward stores and passes along
// NOTES: the only difference is `B`'s major type, so both variants share one kernel template
static void bf16_chunk_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d,
                               const torch::Tensor& task_queue,
                               const int& task_idx,
                               const std::optional<torch::Tensor>& o2,
                               const std::optional<torch::Tensor>& probs,
                               const std::string& compiled_dims) {
    bf16_chunk_gemm_nt(a, b.transpose(1, 2), d, task_queue, task_idx, o2, probs, compiled_dims);
}

// Weighted SwiGLU for one chunk task: `o2[:, j] = silu(o1[:, 2j]) * o1[:, 2j + 1] * probs`
static void chunk_weighted_swiglu(const torch::Tensor& o1, const torch::Tensor& probs,
                                  const torch::Tensor& o2,
                                  const torch::Tensor& task_queue,
                                  const int& task_idx,
                                  const bool& precise,
                                  const bool& interleaved) {
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

    smxx_chunk_weighted_swiglu(o1, probs, o2, task_queue, task_idx, precise, interleaved);
}

// Backward of the weighted SwiGLU for one chunk task: `(o1, probs, do2) -> (o2_bwd, do1, drecv_probs)`
// NOTES: `o1` and `probs` are forward activations, so they are indexed in the forward atomic
//        order, while `do2` and both row outputs are in the backward one; a row is mapped
//        across the two by its unduplicated token, i.e. `atomic_to_zip` (backward) gives the
//        token, the task gives the expert, and `zip_to_atomic` (forward) gives the forward row
// NOTES: `o2_bwd` is the forward `o2` recomputed for the `w_down` weight gradient, and with
//        `precise` it reproduces the forward bit for bit
// NOTES: the `probs` gradient needs no reduction over the experts, as a row owns exactly one
//        `(token, slot)` pair, so it lands straight in its final `[num_recv_tokens, num_topk]`
//        slot; the slots of a token's non-local experts are never written, hence `drecv_probs`
//        must arrive zeroed
// NOTES: the chunk's padded tail is zeroed in `do1`/`o2_bwd`, as they feed the weight gradients,
//        where the other operand's garbage rows would poison the result
static void chunk_weighted_swiglu_grad(const torch::Tensor& o1, const torch::Tensor& probs,
                                       const torch::Tensor& do2,
                                       const torch::Tensor& o2_bwd, const torch::Tensor& do1,
                                       const torch::Tensor& drecv_probs,
                                       const torch::Tensor& atomic_to_zip,
                                       const torch::Tensor& zip_to_atomic,
                                       const torch::Tensor& recv_token_indices,
                                       const torch::Tensor& task_queue,
                                       const int& task_idx,
                                       const bool& precise,
                                       const bool& interleaved) {
    // The row-indexed tensors all share the unzipped layout, which is the same in both orders
    const auto [m, n2] = get_shape<2>(o1);
    const auto [m_, n] = get_shape<2>(do2);
    DG_HOST_ASSERT(m == m_ and n2 == n * 2);
    for (const auto& t: {o1, do1}) {
        DG_HOST_ASSERT(t.sizes() == o1.sizes() and t.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(t.is_contiguous());
    }
    for (const auto& t: {do2, o2_bwd}) {
        DG_HOST_ASSERT(t.sizes() == do2.sizes() and t.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(t.is_contiguous());
    }
    DG_HOST_ASSERT(probs.is_contiguous() and probs.dim() == 1);
    DG_HOST_ASSERT(static_cast<int>(probs.numel()) == m);
    DG_HOST_ASSERT(probs.scalar_type() == torch::kFloat);

    // `atomic_to_zip` follows the rows, the top-k tables and `drecv_probs` the tokens
    DG_HOST_ASSERT(atomic_to_zip.is_contiguous() and atomic_to_zip.dim() == 1);
    DG_HOST_ASSERT(static_cast<int>(atomic_to_zip.numel()) == m);
    DG_HOST_ASSERT(atomic_to_zip.scalar_type() == torch::kInt);
    const auto [num_recv_tokens, num_topk] = get_shape<2>(zip_to_atomic);
    const auto [num_recv_tokens_, num_topk_] = get_shape<2>(recv_token_indices);
    const auto [num_recv_tokens__, num_topk__] = get_shape<2>(drecv_probs);
    DG_HOST_ASSERT(num_recv_tokens_ == num_recv_tokens and num_topk_ == num_topk);
    DG_HOST_ASSERT(num_recv_tokens__ == num_recv_tokens and num_topk__ == num_topk);
    DG_HOST_ASSERT(num_topk <= 32);
    DG_HOST_ASSERT(zip_to_atomic.is_contiguous() and zip_to_atomic.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(recv_token_indices.is_contiguous() and recv_token_indices.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(drecv_probs.is_contiguous() and drecv_probs.scalar_type() == torch::kFloat);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    smxx_chunk_weighted_swiglu_grad(o1, probs, do2, o2_bwd, do1, drecv_probs,
                                    atomic_to_zip, zip_to_atomic, recv_token_indices,
                                    task_queue, task_idx, precise, interleaved);
}

// Publish the token completion of one chunk, to be issued right after its down GEMM
// NOTES: `atomic_to_zip` maps an unzipped row to the unduplicated token in the DeepEP order
//        (`-1` for padding), and `num_valid_topk` holds how many local experts a token has;
//        a token is pushed into `zip_task_queue` once all of its experts have counted it,
//        so `zip` only has to poll the queue for entries other than `-1`
static void chunk_signal_token_done(const torch::Tensor& atomic_to_zip,
                                    const torch::Tensor& num_valid_topk,
                                    const torch::Tensor& token_done,
                                    const torch::Tensor& zip_task_queue,
                                    const torch::Tensor& zip_queue_tail,
                                    const torch::Tensor& task_queue,
                                    const int& task_idx) {
    DG_HOST_ASSERT(atomic_to_zip.is_contiguous() and atomic_to_zip.dim() == 1);
    DG_HOST_ASSERT(num_valid_topk.is_contiguous() and num_valid_topk.dim() == 1);
    DG_HOST_ASSERT(token_done.is_contiguous() and token_done.dim() == 1);
    DG_HOST_ASSERT(zip_task_queue.is_contiguous() and zip_task_queue.dim() == 1);
    DG_HOST_ASSERT(zip_queue_tail.is_contiguous() and zip_queue_tail.numel() == 1);
    DG_HOST_ASSERT(atomic_to_zip.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(num_valid_topk.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(token_done.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(zip_task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(zip_queue_tail.scalar_type() == torch::kInt);

    // All the token-indexed tables use the DeepEP order, so `num_recv_tokens` rows;
    // the queue holds every token at most once, hence the same length
    const auto num_recv_tokens = token_done.numel();
    DG_HOST_ASSERT(num_valid_topk.numel() == num_recv_tokens);
    DG_HOST_ASSERT(zip_task_queue.numel() == num_recv_tokens);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    smxx_chunk_token_done(atomic_to_zip, num_valid_topk, token_done,
                          zip_task_queue, zip_queue_tail, task_queue, task_idx);
}

// Publish the token completion of one chunk and zip whatever it finishes, replacing the
// `chunk_signal_token_done` + standalone zip pair with one kernel on the compute stream
// NOTES: `combine_input` and `zip_done` are updated in place, the former holding the sum of
//        `o3` over a token's local experts in the DeepEP order, the latter flagging the rows
//        that `combine` may send; both are indexed by the unduplicated token
// NOTES: the experts are summed in ascending order of the expert index, which `recv_token_indices`
//        provides, so that the result does not depend on the chunk arrival order
static void chunk_zip(const torch::Tensor& o3, const torch::Tensor& combine_input,
                      const torch::Tensor& atomic_to_zip, const torch::Tensor& zip_to_atomic,
                      const torch::Tensor& recv_token_indices,
                      const torch::Tensor& num_valid_topk,
                      const torch::Tensor& token_done, const torch::Tensor& zip_done,
                      const torch::Tensor& task_queue,
                      const int& task_idx, const int& chunk_size) {
    // Type and shape checks
    const auto [num_unzipped_tokens, hidden_size] = get_shape<2>(o3);
    const auto [num_recv_tokens, hidden_size_] = get_shape<2>(combine_input);
    DG_HOST_ASSERT(hidden_size == hidden_size_);
    DG_HOST_ASSERT(o3.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(combine_input.scalar_type() == torch::kBFloat16);
    check_major_type_cd(o3);
    check_major_type_cd(combine_input);

    // The row-indexed table follows `o3`, and the token-indexed ones follow `combine_input`
    DG_HOST_ASSERT(atomic_to_zip.is_contiguous() and atomic_to_zip.dim() == 1);
    DG_HOST_ASSERT(atomic_to_zip.numel() == num_unzipped_tokens);
    DG_HOST_ASSERT(atomic_to_zip.scalar_type() == torch::kInt);
    for (const auto& table: {num_valid_topk, token_done, zip_done}) {
        DG_HOST_ASSERT(table.is_contiguous() and table.dim() == 1);
        DG_HOST_ASSERT(table.numel() == num_recv_tokens);
        DG_HOST_ASSERT(table.scalar_type() == torch::kInt);
    }

    // The two top-k tables share their layout, and `recv_token_indices` is the only one
    // that carries the expert indices, hence the ordering of the sum
    const auto [num_recv_tokens_, num_topk] = get_shape<2>(zip_to_atomic);
    const auto [num_recv_tokens__, num_topk_] = get_shape<2>(recv_token_indices);
    DG_HOST_ASSERT(num_recv_tokens_ == num_recv_tokens and num_recv_tokens__ == num_recv_tokens);
    DG_HOST_ASSERT(num_topk_ == num_topk and num_topk <= 32);
    DG_HOST_ASSERT(zip_to_atomic.is_contiguous() and zip_to_atomic.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(recv_token_indices.is_contiguous() and recv_token_indices.scalar_type() == torch::kLong);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    // The chunk size only sizes the CTA-local queue, so any upper bound works
    DG_HOST_ASSERT(chunk_size > 0);

    smxx_chunk_zip(o3, combine_input, atomic_to_zip, zip_to_atomic, recv_token_indices,
                   num_valid_topk, token_done, zip_done, task_queue, task_idx, chunk_size);
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("bf16_chunk_gemm_nt", &bf16_chunk_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("o2") = std::nullopt, py::arg("probs") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_chunk_gemm_nn", &bf16_chunk_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("o2") = std::nullopt, py::arg("probs") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("chunk_weighted_swiglu", &chunk_weighted_swiglu,
          py::arg("o1"), py::arg("probs"), py::arg("o2"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("precise") = false, py::arg("interleaved") = false);
    m.def("chunk_weighted_swiglu_grad", &chunk_weighted_swiglu_grad,
          py::arg("o1"), py::arg("probs"), py::arg("do2"),
          py::arg("o2_bwd"), py::arg("do1"), py::arg("drecv_probs"),
          py::arg("atomic_to_zip"), py::arg("zip_to_atomic"), py::arg("recv_token_indices"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("precise") = false, py::arg("interleaved") = false);
    m.def("chunk_signal_token_done", &chunk_signal_token_done,
          py::arg("atomic_to_zip"), py::arg("num_valid_topk"), py::arg("token_done"),
          py::arg("zip_task_queue"), py::arg("zip_queue_tail"),
          py::arg("task_queue"), py::arg("task_idx"));
    m.def("chunk_zip", &chunk_zip,
          py::arg("o3"), py::arg("combine_input"),
          py::arg("atomic_to_zip"), py::arg("zip_to_atomic"),
          py::arg("recv_token_indices"), py::arg("num_valid_topk"),
          py::arg("token_done"), py::arg("zip_done"),
          py::arg("task_queue"), py::arg("task_idx"), py::arg("chunk_size"));
    m.attr("num_chunk_task_fields") = static_cast<int>(kNumChunkTaskFields);
#endif
}

} // namespace deep_gemm::overlap
