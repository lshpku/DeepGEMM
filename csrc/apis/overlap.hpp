#pragma once

#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/smxx_chunk_token_done.hpp"
#include "../jit_kernels/impls/smxx_chunk_weighted_swiglu.hpp"
#include "../jit_kernels/impls/smxx_chunk_weighted_swiglu_grad.hpp"
#include "../jit_kernels/impls/smxx_chunk_zip.hpp"
#include "../jit_kernels/impls/smxx_requant.hpp"
#include "../jit_kernels/impls/smxx_sort_map.hpp"
#include "../jit_kernels/impls/smxx_token_gather.hpp"
#endif

namespace deep_gemm::overlap {

#if DG_TENSORMAP_COMPATIBLE

// Chunk-wise BF16 GEMM for the fine-grained compute-communication overlap
// Shape must be `[M, K] @ [G, N, K].mT`, where the task queue holds
// `[expert_idx, m_start, m_size, ready]` per task, and one call computes one task
static void bf16_chunk_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d,
                               const torch::Tensor& task_queue,
                               const int& task_idx,
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

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 and "Chunk GEMM only supports SM100 for now");
    sm100_bf16_chunk_gemm(a, b, d, task_queue, task_idx,
                          num_groups, m, n, k, major_a, major_b, compiled_dims);
}

// The same GEMM with a `[G, K, N]` weight, which the forward stores and passes along
// NOTES: the only difference is `B`'s major type, so both variants share one kernel template
static void bf16_chunk_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                               const torch::Tensor& d,
                               const torch::Tensor& task_queue,
                               const int& task_idx,
                               const std::string& compiled_dims) {
    bf16_chunk_gemm_nt(a, b.transpose(1, 2), d, task_queue, task_idx, compiled_dims);
}

// Chunk-wise FP8 (1D1D, packed UE8M0) GEMM, the counterpart of `bf16_chunk_gemm_nt`
// Shape must be `[M, K] @ [G, N, K].mT`, with both scaling factors already in the layout
// the non-chunk kernels want: INT32 packed UE8M0, MN-major and TMA aligned
// NOTES: the scale of A is NOT transposed per chunk but over the whole `M`, as one TMA
//        descriptor can only carry one row stride; a per-chunk stride would need one
//        descriptor per chunk, while the global one costs the very same TMA traffic
//        (a block still loads `BLOCK_M` contiguous scales, i.e. 512B, per K-block group)
static void fp8_chunk_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                              const std::pair<torch::Tensor, torch::Tensor>& b,
                              const torch::Tensor& d,
                              const torch::Tensor& task_queue,
                              const int& task_idx,
                              const std::string& compiled_dims) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10 and "Chunk GEMM only supports SM100 for now");
    const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    // Both scales must already be packed UE8M0 and MN-major, as transposing them
    // per launch would defeat the whole point of the chunk pipeline
    constexpr int gran_k = 128;
    check_sf_layout(a.second, m, k, 1, gran_k, std::nullopt, true, false, torch::kInt);
    check_sf_layout(b.second, n, k, 1, gran_k, num_groups, true, false, torch::kInt);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    // D must be N-major
    check_major_type_cd(d);

    sm100_fp8_chunk_gemm_1d1d(a.first, a.second, b.first, b.second, d, task_queue, task_idx,
                              num_groups, m, n, k, gran_k, gran_k, major_a, major_b,
                              compiled_dims);
}

// Weighted SwiGLU for one chunk task: `o2[:, j] = silu(o1[:, 2j]) * o1[:, 2j + 1] * probs`
// NOTES: passing `o2_scales` switches `o2` to FP8 with one packed UE8M0 scale per 128 columns,
//        which reproduces `fuse_weighted_swiglu_fp8_quant` bit for bit; the scale buffer is
//        MN-major over the whole `M`, exactly what the chunk GEMM reads
static void chunk_weighted_swiglu(const torch::Tensor& o1, const torch::Tensor& probs,
                                  const torch::Tensor& o2,
                                  const torch::Tensor& task_queue,
                                  const int& task_idx,
                                  const int& chunk_size,
                                  const bool& precise,
                                  const bool& interleaved,
                                  const std::optional<torch::Tensor>& o2_scales) {
    // Type and shape checks
    const auto [m, n2] = get_shape<2>(o1);
    const auto [m_, n] = get_shape<2>(o2);
    DG_HOST_ASSERT(m == m_ and n2 == n * 2);
    DG_HOST_ASSERT(o1.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(o2.scalar_type() == (o2_scales.has_value() ?
                                        torch::kFloat8_e4m3fn : torch::kBFloat16));
    DG_HOST_ASSERT(probs.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(static_cast<int>(probs.numel()) == m);
    DG_HOST_ASSERT(o1.is_contiguous() and o2.is_contiguous() and probs.is_contiguous());

    // The scale is packed UE8M0 and MN-major, i.e. the layout the FP8 chunk GEMM wants
    if (o2_scales.has_value())
        check_sf_layout(o2_scales.value(), m, n, 1, 128, std::nullopt, true, false, torch::kInt);

    // The task queue is `[num_tasks, kNumChunkTaskFields]` on the device
    DG_HOST_ASSERT(task_queue.is_contiguous());
    DG_HOST_ASSERT(task_queue.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(task_queue.dim() == 2 and task_queue.size(1) == kNumChunkTaskFields);
    DG_HOST_ASSERT(0 <= task_idx and task_idx < task_queue.size(0));

    // The chunk size only decides the number of blocks, so any upper bound works
    DG_HOST_ASSERT(chunk_size > 0);

    smxx_chunk_weighted_swiglu(o1, probs, o2, o2_scales, task_queue, task_idx, chunk_size,
                               precise, interleaved);
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
                                       const int& chunk_size,
                                       const bool& precise,
                                       const bool& interleaved,
                                       const std::optional<torch::Tensor>& o2_bwd_scales,
                                       const std::optional<torch::Tensor>& do1_scales) {
    // The row-indexed tensors all share the unzipped layout, which is the same in both orders
    const auto [m, n2] = get_shape<2>(o1);
    const auto [m_, n] = get_shape<2>(do2);
    DG_HOST_ASSERT(m == m_ and n2 == n * 2);
    const auto quant = o2_bwd_scales.has_value();
    DG_HOST_ASSERT(quant == do1_scales.has_value());
    const auto out_dtype = quant ? torch::kFloat8_e4m3fn : torch::kBFloat16;
    DG_HOST_ASSERT(o1.sizes() == do1.sizes() and o1.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(do2.sizes() == o2_bwd.sizes() and do2.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(do1.scalar_type() == out_dtype and o2_bwd.scalar_type() == out_dtype);
    for (const auto& t: {o1, do1, do2, o2_bwd})
        DG_HOST_ASSERT(t.is_contiguous());

    // Both scales are packed UE8M0 and MN-major, i.e. the layout the FP8 chunk GEMM wants
    if (quant) {
        check_sf_layout(o2_bwd_scales.value(), m, n, 1, 128, std::nullopt, true, false, torch::kInt);
        check_sf_layout(do1_scales.value(), m, n2, 1, 128, std::nullopt, true, false, torch::kInt);
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

    // The chunk size only decides the number of blocks, so any upper bound works
    DG_HOST_ASSERT(chunk_size > 0);

    smxx_chunk_weighted_swiglu_grad(o1, probs, do2, o2_bwd, do1, drecv_probs,
                                    atomic_to_zip, zip_to_atomic, recv_token_indices,
                                    o2_bwd_scales, do1_scales, task_queue, task_idx, chunk_size,
                                    precise, interleaved);
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

// Requantize one wgrad operand along the token dim, fused with the reorder into the standard
// (sorted) order and the re-alignment of every expert's token count to 512
// NOTES: the wgrad GEMM contracts over tokens, so its scale covers `(128 token, 1 channel)`
//        instead of the main branch's `(1 token, 128 channel)`; both are powers of two, so the
//        round trip only shifts exponents
// NOTES: `index` maps a destination row to its source row, `-1` giving a zeroed row, and the
//        returned scale is the contiguous `[sum_k / 512, hidden]` packed UE8M0 that
//        `k_grouped_fp8_gemm_tn_contiguous` reads
static std::tuple<torch::Tensor, torch::Tensor> requant_wgrad_input(const torch::Tensor& src,
                                                                   const torch::Tensor& src_scales,
                                                                   const torch::Tensor& index) {
    // Type and shape checks
    const auto [num_src_rows, hidden] = get_shape<2>(src);
    DG_HOST_ASSERT(src.scalar_type() == torch::kFloat8_e4m3fn and src.is_contiguous());
    DG_HOST_ASSERT(index.dim() == 1 and index.is_contiguous());
    DG_HOST_ASSERT(index.scalar_type() == torch::kInt);

    // The source scale is the main branch's, i.e. packed UE8M0 and MN-major
    // NOTES: the row stride is read as is instead of being required TMA aligned, as this
    //        kernel loads the exponents with plain loads; the source may be `recv_x`'s scale,
    //        whose row count is not a multiple of four
    check_sf_layout(src_scales, num_src_rows, hidden, 1, 128, std::nullopt, false, false, torch::kInt);
    DG_HOST_ASSERT(src_scales.stride(-2) == 1);

    // Every expert's region is 512-aligned, so a packed scale int32 never straddles two experts
    const auto num_dst_rows = static_cast<int>(index.numel());
    DG_HOST_ASSERT(num_dst_rows > 0 and num_dst_rows % 512 == 0);
    DG_HOST_ASSERT(hidden % 512 == 0);

    const auto out = torch::empty({num_dst_rows, hidden}, src.options());
    const auto out_scales = torch::empty({num_dst_rows / 512, hidden},
                                         at::TensorOptions().device(src.device()).dtype(torch::kInt));
    smxx_requant(src, src_scales, index, out, out_scales, num_dst_rows, hidden);
    return {out, out_scales};
}

// Shared checks of the two offline order maps, both of which only scan one `zip_to_atomic`
// NOTES: this runs after the whole dispatch, so there is no consistency logic at all
static std::tuple<torch::Tensor, torch::Tensor> sort_map(
        const torch::Tensor& zip_to_atomic,
        const torch::Tensor& m_start,
        const int& num_output_rows,
        const std::optional<torch::Tensor>& m_start_out) {
    // `zip_to_atomic` uses the DeepEP order, matching `recv_token_indices` slot by slot
    const auto [num_recv_tokens, num_topk] = get_shape<2>(zip_to_atomic);
    DG_HOST_ASSERT(num_recv_tokens > 0 and num_topk > 0);
    DG_HOST_ASSERT(zip_to_atomic.is_contiguous() and zip_to_atomic.scalar_type() == torch::kInt);

    // `m_start` holds the row offset of every expert's region plus `num_unzipped_tokens`,
    // which is all this needs to tell the experts apart and to fill their padded tails
    // NOTES: `m_start_out` is the destination layout's, which differs from the source's when
    //        the wgrad wants every expert re-aligned to 512 tokens
    for (const auto& t: {m_start, m_start_out.value_or(m_start)}) {
        DG_HOST_ASSERT(t.is_contiguous() and t.dim() == 1);
        DG_HOST_ASSERT(t.numel() == m_start.numel() and t.scalar_type() == torch::kInt);
    }
    DG_HOST_ASSERT(m_start.numel() >= 2);
    DG_HOST_ASSERT(num_output_rows > 0);

    const auto ordered_to_zip = torch::empty({num_output_rows}, zip_to_atomic.options());
    const auto ordered_to_atomic = torch::empty({num_output_rows}, zip_to_atomic.options());
    smxx_sort_map(zip_to_atomic, m_start, m_start_out.value_or(m_start),
                  ordered_to_zip, ordered_to_atomic);
    return {ordered_to_zip, ordered_to_atomic};
}

// `paddle.gather(x, index, axis=0)` with `-1` gathering a zeroed row instead of the last one
// NOTES: the rows are copied as 16B vectors, so `x` must be vector aligned in both its base
//        pointer and its row length, which any real token buffer is
static torch::Tensor token_gather(const torch::Tensor& x, const torch::Tensor& index) {
    const auto [num_rows, row_size] = get_shape<2>(x);
    DG_HOST_ASSERT(num_rows > 0 and row_size > 0 and x.is_contiguous());
    DG_HOST_ASSERT(index.is_contiguous() and index.dim() == 1);
    DG_HOST_ASSERT(index.numel() > 0 and index.scalar_type() == torch::kInt);

    const auto out = torch::empty({static_cast<int>(index.numel()), row_size}, x.options());
    smxx_token_gather(x, index, out, row_size * static_cast<int>(x.element_size()));
    return out;
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("bf16_chunk_gemm_nt", &bf16_chunk_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("compiled_dims") = "nk");
    m.def("bf16_chunk_gemm_nn", &bf16_chunk_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("compiled_dims") = "nk");
    m.def("fp8_chunk_gemm_nt", &fp8_chunk_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("task_queue"), py::arg("task_idx"),
          py::arg("compiled_dims") = "nk");
    m.def("chunk_weighted_swiglu", &chunk_weighted_swiglu,
          py::arg("o1"), py::arg("probs"), py::arg("o2"),
          py::arg("task_queue"), py::arg("task_idx"), py::arg("chunk_size"),
          py::arg("precise") = false, py::arg("interleaved") = false,
          py::arg("o2_scales") = std::nullopt);
    m.def("chunk_weighted_swiglu_grad", &chunk_weighted_swiglu_grad,
          py::arg("o1"), py::arg("probs"), py::arg("do2"),
          py::arg("o2_bwd"), py::arg("do1"), py::arg("drecv_probs"),
          py::arg("atomic_to_zip"), py::arg("zip_to_atomic"), py::arg("recv_token_indices"),
          py::arg("task_queue"), py::arg("task_idx"), py::arg("chunk_size"),
          py::arg("precise") = false, py::arg("interleaved") = false,
          py::arg("o2_bwd_scales") = std::nullopt, py::arg("do1_scales") = std::nullopt);
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
    m.def("sort_map", &sort_map,
          py::arg("zip_to_atomic"), py::arg("m_start"), py::arg("num_output_rows"),
          py::arg("m_start_out") = std::nullopt);
    m.def("token_gather", &token_gather,
          py::arg("x"), py::arg("index"));
    m.def("requant_wgrad_input", &requant_wgrad_input,
          py::arg("src"), py::arg("src_scales"), py::arg("index"));
#endif
}

} // namespace deep_gemm::overlap
