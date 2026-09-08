#pragma once

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Build the map between the standard unzip order and an atomic order, offline
//
// NOTES: one CTA owns one expert and scans the whole `zip_to_atomic` in the DeepEP order, so a
//        token's rank inside its expert is just the number of earlier tokens of that expert,
//        i.e. a prefix sum over the scan instead of a sort
// NOTES: a slot belongs to this expert iff its atomic row falls into the expert's row range,
//        as the unzipped buffer gives every expert a disjoint (and densely filled) region;
//        that makes `recv_token_indices` unnecessary here
// NOTES: `kOutputAtomic` picks what a row holds, either the unduplicated token in the DeepEP
//        order (`ordered_to_zip`) or the row of the given atomic order (`ordered_to_atomic`)
template <uint32_t kNumThreads, uint32_t kNumTopk, bool kOutputAtomic>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_sort_map_impl(const int* zip_to_atomic, const int* m_start, int* out,
                   uint32_t num_recv_tokens) {
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");
    constexpr uint32_t kNumWarps = kNumThreads / 32;

    // Offline, so the producer is whatever filled the tables in
    cudaGridDependencySynchronize();

    const auto row_begin = static_cast<uint32_t>(m_start[blockIdx.x]);
    const auto row_end = static_cast<uint32_t>(m_start[blockIdx.x + 1]);
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    __shared__ uint32_t smem_warp_count[kNumWarps];

    // One token per thread, in the DeepEP order, so that the ranks come out sorted by token
    uint32_t num_ranked = 0;
    for (uint32_t base = 0; base < num_recv_tokens; base += kNumThreads) {
        const auto token_idx = base + threadIdx.x;

        // At most one slot of a token hits a given expert, hence at most one row
        int atomic_row = -1;
        if (token_idx < num_recv_tokens) {
            #pragma unroll
            for (uint32_t j = 0; j < kNumTopk; ++ j) {
                const auto row = zip_to_atomic[static_cast<uint64_t>(token_idx) * kNumTopk + j];
                if (row >= 0 and static_cast<uint32_t>(row) >= row_begin
                             and static_cast<uint32_t>(row) < row_end)
                    atomic_row = row;
            }
        }
        const bool valid = atomic_row >= 0;

        // Rank in the warp, then in the CTA, on top of what the earlier steps already ranked
        const uint32_t bits = __ballot_sync(0xffffffff, valid);
        const uint32_t warp_prefix = __popc(bits & ((1u << lane_idx) - 1u));
        if (lane_idx == 0)
            smem_warp_count[warp_idx] = __popc(bits);
        __syncthreads();

        uint32_t warp_offset = 0, step_count = 0;
        #pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++ w) {
            warp_offset += w < warp_idx ? smem_warp_count[w] : 0;
            step_count += smem_warp_count[w];
        }

        if (valid)
            out[row_begin + num_ranked + warp_offset + warp_prefix] =
                kOutputAtomic ? atomic_row : static_cast<int>(token_idx);

        num_ranked += step_count;
        __syncthreads();
    }

    // The expert's padded tail, which no token maps to
    DG_TRAP_ONLY_DEVICE_ASSERT(row_begin + num_ranked <= row_end);
    for (uint32_t row = row_begin + num_ranked + threadIdx.x; row < row_end; row += kNumThreads)
        out[row] = -1;
}

}  // namespace deep_gemm
