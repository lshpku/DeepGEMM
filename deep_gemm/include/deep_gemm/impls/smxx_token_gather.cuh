#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Row gather with `-1` meaning a zeroed row, i.e. `paddle.gather` for a padded index
// NOTES: the rows are traversed as 16B vectors, so both pointers must be vector aligned and a
//        row must be a whole number of vectors, which any real token buffer satisfies
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumVecsPerRow>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_token_gather_impl(const int4* x, const int* index, int4* out, uint32_t num_rows) {
    cudaGridDependencySynchronize();

    constexpr uint32_t kGridStride = kNumSMs * kNumThreads;
    const auto num_vecs = num_rows * kNumVecsPerRow;

    // Persistent: a fixed number of CTAs strides over the flattened output
    // NOTES: the row length is a template parameter, so the row/column split of a flat index
    //        is a shift instead of a division for the usual power-of-two row size
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < num_vecs; i += kGridStride) {
        const auto src_row = index[i / kNumVecsPerRow];
        out[i] = src_row >= 0
               ? __ldg(x + static_cast<uint64_t>(src_row) * kNumVecsPerRow + i % kNumVecsPerRow)
               : int4{0, 0, 0, 0};
    }
}

}  // namespace deep_gemm
