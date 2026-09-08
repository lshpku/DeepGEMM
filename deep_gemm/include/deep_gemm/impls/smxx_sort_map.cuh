#pragma once

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

static CUTLASS_DEVICE uint32_t
warp_inclusive_cumsum(uint32_t value) {
    const auto lane_idx = ptx::get_lane_idx();
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t other = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += other;
    }
    return value;
}

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

    const auto row_begin = m_start[blockIdx.x];
    const auto row_end = m_start[blockIdx.x + 1];
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    const auto lane_active = kNumWarps == 32 || lane_idx < kNumWarps;
    __shared__ uint32_t smem_warp_count[32];
    __shared__ uint32_t smem_block_count;

    constexpr uint32_t kNumVecsPerThread = 2;
    constexpr auto kNumElemsPerAccess = static_cast<uint32_t>(sizeof(int4) / sizeof(int));
    constexpr auto kNumElemsPerStep = kNumThreads * kNumVecsPerThread * kNumElemsPerAccess;
    const auto num_slots = num_recv_tokens * kNumTopk;
    const auto num_slots_floor = num_slots / kNumElemsPerStep * kNumElemsPerStep;
    const auto num_slots_ceil = (num_slots + kNumThreads - 1) / kNumThreads * kNumThreads;
    uint32_t global_count = 0;

    // Iterate over the divisible part using vectorized loads
    for (uint32_t base = threadIdx.x * kNumVecsPerThread * kNumElemsPerAccess;
         base < num_slots_floor; base += kNumElemsPerStep) {
        int4 rows_vec[kNumVecsPerThread];
        uint32_t local_count = 0;

        #pragma unroll
        for (uint32_t i = 0; i < kNumVecsPerThread; ++ i) {
            const auto slot = base + i * kNumElemsPerAccess;
            rows_vec[i] = *reinterpret_cast<const int4*>(zip_to_atomic + slot);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumVecsPerThread; ++ i) {
            const auto* rows = reinterpret_cast<const int*>(rows_vec + i);
            #pragma unroll
            for (uint32_t j = 0; j < kNumElemsPerAccess; ++ j) {
                const auto row = rows[j];
                local_count += row >= row_begin && row < row_end;
            }
        }

        // warp cumsum
        const auto warp_prefix_inclusive = warp_inclusive_cumsum(local_count);
        const auto warp_prefix = warp_prefix_inclusive - local_count;
        if (lane_idx == 31)
            smem_warp_count[warp_idx] = warp_prefix_inclusive;
        __syncthreads();

        // block cumsum
        if (warp_idx == 0) {
            const auto warp_count = lane_active ? smem_warp_count[lane_idx] : 0u;
            const auto block_prefix_inclusive = warp_inclusive_cumsum(warp_count);
            smem_warp_count[lane_idx] = block_prefix_inclusive - warp_count;
            if (lane_idx == kNumWarps - 1)
                smem_block_count = block_prefix_inclusive;
        }
        __syncthreads();

        // global cumsum
        auto count = global_count + smem_warp_count[warp_idx] + warp_prefix;
        global_count += smem_block_count;
        __syncthreads();

        #pragma unroll
        for (uint32_t i = 0; i < kNumVecsPerThread; ++ i) {
            const auto* rows = reinterpret_cast<const int*>(rows_vec + i);
            #pragma unroll
            for (uint32_t j = 0; j < kNumElemsPerAccess; ++ j) {
                const auto row = rows[j];
                if (row >= row_begin && row < row_end) {
                    const auto slot = base + i * kNumElemsPerAccess + j;
                    const auto token_idx = slot / kNumTopk;
                    out[row_begin + count] =
                        kOutputAtomic ? row : static_cast<int>(token_idx);
                    ++ count;
                }
            }
        }
    }

    // Handle the remaining part
    // NOTES: all threads should participate
    for (uint32_t slot = num_slots_floor + threadIdx.x; slot < num_slots_ceil;
         slot += kNumThreads) {
        const auto row = slot < num_slots ? zip_to_atomic[slot] : -1;
        const auto valid = row >= row_begin && row < row_end;

        // warp cumsum
        const auto bits = __ballot_sync(0xffffffff, valid);
        const auto warp_prefix = __popc(bits & ((1u << lane_idx) - 1u));
        if (lane_idx == 0)
            smem_warp_count[warp_idx] = __popc(bits);
        __syncthreads();

        // block cumsum
        if (warp_idx == 0) {
            const auto warp_count = lane_active ? smem_warp_count[lane_idx] : 0u;
            const auto block_prefix_inclusive = warp_inclusive_cumsum(warp_count);
            smem_warp_count[lane_idx] = block_prefix_inclusive - warp_count;
            if (lane_idx == kNumWarps - 1)
                smem_block_count = block_prefix_inclusive;
        }
        __syncthreads();

        // global cumsum
        const auto count = global_count + smem_warp_count[warp_idx] + warp_prefix;
        global_count += smem_block_count;
        __syncthreads();

        if (valid) {
            const auto token_idx = slot / kNumTopk;
            out[row_begin + count] =
                kOutputAtomic ? row : static_cast<int>(token_idx);
        }
    }

    if (threadIdx.x == 0) {
        if (row_begin + global_count > row_end) {
            printf("sort_map global_count overflow: expert_idx=%u, global_count=%u, "
                   "row_begin=%d, row_end=%d\n", blockIdx.x, global_count, row_begin, row_end);
            DG_TRAP_ONLY_DEVICE_ASSERT(0);
        }
    }

    // The expert's padded tail, which no token maps to
    for (uint32_t row = row_begin + global_count + threadIdx.x; row < row_end; row += kNumThreads)
        out[row] = -1;
}

}  // namespace deep_gemm
