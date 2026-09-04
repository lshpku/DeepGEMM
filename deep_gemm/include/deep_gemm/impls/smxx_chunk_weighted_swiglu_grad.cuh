#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Backward of the weighted SwiGLU over one chunk of one expert, claimed from the chunk task queue
//
// With `w = silu(gate) * up` and `o2 = w * prob`, one row produces
//   `o2_bwd = w * prob`                                  (the forward activation, recomputed)
//   `do1[2j]   = do2 * prob * up * silu'(gate)`          (the gate half)
//   `do1[2j+1] = do2 * prob * silu(gate)`                (the up half)
//   `sum_j do2[j] * w[j]`, scattered straight into `drecv_probs`  (a whole-row reduction)
//
// NOTES: the chunk's padded tail is zeroed, as `do1`/`o2_bwd` feed the weight gradients, where
//        a `0 * garbage` product of the other operand would still poison the accumulator
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumTopk, uint32_t kNumVecsPerRow,
          uint32_t kNumElemsPerAccess, bool kPrecise>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_weighted_swiglu_grad_impl(const int* task_queue, uint32_t task_idx,
                                     const cutlass::bfloat16_t* __restrict__ o1,
                                     const float* __restrict__ probs,
                                     const cutlass::bfloat16_t* __restrict__ do2,
                                     cutlass::bfloat16_t* __restrict__ o2_bwd,
                                     cutlass::bfloat16_t* __restrict__ do1,
                                     float* __restrict__ drecv_probs,
                                     const int* __restrict__ atomic_to_zip,
                                     const int* __restrict__ zip_to_atomic,
                                     const int64_t* __restrict__ recv_token_indices,
                                     uint32_t m_alignment) {
    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Invalid vector size");
    DG_STATIC_ASSERT(kNumElemsPerAccess % 2 == 0, "A gate/up pair must fit into a vector");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit into a warp");

    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kSumStride = 8;

    // Wait for the producer (the down-grad GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    // Claim the task: `[expert_idx, m_start, m_size, ready]`
    __shared__ int smem_task[4];
    if (threadIdx.x == 0) {
        const auto timed_out = chunk::wait_task_ready(task_queue, task_idx);
        chunk::stage_task(smem_task, chunk::read_task(task_queue, task_idx));
        smem_task[3] = timed_out ? 1 : 0;
    }
    __syncthreads();
    DG_TRAP_ONLY_DEVICE_ASSERT(smem_task[3] == 0);
    const auto task = chunk::load_staged_task(smem_task);
    const auto m_start = static_cast<uint32_t>(task.m_start);
    const auto m_size = static_cast<uint32_t>(task.m_size);
    const auto expert_idx = task.expert_idx;

    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = threadIdx.x / 32;

    // Partition the rows evenly across CTAs
    const auto m_aligned = math::ceil_div(m_size, m_alignment) * m_alignment;
    const auto rows_per_cta = math::ceil_div(m_aligned, kNumSMs);
    const auto row_begin = blockIdx.x * rows_per_cta;
    const auto row_end = min(row_begin + rows_per_cta, m_aligned);

    __shared__ float smem_dprobs_sum[kNumThreads][kSumStride];
    auto* dprobs_sum = smem_dprobs_sum[threadIdx.x];

    // Iterate over m_size, each warp handles one row per loop
    for (uint32_t it = row_begin + warp_idx; it < row_end; it += kNumWarps) {
        const uint32_t bwd_row = m_start + it;
        uint32_t fwd_row = bwd_row;
        uint64_t slot = 0;
        bool owns_slot = false;

        // Map the bwd_row in backward atomic order to forward atomic order
        // NOTES: only the real rows get mapped, padding rows use the same bwd_row as fwd_row.
        //        This is valid because paddings are always on the end of each expert
        if (it < m_size) {
            const auto token_idx = atomic_to_zip[bwd_row];
            DG_TRAP_ONLY_DEVICE_ASSERT(token_idx >= 0);

            slot = static_cast<uint64_t>(token_idx) * kNumTopk + lane_idx;
            if (lane_idx < kNumTopk)
                owns_slot = static_cast<int>(recv_token_indices[slot]) == expert_idx;

            const auto bits = __ballot_sync(0xffffffff, owns_slot);
            DG_TRAP_ONLY_DEVICE_ASSERT(__popc(bits) == 1);

            if (owns_slot)
                fwd_row = zip_to_atomic[slot];
            fwd_row = __shfl_sync(0xffffffff, fwd_row, __ffs(bits) - 1);
        }

        const auto prob = probs[fwd_row];

        #pragma unroll
        for (uint32_t step = 0; step < kSumStride; ++ step)
            dprobs_sum[step] = 0.0f;

        // Iterate over kNumVecsPerRow, each thread handles kSumStride vecs per loop
        for (uint32_t col = lane_idx; col < kNumVecsPerRow; col += 32 * kSumStride) {

            // no unroll
            // NOTES: this kernel must stay free of any stack frame (`cuobjdump -res-usage` has
            //        to report `STACK:0`), otherwise the launch itself is broken and reports an
            //        invalid address (700). This may be a bug of the cubin JIT mechanism. Here
            //        we omit the unroll pragma to avoid register spilling
            for (uint32_t step = 0; step < kSumStride; ++ step) {
                const auto idx = step * 32 + col;
                if (idx >= kNumVecsPerRow)
                    continue;

                const auto bwd_off = static_cast<uint64_t>(bwd_row) * kNumVecsPerRow + idx;
                const auto fwd_off = static_cast<uint64_t>(fwd_row) * kNumVecsPerRow + idx;

                vec_t do2_vec = __ldg(reinterpret_cast<const vec_t*>(do2) + bwd_off);
                vec_t lo_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + fwd_off * 2);
                vec_t hi_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + fwd_off * 2 + 1);
                vec_t o2_vec, d_lo_vec, d_hi_vec;

                const auto* grad = reinterpret_cast<const cutlass::bfloat16_t*>(&do2_vec);
                const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec);
                const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec);
                auto* o2_out = reinterpret_cast<cutlass::bfloat16_t*>(&o2_vec);
                auto* d_lo = reinterpret_cast<cutlass::bfloat16_t*>(&d_lo_vec);
                auto* d_hi = reinterpret_cast<cutlass::bfloat16_t*>(&d_hi_vec);

                #pragma unroll
                for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
                    const auto* pair = i < kNumPairsPerVec ? lo : hi;
                    auto* d_pair = i < kNumPairsPerVec ? d_lo : d_hi;
                    const auto k = (i % kNumPairsPerVec) * 2;
                    const auto g = static_cast<float>(pair[k]);
                    const auto u = static_cast<float>(pair[k + 1]);
                    const auto d = static_cast<float>(grad[i]);

                    const auto s = kPrecise ? 1.0f / (1.0f + expf(-g))
                                            : __fdividef(1.0f, 1.0f + __expf(-g));
                    const auto silu = g * s;
                    const auto w = silu * u;

                    const auto dw = d * prob;
                    const auto du = dw * silu;
                    const auto dg = dw * u * s * (1.0f + g * (1.0f - s));

                    dprobs_sum[step] += d * w;

                    o2_out[i] = static_cast<cutlass::bfloat16_t>(w * prob);
                    d_pair[k] = static_cast<cutlass::bfloat16_t>(dg);
                    d_pair[k + 1] = static_cast<cutlass::bfloat16_t>(du);
                }

                *(reinterpret_cast<vec_t*>(o2_bwd) + bwd_off) = o2_vec;
                *(reinterpret_cast<vec_t*>(do1) + bwd_off * 2) = d_lo_vec;
                *(reinterpret_cast<vec_t*>(do1) + bwd_off * 2 + 1) = d_hi_vec;
            }
        }

        // Reduce dprobs. Strictly fold the back half to the front half
        #pragma unroll
        for (uint32_t stride = kSumStride / 2; stride > 0; stride /= 2) {
            #pragma unroll
            for (uint32_t i = 0; i < stride; ++ i) {
                dprobs_sum[i] += dprobs_sum[i + stride];
            }
        }
        auto dprobs = dprobs_sum[0];
        #pragma unroll
        for (uint32_t stride = 16; stride > 0; stride /= 2) {
            dprobs += __shfl_down_sync(0xffffffff, dprobs, stride);
        }
        dprobs = __shfl_sync(0xffffffff, dprobs, 0);
        if (owns_slot)
            drecv_probs[slot] = dprobs;
    }
}

}  // namespace deep_gemm
