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
template <uint32_t kNumThreads, uint32_t kNumTopk, uint32_t kNumVecsPerRow,
          uint32_t kNumElemsPerAccess, bool kPrecise, bool kInterleaved>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
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
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kSumStride = 8;

    // Wait for the producer (the down-grad GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    // Claim the task: `[expert_idx, m_start, m_size, ready]` without fence
    const auto task = __ldg(reinterpret_cast<const int4*>(task_queue) + task_idx);
    const auto expert_idx = task.x;
    const auto m_start = static_cast<uint32_t>(task.y);
    const auto m_size = static_cast<uint32_t>(task.z);
    if (threadIdx.x == 0) {
        const auto ready = static_cast<bool>(task.w);
        DG_TRAP_ONLY_DEVICE_ASSERT(ready);
    }

    const auto m_aligned = math::ceil_div(m_size, m_alignment) * m_alignment;
    if (blockIdx.x >= m_aligned)
        return;

    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = threadIdx.x / 32;

    __shared__ uint32_t smem_fwd_row;
    uint32_t slot;
    bool owns_slot = false;

    // Map the bwd_row in backward atomic order to forward atomic order
    // NOTES: only the real rows get mapped, padding rows use the same bwd_row as fwd_row.
    //        This is valid because paddings are always on the end of each expert
    if (warp_idx == 0) {
        if (blockIdx.x < m_size) {
            const auto bwd_row = m_start + blockIdx.x;
            const auto token_idx = atomic_to_zip[bwd_row];
            DG_TRAP_ONLY_DEVICE_ASSERT(token_idx >= 0);

            slot = static_cast<uint64_t>(token_idx) * kNumTopk + lane_idx;
            if (lane_idx < kNumTopk)
                owns_slot = static_cast<int>(recv_token_indices[slot]) == expert_idx;

            const auto bits = __ballot_sync(0xffffffff, owns_slot);
            DG_TRAP_ONLY_DEVICE_ASSERT(__popc(bits) == 1);

            if (owns_slot) {
                const auto fwd_row = zip_to_atomic[slot];
                DG_TRAP_ONLY_DEVICE_ASSERT(fwd_row >= 0);
                smem_fwd_row = static_cast<uint32_t>(fwd_row);
            }
        }
    }
    __syncthreads();

    // Fill padding rows with zero
    if (blockIdx.x >= m_size) {
        const auto row = m_start + blockIdx.x;
        const vec_t zeros = {};
        for (uint32_t col = threadIdx.x; col < kNumVecsPerRow; col += kNumThreads) {
            const auto off = static_cast<uint64_t>(row) * kNumVecsPerRow + col;
            *(reinterpret_cast<vec_t*>(o2_bwd) + off) = zeros;
            const auto off2 = static_cast<uint64_t>(row) * kNumVecsPerRow * 2 + col;
            *(reinterpret_cast<vec_t*>(do1) + off2) = zeros;
            *(reinterpret_cast<vec_t*>(do1) + off2 + kNumVecsPerRow) = zeros;
        }
        return;
    }

    const auto bwd_row = m_start + blockIdx.x;
    const auto fwd_row = blockIdx.x < m_size ? smem_fwd_row : bwd_row;
    const auto prob = probs[fwd_row];
    float dprobs_sum = 0.0;

    for (uint32_t col = threadIdx.x; col < kNumVecsPerRow; col += kNumThreads) {
        const auto off = static_cast<uint64_t>(bwd_row) * kNumVecsPerRow + col;
        const auto do2_vec = __ldg(reinterpret_cast<const vec_t*>(do2) + off);
        vec_t lo_vec, hi_vec, o2_vec, d_lo_vec, d_hi_vec;

        if constexpr (kInterleaved) {
            const auto off = static_cast<uint64_t>(fwd_row) * kNumVecsPerRow + col;
            lo_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off * 2);
            hi_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off * 2 + 1);
        } else {
            const auto off = static_cast<uint64_t>(fwd_row) * kNumVecsPerRow * 2 + col;
            lo_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off);
            hi_vec = __ldg(reinterpret_cast<const vec_t*>(o1) + off + kNumVecsPerRow);
        }

        const auto* grad = reinterpret_cast<const cutlass::bfloat16_t*>(&do2_vec);
        const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec);
        const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec);
        auto* o2_out = reinterpret_cast<cutlass::bfloat16_t*>(&o2_vec);
        auto* d_lo = reinterpret_cast<cutlass::bfloat16_t*>(&d_lo_vec);
        auto* d_hi = reinterpret_cast<cutlass::bfloat16_t*>(&d_hi_vec);

        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
            const auto k = (i % kNumPairsPerVec) * 2;
            const auto d = static_cast<float>(grad[i]);
            float g, u;
            if constexpr (kInterleaved) {
                const auto* pair = i < kNumPairsPerVec ? lo : hi;
                g = static_cast<float>(pair[k]);
                u = static_cast<float>(pair[k + 1]);
            } else {
                g = static_cast<float>(lo[i]);
                u = static_cast<float>(hi[i]);
            }

            const auto s = kPrecise ? 1.0f / (1.0f + expf(-g))
                                    : __fdividef(1.0f, 1.0f + __expf(-g));
            const auto silu = g * s;
            const auto w = silu * u;

            const auto dw = d * prob;
            const auto du = dw * silu;
            const auto dg = dw * u * s * (1.0f + g * (1.0f - s));

            dprobs_sum += d * w;

            o2_out[i] = static_cast<cutlass::bfloat16_t>(w * prob);

            if constexpr (kInterleaved) {
                auto* pair = i < kNumPairsPerVec ? d_lo : d_hi;
                pair[k] = static_cast<cutlass::bfloat16_t>(dg);
                pair[k + 1] = static_cast<cutlass::bfloat16_t>(du);
            } else {
                d_lo[i] = static_cast<cutlass::bfloat16_t>(dg);
                d_hi[i] = static_cast<cutlass::bfloat16_t>(du);
            }
        }

        *(reinterpret_cast<vec_t*>(o2_bwd) + off) = o2_vec;

        if constexpr (kInterleaved) {
            *(reinterpret_cast<vec_t*>(do1) + off * 2) = d_lo_vec;
            *(reinterpret_cast<vec_t*>(do1) + off * 2 + 1) = d_hi_vec;
        } else {
            const auto off = static_cast<uint64_t>(bwd_row) * kNumVecsPerRow * 2 + col;
            *(reinterpret_cast<vec_t*>(do1) + off) = d_lo_vec;
            *(reinterpret_cast<vec_t*>(do1) + off + kNumVecsPerRow) = d_hi_vec;
        }
    }

    __shared__ float smem_dprobs_sum[kNumThreads];
    smem_dprobs_sum[threadIdx.x] = dprobs_sum;
    __syncthreads();

    // Reduce dprobs in warp 0. Strictly fold the back half to the front half
    if (warp_idx == 0) {
        float reg_dprobs_sum[kNumThreads / 32];
        #pragma unroll
        for (uint32_t i = 0; i < kNumThreads / 32; ++ i)
            reg_dprobs_sum[i] = smem_dprobs_sum[i * 32 + lane_idx];

        #pragma unroll
        for (uint32_t stride = kNumThreads / 64; stride > 0; stride /= 2) {
            #pragma unroll
            for (uint32_t i = 0; i < stride; ++ i)
                reg_dprobs_sum[i] += reg_dprobs_sum[i + stride];
        }

        auto dprobs = reg_dprobs_sum[0];
        #pragma unroll
        for (uint32_t stride = 16; stride > 0; stride /= 2)
            dprobs += __shfl_down_sync(0xffffffff, dprobs, stride);
        dprobs = __shfl_sync(0xffffffff, dprobs, 0);

        if (owns_slot)
            drecv_probs[slot] = dprobs;
    }
}

}  // namespace deep_gemm
