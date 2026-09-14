#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/fp8_quant.cuh>
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
          uint32_t kNumElemsPerAccess, bool kPrecise, bool kInterleaved, bool kQuant>
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
                                     uint8_t* __restrict__ o2_bwd_scales,
                                     uint8_t* __restrict__ do1_scales,
                                     uint32_t scale_stride,
                                     uint32_t m_alignment) {
    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Invalid vector size");
    DG_STATIC_ASSERT(kNumElemsPerAccess % 2 == 0, "A gate/up pair must fit into a vector");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit into a warp");
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    constexpr uint32_t kNumElemsPerPack = quant::kGranK * quant::kNumGranKPerPack;
    DG_STATIC_ASSERT(!kQuant || quant::kGranK % kNumElemsPerAccess == 0, "Invalid granularity");
    DG_STATIC_ASSERT(!kQuant || kShapeN % kNumElemsPerPack == 0, "Invalid shape N");
    constexpr uint32_t kNumLanesPerBlock = quant::kGranK / kNumElemsPerAccess;
    constexpr uint32_t kNumPacksPerRow = kShapeN / kNumElemsPerPack;
    constexpr uint32_t kNumScalesPerRow = kShapeN / quant::kGranK;

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
        constexpr auto kOutVecs = kQuant ? kNumVecsPerRow / 2 : kNumVecsPerRow;
        for (uint32_t col = threadIdx.x; col < kOutVecs; col += kNumThreads) {
            const auto off = static_cast<uint64_t>(row) * kOutVecs + col;
            *(reinterpret_cast<vec_t*>(o2_bwd) + off) = {};
            const auto off2 = static_cast<uint64_t>(row) * kOutVecs * 2 + col;
            *(reinterpret_cast<vec_t*>(do1) + off2) = {};
            *(reinterpret_cast<vec_t*>(do1) + off2 + kOutVecs) = {};
        }
        if constexpr (kQuant) {
            for (uint32_t p = threadIdx.x; p < kNumPacksPerRow; p += kNumThreads) {
                quant::store_zero_packs(o2_bwd_scales, scale_stride, row, kNumPacksPerRow, p);
                quant::store_zero_packs(do1_scales, scale_stride, row, kNumPacksPerRow * 2, p);
                quant::store_zero_packs(do1_scales, scale_stride, row, kNumPacksPerRow * 2,
                                        p + kNumPacksPerRow);
            }
        }
        return;
    }

    const auto bwd_row = m_start + blockIdx.x;
    const auto fwd_row = blockIdx.x < m_size ? smem_fwd_row : bwd_row;
    const auto prob = probs[fwd_row];
    float dprobs_sum = 0.0;

    // The exponents are staged and flushed as whole int32 packs to improve cache granularity
    __shared__ __align__(4) uint8_t smem_exp_o2[kQuant ? kNumScalesPerRow : 1];
    __shared__ __align__(4) uint8_t smem_exp_do1[kQuant ? kNumScalesPerRow * 2 : 1];

    for (uint32_t col = threadIdx.x; col < kNumVecsPerRow; col += kNumThreads) {
        const auto off = static_cast<uint64_t>(bwd_row) * kNumVecsPerRow + col;
        const auto do2_vec = __ldg(reinterpret_cast<const vec_t*>(do2) + off);
        vec_t lo_vec, hi_vec, o2_vec, d_lo_vec, d_hi_vec;
        float o2_act[kNumElemsPerAccess], dg_act[kNumElemsPerAccess], du_act[kNumElemsPerAccess];

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
                                    : __frcp_rn(1.0f + __expf(-g));
            const auto silu = g * s;
            const auto w = silu * u;

            const auto dw = d * prob;
            const auto du = dw * silu;
            const auto dg = kPrecise ? dw * u * s * (1.0f + g * (1.0f - s))
                                     : dw * (u * (s * ((g + 1.0f) - silu)));

            dprobs_sum += d * w;

            if constexpr (kQuant) {
                // NOTES: rounds through BF16 here because the paddle reference path downcasts
                //        do1/o2 before quantization
                dg_act[i] = static_cast<float>(static_cast<cutlass::bfloat16_t>(dg));
                du_act[i] = static_cast<float>(static_cast<cutlass::bfloat16_t>(dw * silu));
                o2_act[i] = static_cast<float>(static_cast<cutlass::bfloat16_t>(w * prob));
            } else {
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
        }

        if constexpr (kQuant) {
            const auto group = col / kNumLanesPerBlock;
            const auto e_o2 = quant::store_quantized(o2_bwd, o2_act, bwd_row, col,
                                                     true, kNumVecsPerRow);
            const auto e_dg = quant::store_quantized(do1, dg_act, bwd_row, col,
                                                     true, kNumVecsPerRow * 2);
            const auto e_du = quant::store_quantized(do1, du_act, bwd_row, col + kNumVecsPerRow,
                                                     true, kNumVecsPerRow * 2);
            if (lane_idx % kNumLanesPerBlock == 0) {
                smem_exp_o2[group] = e_o2;
                smem_exp_do1[group] = e_dg;
                smem_exp_do1[group + kNumScalesPerRow] = e_du;
            }
        } else {
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
    }

    __shared__ float smem_dprobs_sum[kNumThreads];
    smem_dprobs_sum[threadIdx.x] = dprobs_sum;
    __syncthreads();

    if constexpr (kQuant) {
        quant::store_scale_packs(o2_bwd_scales, scale_stride, bwd_row, smem_exp_o2,
                                 kNumPacksPerRow, threadIdx.x);
        quant::store_scale_packs(do1_scales, scale_stride, bwd_row, smem_exp_do1,
                                 kNumPacksPerRow * 2, threadIdx.x);
    }

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
