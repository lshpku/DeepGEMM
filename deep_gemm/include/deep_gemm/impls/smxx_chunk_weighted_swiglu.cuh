#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Weighted SwiGLU over one chunk of one expert, claimed from the chunk task queue
// NOTES: the gate/up columns are fully interleaved, i.e. `o2[:, j] = silu(o1[:, 2j]) * o1[:, 2j+1] * prob`
//        the chunk's padded tail is zeroed, or the down GEMM would consume garbage rows
template <uint32_t kNumThreads, uint32_t kNumElemsPerAccess, uint32_t kNumVecsPerRow,
          uint32_t kNumVecsPerThread, bool kPrecise, bool kInterleaved, bool kQuant>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_chunk_weighted_swiglu_impl(const int* task_queue, uint32_t task_idx,
                                const cutlass::bfloat16_t* o1, const float* probs,
                                cutlass::bfloat16_t* o2, uint8_t* o2_scales, uint32_t scale_stride,
                                uint32_t m_alignment) {
    // Wait for the producer (the gate-up GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    // Claim the task: `[expert_idx, m_start, m_size, ready]` without fence
    const auto task = __ldg(reinterpret_cast<const int4*>(task_queue) + task_idx);
    const auto m_start = static_cast<uint32_t>(task.y);
    const auto m_size = static_cast<uint32_t>(task.z);
    if (threadIdx.x == 0) {
        const auto expert_idx = task.x;
        const auto ready = task.w;
        if (!ready) {
            printf("swiglu task not ready: expert_idx=%u, m_start=%u, m_size=%u, ready=%d, "
                   "block=%u\n", expert_idx, m_start, m_size, ready, blockIdx.x);
            DG_TRAP_ONLY_DEVICE_ASSERT(0);
        }
    }

    // Vectorized element-wise traversal over `o2`'s `[align(m_size, m_alignment), kShapeN]`
    // NOTES: one output vector consumes two adjacent input vectors, as a gate/up pair is
    //        adjacent in the fully interleaved layout; a lane's two loads are 32B apart so each
    //        one alone uses half of every sector it fetches, but together they cover exactly
    //        the contiguous range the warp needs and the second one hits L1
    // NOTES: the row length is a template parameter, so the row/column split of a flat index
    //        is a shift instead of a division for the usual power-of-two `N`
    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t),
                     "Invalid vector size");
    // One quantization block is `kGranK` columns, so it must be an exact number of vectors,
    // and a warp must never straddle two rows, or the lanes of a block would not be a group
    DG_STATIC_ASSERT(!kQuant || quant::kGranK % kNumElemsPerAccess == 0, "Invalid granularity");
    DG_STATIC_ASSERT(!kQuant || kNumVecsPerRow % 32 == 0, "Invalid row length");
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    const auto grid_stride = gridDim.x * kNumThreads;
    const auto m_padded = math::ceil_div(m_size, m_alignment) * m_alignment;
    const auto num_vecs = m_padded * kNumVecsPerRow;

    // Strides over the flattened o2 space and calculates the row and col in each step
    const uint32_t base = blockIdx.x * kNumThreads + threadIdx.x;
    if (base >= num_vecs)
        return;

    vec_t lo_vec[kNumVecsPerThread], hi_vec[kNumVecsPerThread];
    float prob[kNumVecsPerThread];
    uint32_t row[kNumVecsPerThread], col[kNumVecsPerThread];
    bool in_range[kNumVecsPerThread], is_real[kNumVecsPerThread];

    // Issue kNumVecsPerThread loads parallely to increase bandwidth
    #pragma unroll
    for (uint32_t i = 0; i < kNumVecsPerThread; ++ i) {
        const auto idx = base + i * grid_stride;
        const auto row_in_chunk = idx / kNumVecsPerRow;
        in_range[i] = idx < num_vecs;
        is_real[i] = in_range[i] && row_in_chunk < m_size;
        row[i] = m_start + row_in_chunk;
        col[i] = (idx % kNumVecsPerRow) * kNumElemsPerAccess;

        if (is_real[i]) {
            if constexpr (kInterleaved) {
                const auto* ptr = o1 + static_cast<uint64_t>(row[i]) * kShapeN * 2 + col[i] * 2;
                lo_vec[i] = __ldg(reinterpret_cast<const vec_t*>(ptr));
                hi_vec[i] = __ldg(reinterpret_cast<const vec_t*>(ptr + kNumElemsPerAccess));
            } else {
                const auto* ptr = o1 + static_cast<uint64_t>(row[i]) * kShapeN * 2 + col[i];
                lo_vec[i] = __ldg(reinterpret_cast<const vec_t*>(ptr));
                hi_vec[i] = __ldg(reinterpret_cast<const vec_t*>(ptr + kShapeN));
            }
            prob[i] = probs[row[i]];
        }
    }

    // Compute in FP32, and zero the chunk's padded tail
    #pragma unroll
    for (uint32_t i = 0; i < kNumVecsPerThread; ++ i) {
        if (!in_range[i])
            continue;

        // A padded row keeps its zeros, which quantize to zeroed data and the `1.0f` exponent
        float act[kNumElemsPerAccess] = {};

        if (is_real[i]) {
            const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec[i]);
            const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec[i]);
            #pragma unroll
            for (uint32_t j = 0; j < kNumElemsPerAccess; ++ j) {
                float g, u;
                if constexpr (kInterleaved) {
                    const auto* pair = j < kNumElemsPerAccess / 2 ? lo : hi;
                    const auto k = (j % (kNumElemsPerAccess / 2)) * 2;
                    g = static_cast<float>(pair[k]);
                    u = static_cast<float>(pair[k + 1]);
                } else {
                    g = static_cast<float>(lo[j]);
                    u = static_cast<float>(hi[j]);
                }
                const auto silu = kPrecise ? g * (1.0f / (1.0f + expf(-g)))
                                           : g * __frcp_rn(1.0f + __expf(-g));
                act[j] = silu * u * prob[i];
            }
        }

        if constexpr (kQuant) {
            // One scale per `kGranK` columns, all-reduced over the lanes that share the block
            const auto exponent = quant::store_quantized(
                o2, act, row[i], col[i] / kNumElemsPerAccess, true, kNumVecsPerRow);
            if (ptx::get_lane_idx() % quant::kNumLanesPerBlock == 0)
                quant::store_scale(o2_scales, scale_stride, row[i], col[i] / quant::kGranK,
                                   exponent);
        } else {
            cutlass::bfloat16_t out[kNumElemsPerAccess];
            #pragma unroll
            for (uint32_t j = 0; j < kNumElemsPerAccess; ++ j)
                out[j] = static_cast<cutlass::bfloat16_t>(act[j]);
            auto* out_ptr = o2 + static_cast<uint64_t>(row[i]) * kShapeN + col[i];
            *reinterpret_cast<vec_t*>(out_ptr) = *reinterpret_cast<const vec_t*>(out);
        }
    }
}

}  // namespace deep_gemm
