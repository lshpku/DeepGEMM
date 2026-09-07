#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Weighted SwiGLU over one chunk of one expert, claimed from the chunk task queue
// NOTES: the gate/up columns are fully interleaved, i.e. `o2[:, j] = silu(o1[:, 2j]) * o1[:, 2j+1] * prob`
//        the chunk's padded tail is zeroed, or the down GEMM would consume garbage rows
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumElemsPerAccess,
          uint32_t kNumVecsPerRow, uint32_t kNumVecsPerThread, bool kPrecise, bool kInterleaved>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_weighted_swiglu_impl(const int* task_queue, uint32_t task_idx,
                                const cutlass::bfloat16_t* o1, const float* probs,
                                cutlass::bfloat16_t* o2,
                                uint32_t m_alignment) {
    // Wait for the producer (the gate-up GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    // Claim the task: `[expert_idx, m_start, m_size, ready]`
    // NOTES: only one thread touches the queue, which may be slow mapped host memory
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
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    constexpr uint32_t kGridStride = kNumSMs * kNumThreads;
    const auto m_padded = math::ceil_div(m_size, m_alignment) * m_alignment;
    const auto num_vecs = m_padded * kNumVecsPerRow;

    // Persistent: use a fixed number of CTAs strides over the flattened o2 space and calculates
    // the row and col in each step
    for (uint32_t base = blockIdx.x * kNumThreads + threadIdx.x; base < num_vecs;
         base += kGridStride * kNumVecsPerThread) {
        vec_t lo_vec[kNumVecsPerThread], hi_vec[kNumVecsPerThread];
        float prob[kNumVecsPerThread];
        uint32_t row[kNumVecsPerThread], col[kNumVecsPerThread];
        bool in_range[kNumVecsPerThread], is_real[kNumVecsPerThread];

        // Issue kNumVecsPerThread loads parallely to increase bandwidth
        #pragma unroll
        for (uint32_t j = 0; j < kNumVecsPerThread; ++ j) {
            const auto idx = base + j * kGridStride;
            const auto row_in_chunk = idx / kNumVecsPerRow;
            in_range[j] = idx < num_vecs;
            is_real[j] = in_range[j] and row_in_chunk < m_size;
            row[j] = m_start + row_in_chunk;
            col[j] = (idx % kNumVecsPerRow) * kNumElemsPerAccess;

            if (is_real[j]) {
                if constexpr (kInterleaved) {
                    const auto* ptr = o1 + static_cast<uint64_t>(row[j]) * kShapeN * 2 + col[j] * 2;
                    lo_vec[j] = __ldg(reinterpret_cast<const vec_t*>(ptr));
                    hi_vec[j] = __ldg(reinterpret_cast<const vec_t*>(ptr + kNumElemsPerAccess));
                } else {
                    const auto* ptr = o1 + static_cast<uint64_t>(row[j]) * kShapeN * 2 + col[j];
                    lo_vec[j] = __ldg(reinterpret_cast<const vec_t*>(ptr));
                    hi_vec[j] = __ldg(reinterpret_cast<const vec_t*>(ptr + kShapeN));
                }
                prob[j] = probs[row[j]];
            }
        }

        // Compute in FP32, and zero the chunk's padded tail
        #pragma unroll
        for (uint32_t j = 0; j < kNumVecsPerThread; ++ j) {
            if (not in_range[j])
                continue;

            cutlass::bfloat16_t out[kNumElemsPerAccess] = {};
            if (is_real[j]) {
                const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec[j]);
                const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec[j]);
                #pragma unroll
                for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
                    float g, u;
                    if constexpr (kInterleaved) {
                        const auto* pair = i < kNumElemsPerAccess / 2 ? lo : hi;
                        const auto k = (i % (kNumElemsPerAccess / 2)) * 2;
                        g = static_cast<float>(pair[k]);
                        u = static_cast<float>(pair[k + 1]);
                    } else {
                        g = static_cast<float>(lo[i]);
                        u = static_cast<float>(hi[i]);
                    }
                    const auto silu = kPrecise ? g * (1.0f / (1.0f + expf(-g)))
                                               : __fdividef(g, 1.0f + __expf(-g));
                    out[i] = static_cast<cutlass::bfloat16_t>(silu * u * prob[j]);
                }
            }
            auto* out_ptr = o2 + static_cast<uint64_t>(row[j]) * kShapeN + col[j];
            *reinterpret_cast<vec_t*>(out_ptr) = *reinterpret_cast<const vec_t*>(out);
        }
    }
}

}  // namespace deep_gemm
