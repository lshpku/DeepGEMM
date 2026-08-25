#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Weighted SwiGLU over one chunk of one expert, claimed from the chunk task queue
// NOTES: the gate/up halves are concatenated along N, i.e. `o2 = silu(o1[:, :N]) * o1[:, N:] * prob`
//        the chunk's padded tail is zeroed, or the down GEMM would consume garbage rows
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumElemsPerAccess>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_weighted_swiglu_impl(const int* task_queue, uint32_t task_idx,
                                const cutlass::bfloat16_t* o1, const float* probs,
                                cutlass::bfloat16_t* o2,
                                uint32_t shape_n, uint32_t m_alignment) {
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

    // Vectorized element-wise traversal over `[align(m_size, m_alignment), shape_n]`
    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Invalid vector size");
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;
    const auto m_padded = math::ceil_div(m_size, m_alignment) * m_alignment;
    const auto num_vecs = static_cast<uint64_t>(m_padded) * num_vecs_per_row;

    // Persistent: a fixed number of CTAs strides over all the work
    for (uint64_t idx = blockIdx.x * kNumThreads + threadIdx.x; idx < num_vecs; idx += kNumSMs * kNumThreads) {
        const auto row_in_chunk = static_cast<uint32_t>(idx / num_vecs_per_row);
        const auto row = m_start + row_in_chunk;
        const auto col = static_cast<uint32_t>(idx % num_vecs_per_row) * kNumElemsPerAccess;
        auto* out_ptr = o2 + static_cast<uint64_t>(row) * shape_n + col;

        // Zero the padded tail
        if (row_in_chunk >= m_size) {
            cutlass::bfloat16_t zeros[kNumElemsPerAccess] = {};
            *reinterpret_cast<vec_t*>(out_ptr) = *reinterpret_cast<const vec_t*>(zeros);
            continue;
        }

        const auto* gate_ptr = o1 + static_cast<uint64_t>(row) * shape_n * 2 + col;
        const auto* up_ptr = gate_ptr + shape_n;

        // Load 2 vectors and the router score
        const auto gate_vec = __ldg(reinterpret_cast<const vec_t*>(gate_ptr));
        const auto up_vec = __ldg(reinterpret_cast<const vec_t*>(up_ptr));
        const auto prob = probs[row];

        // Compute in FP32
        const auto* gate = reinterpret_cast<const cutlass::bfloat16_t*>(&gate_vec);
        const auto* up = reinterpret_cast<const cutlass::bfloat16_t*>(&up_vec);
        cutlass::bfloat16_t out[kNumElemsPerAccess];
        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
            const auto g = static_cast<float>(gate[i]);
            const auto u = static_cast<float>(up[i]);
            out[i] = static_cast<cutlass::bfloat16_t>(g / (1.0f + __expf(-g)) * u * prob);
        }
        *reinterpret_cast<vec_t*>(out_ptr) = *reinterpret_cast<const vec_t*>(out);
    }
}

}  // namespace deep_gemm
