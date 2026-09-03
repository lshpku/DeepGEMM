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
// NOTES: one row is owned by a group of `kNumWarpsPerRow` warps, which is what makes the
//        `probs` gradient a warp reduction plus one tiny shared-memory reduction, with no
//        atomics; the whole CTA runs a uniform number of iterations so that the
//        `__syncthreads()` around that reduction stays safe
// NOTES: the chunk's padded tail is zeroed, as `do1`/`o2_bwd` feed the weight gradients, where
//        a `0 * garbage` product of the other operand would still poison the accumulator
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kNumWarpsPerRow, uint32_t kNumTopk,
          uint32_t kNumVecsPerRow, uint32_t kNumElemsPerAccess, bool kPrecise>
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
    constexpr uint32_t kNumRowGroups = kNumWarps / kNumWarpsPerRow;
    constexpr uint32_t kNumVecsPerStep = 32 * kNumWarpsPerRow;
    constexpr uint32_t kNumSteps = kNumVecsPerRow / kNumVecsPerStep;
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kShapeN = kNumVecsPerRow * kNumElemsPerAccess;
    DG_STATIC_ASSERT(kNumWarps % kNumWarpsPerRow == 0, "Warps must split into whole row groups");
    DG_STATIC_ASSERT(kNumVecsPerRow % kNumVecsPerStep == 0, "A row must split into whole steps");

    // Wait for the producer (the down-grad GEMM) when PDL is enabled
    cudaGridDependencySynchronize();

    // Claim the task: `[expert_idx, m_start, m_size, ready]`
    __shared__ int smem_task[4];
    __shared__ float smem_partial[kNumRowGroups][kNumWarpsPerRow];
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
    const auto group_idx = warp_idx / kNumWarpsPerRow;
    const auto warp_in_group = warp_idx % kNumWarpsPerRow;

    // The rows are split evenly over the CTAs, and every CTA runs the same number of iterations
    // so that a short remainder chunk still uses every SM and the barriers stay uniform
    const auto m_padded = math::ceil_div(m_size, m_alignment) * m_alignment;
    const auto rows_per_cta = math::ceil_div(m_padded, kNumSMs);
    const auto row_begin = blockIdx.x * rows_per_cta;
    const auto row_end = min(row_begin + rows_per_cta, m_padded);
    const auto num_iters = math::ceil_div(rows_per_cta, kNumRowGroups);

    for (uint32_t it = 0; it < num_iters; ++ it) {
        const auto row_raw = row_begin + it * kNumRowGroups + group_idx;
        const auto in_range = row_raw < row_end;
        const auto is_real = in_range and row_raw < m_size;

        // The row this iteration writes, and the row it reads
        // NOTES: a group whose row is padding, or past this CTA's range, still reads row 0 of the
        //        chunk, which is always a real one; that keeps the mapping below well defined for
        //        every group, and the compiler hoists its loads out of any predicate anyway
        const auto store_row = m_start + row_raw;
        const auto read_row = m_start + (is_real ? row_raw : 0);

        // Map the row onto the forward one, through the token and this task's expert
        // NOTES: unconditional on purpose: a `-1` token or an undefined ballot would form a wild
        //        address, and predicating it buys nothing as the loads get hoisted regardless
        // NOTES: a lane that owns no slot still reads an in-bounds one, for the same reason
        const auto token_idx = atomic_to_zip[read_row];
        DG_TRAP_ONLY_DEVICE_ASSERT(token_idx >= 0);
        const auto slot = static_cast<uint64_t>(token_idx) * kNumTopk + min(lane_idx, kNumTopk - 1);

        // The experts of one token are distinct, hence exactly one slot matches
        const auto owns_slot = lane_idx < kNumTopk and
                               static_cast<int>(recv_token_indices[slot]) == expert_idx;
        const auto bits = __ballot_sync(0xffffffff, owns_slot);
        DG_TRAP_ONLY_DEVICE_ASSERT(__popc(bits) == 1);
        const auto fwd_row = static_cast<uint32_t>(__shfl_sync(0xffffffff, zip_to_atomic[slot],
                                                               __ffs(bits) - 1));
        const auto prob = probs[fwd_row];

        // The row itself: a lane owns one output vector per step, i.e. one gate/up pair per channel
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t step = 0; step < kNumSteps; ++ step) {
            const auto col = (step * kNumVecsPerStep + warp_in_group * 32 + lane_idx) * kNumElemsPerAccess;

            vec_t do2_vec, lo_vec, hi_vec;
            if (is_real) {
                const auto* gate_up_ptr = o1 + static_cast<uint64_t>(fwd_row) * kShapeN * 2 + col * 2;
                lo_vec = __ldg(reinterpret_cast<const vec_t*>(gate_up_ptr));
                hi_vec = __ldg(reinterpret_cast<const vec_t*>(gate_up_ptr + kNumElemsPerAccess));
                do2_vec = __ldg(reinterpret_cast<const vec_t*>(
                    do2 + static_cast<uint64_t>(read_row) * kShapeN + col));
            }

            // Compute in FP32, and leave the padded tail at zero
            // NOTES: the staging arrays are read back as vectors, so they must carry the vector's
            //        alignment: once the register pressure spills them to local memory, a
            //        naturally aligned `bfloat16_t[]` would make those reads misaligned
            __align__(sizeof(vec_t)) cutlass::bfloat16_t o2_out[kNumElemsPerAccess] = {};
            __align__(sizeof(vec_t)) cutlass::bfloat16_t do1_out[kNumElemsPerAccess * 2] = {};
            if (is_real) {
                const auto* lo = reinterpret_cast<const cutlass::bfloat16_t*>(&lo_vec);
                const auto* hi = reinterpret_cast<const cutlass::bfloat16_t*>(&hi_vec);
                const auto* grad = reinterpret_cast<const cutlass::bfloat16_t*>(&do2_vec);

                #pragma unroll
                for (uint32_t i = 0; i < kNumElemsPerAccess; ++ i) {
                    const auto* pair = i < kNumPairsPerVec ? lo : hi;
                    const auto k = (i % kNumPairsPerVec) * 2;
                    const auto g = static_cast<float>(pair[k]);
                    const auto u = static_cast<float>(pair[k + 1]);
                    const auto d = static_cast<float>(grad[i]);

                    // NOTES: `silu` is `g * sigmoid(g)` in exactly the forward's expression, so
                    //        `o2_bwd` reproduces the forward `o2` bit for bit in the precise mode
                    const auto s = kPrecise ? 1.0f / (1.0f + expf(-g))
                                            : __fdividef(1.0f, 1.0f + __expf(-g));
                    const auto silu = g * s;
                    const auto w = silu * u;

                    o2_out[i] = static_cast<cutlass::bfloat16_t>(w * prob);
                    acc += d * w;

                    // `d(silu)/dg = s * (1 + g * (1 - s))`
                    const auto dw = d * prob;
                    do1_out[k + (i < kNumPairsPerVec ? 0 : kNumElemsPerAccess)] =
                        static_cast<cutlass::bfloat16_t>(dw * u * s * (1.0f + g * (1.0f - s)));
                    do1_out[k + 1 + (i < kNumPairsPerVec ? 0 : kNumElemsPerAccess)] =
                        static_cast<cutlass::bfloat16_t>(dw * silu);
                }
            }

            // NOTES: the rows past this CTA's range belong to the next one, so they must not be
            //        written at all, not even zeroed, or the two CTAs would race over them
            if (not in_range)
                continue;

            *reinterpret_cast<vec_t*>(o2_bwd + static_cast<uint64_t>(store_row) * kShapeN + col) =
                *reinterpret_cast<const vec_t*>(o2_out);
            auto* do1_ptr = do1 + static_cast<uint64_t>(store_row) * kShapeN * 2 + col * 2;
            *reinterpret_cast<vec_t*>(do1_ptr) = *reinterpret_cast<const vec_t*>(do1_out);
            *reinterpret_cast<vec_t*>(do1_ptr + kNumElemsPerAccess) =
                *reinterpret_cast<const vec_t*>(do1_out + kNumElemsPerAccess);
        }

        // The `probs` gradient is the row's reduction: within the warp, then over the group's
        // warps, and the lane that owns the token's slot publishes it
        // NOTES: a padded row owns no slot, so it writes nothing at all
        #pragma unroll
        for (uint32_t offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        if (lane_idx == 0)
            smem_partial[group_idx][warp_in_group] = acc;
        __syncthreads();

        if (warp_in_group == 0 and is_real and owns_slot) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < kNumWarpsPerRow; ++ w)
                sum += smem_partial[group_idx][w];
            drecv_probs[slot] = sum;
        }
        __syncthreads();
    }
}

}  // namespace deep_gemm
