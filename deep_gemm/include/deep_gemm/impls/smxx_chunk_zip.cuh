#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Publish the token completion of one chunk and zip the tokens it finishes, in one kernel
//
// NOTES: a token is ready for `combine` once all of its local experts have computed it, so
//        step one bumps a counter per row of the chunk and keeps the finished tokens in a
//        CTA-local queue in shared memory, then step two sums each of them over its experts
// NOTES: no `acquire` is needed on `o3`: all the other experts of a finished token belong to
//        earlier chunks, hence earlier kernels of this same (serialized) compute stream, and
//        this chunk's own rows are ordered by the CTA's `acquire` on the task's `ready` flag
// NOTES: the experts are accumulated in ascending order of the expert index, which is the
//        only order that does not depend on the (non-deterministic) arrival order
template <uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kMaxRowsPerCTA,
          uint32_t kNumTopk, uint32_t kNumVecsPerRow, uint32_t kNumElemsPerAccess>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_zip_impl(const int* task_queue, uint32_t task_idx,
                    const cutlass::bfloat16_t* o3, cutlass::bfloat16_t* combine_input,
                    const int* atomic_to_zip, const int* zip_to_atomic,
                    const int64_t* recv_token_indices, const int* num_valid_topk,
                    int* token_done, int* zip_done,
                    uint32_t num_tokens) {
    using vec_t = int4;
    DG_STATIC_ASSERT(kNumElemsPerAccess * sizeof(cutlass::bfloat16_t) == sizeof(vec_t), "Invalid vector size");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread number");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit into a warp");
    constexpr uint32_t kNumWarps = kNumThreads / 32;

    // How many vectors each lane accumulates per step, and how many `bfloat162` a vector holds
    // NOTES: four independent 128-bit loads in flight per lane, so the expert loop is no longer
    //        serialized on a single miss, and each of them is still a coalesced 32-lane transaction
    constexpr uint32_t kNumVecsPerLane = 8;
    constexpr uint32_t kNumPairsPerVec = kNumElemsPerAccess / 2;
    constexpr uint32_t kNumVecsPerStep = 32 * kNumVecsPerLane;
    constexpr uint32_t kNumFullVecs = kNumVecsPerRow / kNumVecsPerStep * kNumVecsPerStep;

    // Wait for the producer (the down GEMM), whose stores must be visible before the counters
    cudaGridDependencySynchronize();

    // Claim the task, in the same way as the compute kernels
    __shared__ int smem_task[4];
    __shared__ int smem_num_done;
    __shared__ int smem_done_queue[kMaxRowsPerCTA];
    __shared__ int smem_sorted_rows[kNumWarps][kNumTopk];
    if (threadIdx.x == 0) {
        const auto timed_out = chunk::wait_task_ready(task_queue, task_idx);
        chunk::stage_task(smem_task, chunk::read_task(task_queue, task_idx));
        smem_task[3] = timed_out ? 1 : 0;
        smem_num_done = 0;
    }
    __syncthreads();
    DG_TRAP_ONLY_DEVICE_ASSERT(smem_task[3] == 0);
    const auto task = chunk::load_staged_task(smem_task);
    const auto m_start = static_cast<uint32_t>(task.m_start);
    const auto m_size = static_cast<uint32_t>(task.m_size);
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();

    // Step 1: count this chunk's rows and collect the tokens it finishes
    // NOTES: the rows are split evenly over all the CTAs, so a short remainder chunk still
    //        uses every SM; only the queue's capacity is a power of two, sized for a full chunk
    const auto rows_per_cta = math::ceil_div(m_size, kNumSMs);
    const auto row_begin = blockIdx.x * rows_per_cta;
    const auto row_end = min(row_begin + rows_per_cta, m_size);
    DG_TRAP_ONLY_DEVICE_ASSERT(rows_per_cta <= kMaxRowsPerCTA);
    for (uint32_t i = row_begin + threadIdx.x; i < row_end; i += kNumThreads) {
        const auto token_idx = atomic_to_zip[m_start + i];
        bool ready = false;

        // NOTES: padded rows carry `-1`, and they are never part of a chunk anyway
        if (token_idx >= 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(static_cast<uint32_t>(token_idx) < num_tokens);

            // NOTES: the counter itself needs no memory ordering, as all the chunks of one
            //        token are computed by earlier kernels of this same (serialized) stream
            const auto counted = ptx::atomic_add(token_done + token_idx, 1) + 1;
            const auto expected = num_valid_topk[token_idx];
            DG_TRAP_ONLY_DEVICE_ASSERT(counted <= expected);

            ready = counted == expected;
        }

        const uint32_t mask = __activemask();
        const uint32_t bits = __ballot_sync(mask, ready);
        const uint32_t lower_mask = (1u << lane_idx) - 1u;
        const uint32_t warp_prefix = __popc(bits & lower_mask);
        const uint32_t warp_count = __popc(bits);

        // Only lane 0 does the atomic add; other lanes share the base slot and
        // add their prefix in the warp to get their own slots
        if (warp_count > 0) {
            uint32_t slot;
            if (lane_idx == 0)
                slot = atomicAdd(&smem_num_done, warp_count);
            slot = __shfl_sync(mask, slot, 0);
            if (ready)
                smem_done_queue[slot + warp_prefix] = token_idx;
        }
    }
    __syncthreads();
    const auto num_done = static_cast<uint32_t>(smem_num_done);

    // Step 2: one warp zips one token of the local queue
    const auto* o3_vecs = reinterpret_cast<const vec_t*>(o3);
    auto* combine_input_vecs = reinterpret_cast<vec_t*>(combine_input);
    for (uint32_t q = warp_idx; q < num_done; q += kNumWarps) {
        const auto token_idx = smem_done_queue[q];

        // Rank the token's experts, so that the accumulation order is the ascending expert index
        // NOTES: only `zip_to_atomic` tells whether a slot is valid, `recv_token_indices` may hold
        //        garbage in the invalid slots, which sit at `INT_MAX` and are never counted
        int expert_idx = INT_MAX, o3_row = -1;
        if (lane_idx < kNumTopk) {
            const auto slot = static_cast<uint64_t>(token_idx) * kNumTopk + lane_idx;
            o3_row = zip_to_atomic[slot];
            if (o3_row >= 0)
                expert_idx = static_cast<int>(recv_token_indices[slot]);
        }
        const auto num_experts = static_cast<uint32_t>(__popc(__ballot_sync(0xffffffff, o3_row >= 0)));
        DG_TRAP_ONLY_DEVICE_ASSERT(num_experts == static_cast<uint32_t>(num_valid_topk[token_idx]));

        // NOTES: the experts of one token are distinct, so counting the smaller ones is a sort
        uint32_t rank = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumTopk; ++ j)
            rank += __shfl_sync(0xffffffff, expert_idx, j) < expert_idx ? 1 : 0;
        if (o3_row >= 0)
            smem_sorted_rows[warp_idx][rank] = o3_row;
        __syncwarp();

        // Accumulate in FP32, kept as `float2` so a whole `bfloat162` is converted at a time
        // NOTES: the step loop must stay free of any per-`k` predicate, or the unroll breaks down
        //        and the accumulators get indexed dynamically, i.e. spilled to local memory;
        //        a row length that is not a whole number of steps goes to the scalar tail instead
        auto* dst_row = combine_input_vecs + static_cast<uint64_t>(token_idx) * kNumVecsPerRow;
        for (uint32_t base = lane_idx; base < kNumFullVecs; base += kNumVecsPerStep) {
            float2 acc[kNumVecsPerLane][kNumPairsPerVec] = {};

            for (uint32_t r = 0; r < num_experts; ++ r) {
                const auto* src_row = o3_vecs + static_cast<uint64_t>(smem_sorted_rows[warp_idx][r]) * kNumVecsPerRow;
                #pragma unroll
                for (uint32_t k = 0; k < kNumVecsPerLane; ++ k) {
                    auto value = __ldg(src_row + base + k * 32);
                    const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&value);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumPairsPerVec; ++ j)
                        ptx::accumulate(acc[k][j], pairs[j]);
                }
            }

            #pragma unroll
            for (uint32_t k = 0; k < kNumVecsPerLane; ++ k) {
                vec_t out;
                auto* pairs = reinterpret_cast<nv_bfloat162*>(&out);
                #pragma unroll
                for (uint32_t j = 0; j < kNumPairsPerVec; ++ j)
                    pairs[j] = __float22bfloat162_rn(acc[k][j]);
                dst_row[base + k * 32] = out;
            }
        }

        // Scalar tail, one vector per lane
        if constexpr (kNumFullVecs < kNumVecsPerRow) {
            for (uint32_t v = kNumFullVecs + lane_idx; v < kNumVecsPerRow; v += 32) {
                float2 acc[kNumPairsPerVec] = {};

                for (uint32_t r = 0; r < num_experts; ++ r) {
                    const auto* src_row = o3_vecs + static_cast<uint64_t>(smem_sorted_rows[warp_idx][r]) * kNumVecsPerRow;
                    auto value = __ldg(src_row + v);
                    const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&value);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumPairsPerVec; ++ j)
                        ptx::accumulate(acc[j], pairs[j]);
                }

                vec_t out;
                auto* pairs = reinterpret_cast<nv_bfloat162*>(&out);
                #pragma unroll
                for (uint32_t j = 0; j < kNumPairsPerVec; ++ j)
                    pairs[j] = __float22bfloat162_rn(acc[j]);
                dst_row[v] = out;
            }
        }

        // The next token of this warp reuses the same shared memory slots
        __syncwarp();
    }

    // Step 3: publish the zipped tokens to `combine`
    // NOTES: the previous __syncwarp makes the whole warp's stores visible to lane 0,
    //        so one fence per warp is enough
    if (lane_idx == 0) {
        for (uint32_t q = warp_idx; q < num_done; q += kNumWarps)
            ptx::st_rel(zip_done + smem_done_queue[q], 1);
    }
}

}  // namespace deep_gemm
