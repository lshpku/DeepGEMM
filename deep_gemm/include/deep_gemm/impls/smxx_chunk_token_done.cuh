#pragma once

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Publish the token completion of one chunk, i.e. what `zip` waits on
// NOTES: a token is summed by `zip` once all of its local experts have counted it, so this
//        bumps a counter per row and pushes the token into `zip_task_queue` when it is full
template <uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_token_done_impl(const int* task_queue, uint32_t task_idx,
                           const int* atomic_to_zip, const int* num_valid_topk,
                           int* token_done, int* zip_task_queue, int* zip_queue_tail,
                           uint32_t num_tokens) {
    // Wait for the producer (the down GEMM), whose stores must be visible before the counters
    cudaGridDependencySynchronize();

    // Claim the task, in the same way as the compute kernels
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
    const auto lane_idx = ptx::get_lane_idx();

    // Persistent: a fixed number of CTAs strides over the chunk's rows
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < m_size; i += kNumSMs * kNumThreads) {
        const auto token_idx = atomic_to_zip[m_start + i];

        // NOTES: padded rows carry `-1`, and they are never part of a chunk anyway
        if (token_idx >= 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(static_cast<uint32_t>(token_idx) < num_tokens);

            // NOTES: the counter itself needs no memory ordering, as all the chunks of one
            //        token are computed by earlier kernels of this same (serialized) stream
            const auto counted = ptx::atomic_add(token_done + token_idx, 1) + 1;
            const auto expected = num_valid_topk[token_idx];
            DG_TRAP_ONLY_DEVICE_ASSERT(counted <= expected);

            const unsigned bits = __ballot_sync(0xffffffff, counted == expected);
            const unsigned lower_mask = (1u << lane_idx) - 1u;
            const int warp_prefix = __popc(bits & lower_mask);
            const int warp_count = __popc(bits);

            if (bits != 0) {
                // Only lane 0 does the atomic add; other lanes share the base slot and
                // add their prefix in the warp to get their own slots.
                int slot;
                if (lane_idx == 0) {
                    slot = ptx::atomic_add(zip_queue_tail, warp_count);
                    DG_TRAP_ONLY_DEVICE_ASSERT(static_cast<uint32_t>(slot) < num_tokens);
                }
                slot = __shfl_sync(0xffffffff, slot, 0);
                if (counted == expected) {
                    zip_task_queue[slot + warp_prefix] = token_idx;
                }
            }
        }
    }
}

}  // namespace deep_gemm
