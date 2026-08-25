#pragma once

#include <deep_gemm/common/chunk_task.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Publish the token completion of one chunk, i.e. what `zip` waits on
// NOTES: a token is summed by `zip` once all of its top-k experts have counted it,
//        so this only bumps a counter per row, mapped from the unzipped row to the token
template <uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_token_done_impl(const int* task_queue, uint32_t task_idx,
                           const int* row_to_token, int* token_done,
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

    // Persistent: a fixed number of CTAs strides over the chunk's rows
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < m_size; i += kNumSMs * kNumThreads) {
        const auto token_idx = row_to_token[m_start + i];

        // NOTES: padded rows carry `-1`, and they are never part of a chunk anyway
        if (token_idx >= 0) {
            DG_TRAP_ONLY_DEVICE_ASSERT(static_cast<uint32_t>(token_idx) < num_tokens);

            // Release, so that the consumer seeing the counter also sees the down GEMM's output
            ptx::red_add_rel_sys(token_done + token_idx, 1);
        }
    }
}

}  // namespace deep_gemm
