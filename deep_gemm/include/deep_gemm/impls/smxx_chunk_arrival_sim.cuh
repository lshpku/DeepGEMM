#pragma once

#include <deep_gemm/common/chunk_task.cuh>

namespace deep_gemm {

// Mark the chunks as arrived one by one, standing in for the communication kernel
// NOTES: for single-card debugging only, the real producer is the DeepEP dispatch
template <uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
smxx_chunk_arrival_sim_impl(int* task_queue, uint32_t num_tasks, uint64_t interval_cycles) {
    if (threadIdx.x != 0)
        return;

    for (uint32_t task_idx = 0; task_idx < num_tasks; ++ task_idx) {
        chunk::wait_cycles(interval_cycles);
        chunk::set_task_ready(task_queue, task_idx);
    }
}

}  // namespace deep_gemm
