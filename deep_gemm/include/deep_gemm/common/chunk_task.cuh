#pragma once

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::chunk {

// Give up the SM for a while between polls, to keep the traffic low
// NOTES: the queue may live in mapped host memory, where every poll is a PCIe read
constexpr uint32_t kPollIntervalNs = 512;

// Bail out instead of hanging the device forever, if the producer never arrives
// NOTES: about 15 seconds on a 2GHz SM clock
constexpr uint64_t kReadyTimeoutCycles = 30000000000ull;

struct ChunkTask {
    int expert_idx;
    int m_start;
    int m_size;
};

// Wait until the chunk has arrived, to be called by one thread per CTA
// NOTES: the flag is loaded with system scope, so both a device-memory queue (the real
//        dispatch) and a mapped host memory queue (single-card simulation) work
CUTLASS_DEVICE void wait_task_ready(const int* task_queue, const uint32_t& task_idx) {
    const auto* ready = task_queue + task_idx * kNumChunkTaskFields + 3;
    const auto start_clock = clock64();
    while (ptx::ld_acq_sys(ready) == 0) {
        __nanosleep(kPollIntervalNs);
        DG_DEVICE_ASSERT(static_cast<uint64_t>(clock64() - start_clock) < kReadyTimeoutCycles and
                         "Timeout on waiting for the chunk to arrive");
    }
}

// Read the task descriptor from the queue, to be called by one thread per CTA
// NOTES: system-scope loads as well, and the result should be staged into shared memory:
//        every thread reading the queue directly costs thousands of uncached reads
CUTLASS_DEVICE ChunkTask read_task(const int* task_queue, const uint32_t& task_idx) {
    const auto* task = task_queue + task_idx * kNumChunkTaskFields;
    return {ptx::ld_acq_sys(task), ptx::ld_acq_sys(task + 1), ptx::ld_acq_sys(task + 2)};
}

// Stage the descriptor for the whole CTA, to be called by one thread before a barrier
CUTLASS_DEVICE void stage_task(int* staged, const ChunkTask& task) {
    staged[0] = task.expert_idx;
    staged[1] = task.m_start;
    staged[2] = task.m_size;
}

// Load the staged descriptor, to be called by all the threads after the barrier
CUTLASS_DEVICE ChunkTask load_staged_task(const int* staged) {
    return {staged[0], staged[1], staged[2]};
}

// Publish the arrival of a chunk, i.e. what the communication kernel does
CUTLASS_DEVICE void set_task_ready(int* task_queue, const uint32_t& task_idx) {
    ptx::st_rel_sys(task_queue + task_idx * kNumChunkTaskFields + 3, 1);
}

// Wait for a given number of SM cycles, polling instead of busy-spinning
// NOTES: `nanosleep` alone overshoots badly when chained, so the clock decides when to stop
CUTLASS_DEVICE void wait_cycles(uint64_t cycles) {
    const auto start_clock = clock64();
    while (static_cast<uint64_t>(clock64() - start_clock) < cycles)
        __nanosleep(kPollIntervalNs);
}

} // namespace deep_gemm::chunk
