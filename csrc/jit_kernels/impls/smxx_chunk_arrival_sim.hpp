#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXChunkArrivalSimRuntime final: public LaunchRuntime<SMXXChunkArrivalSimRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads;
        void* task_queue;
        uint32_t num_tasks;
        uint64_t interval_cycles;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_chunk_arrival_sim.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_chunk_arrival_sim_impl<
        {}
    >);
}};
)",
        args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.num_tasks, args.interval_cycles));
    }
};

// A stand-in producer for single-card debugging, occupying a single SM
static void smxx_chunk_arrival_sim(const torch::Tensor& task_queue, const int& interval_ns) {
    constexpr int kNumThreads = 32;

    // The kernel counts SM cycles, as chained `nanosleep`s overshoot badly
    // NOTES: `cudaDeviceProp::clockRate` is gone since CUDA 13, so query the attribute
    int device_index = 0, clock_khz = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
    DG_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_index));
    const auto cycles_per_ns = static_cast<double>(clock_khz) / 1e6;
    const SMXXChunkArrivalSimRuntime::Args args = {
        .launch_args = LaunchArgs(1, kNumThreads),
        .num_threads = kNumThreads,
        .task_queue = task_queue.data_ptr(),
        .num_tasks = static_cast<uint32_t>(task_queue.size(0)),
        .interval_cycles = static_cast<uint64_t>(interval_ns * cycles_per_ns)
    };
    const auto code = SMXXChunkArrivalSimRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_arrival_sim", code);
    SMXXChunkArrivalSimRuntime::launch(runtime, args);
}

} // namespace deep_gemm
