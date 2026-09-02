#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class SMXXChunkWeightedSwigluRuntime final: public LaunchRuntime<SMXXChunkWeightedSwigluRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_sms, num_threads, num_elems_per_access;
        int num_vecs_per_row, num_vecs_per_thread;
        bool precise;
        void* task_queue;
        uint32_t task_idx;
        void* o1;
        void* probs;
        void* o2;
        uint32_t m_alignment;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_chunk_weighted_swiglu.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_chunk_weighted_swiglu_impl<
        {}, {}, {}, {}, {}, {}
    >);
}};
)",
        args.num_sms, args.num_threads, args.num_elems_per_access,
        args.num_vecs_per_row, args.num_vecs_per_thread, args.precise);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.task_idx,
            args.o1, args.probs, args.o2,
            args.m_alignment));
    }
};

// Weighted SwiGLU over one chunk task, in the persistent form
static void smxx_chunk_weighted_swiglu(const torch::Tensor& o1,
                                       const torch::Tensor& probs,
                                       const torch::Tensor& o2,
                                       const torch::Tensor& task_queue,
                                       const int& task_idx,
                                       const bool& precise) {
    constexpr int kNumThreads = 1024;
    constexpr int kNumElemsPerAccess = 8;
    // NOTES: one vector already issues two loads with the interleaved layout, so a single
    //        vector per thread keeps enough in flight; sweeping 2/4/8 only made it slower
    constexpr int kNumVecsPerThread = 1;

    const auto num_sms = device_runtime->get_num_sms();
    const auto shape_n = static_cast<int>(o2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);

    const SMXXChunkWeightedSwigluRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .num_elems_per_access = kNumElemsPerAccess,
        .num_vecs_per_row = shape_n / kNumElemsPerAccess,
        .num_vecs_per_thread = kNumVecsPerThread,
        .precise = precise,
        .task_queue = task_queue.data_ptr(),
        .task_idx = static_cast<uint32_t>(task_idx),
        .o1 = o1.data_ptr(),
        .probs = probs.data_ptr(),
        .o2 = o2.data_ptr(),
        .m_alignment = static_cast<uint32_t>(heuristics_runtime->get_mk_alignment_for_contiguous_layout())
    };
    const auto code = SMXXChunkWeightedSwigluRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_weighted_swiglu", code);
    SMXXChunkWeightedSwigluRuntime::launch(runtime, args);
}

} // namespace deep_gemm
