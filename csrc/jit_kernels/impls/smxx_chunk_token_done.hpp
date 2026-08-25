#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXChunkTokenDoneRuntime final: public LaunchRuntime<SMXXChunkTokenDoneRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_sms, num_threads;
        void* task_queue;
        uint32_t task_idx;
        void* row_to_token;
        void* token_done;
        uint32_t num_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_chunk_token_done.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_chunk_token_done_impl<
        {}, {}
    >);
}};
)",
        args.num_sms, args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.task_idx,
            args.row_to_token, args.token_done,
            args.num_tokens));
    }
};

// Bump the token completion counters of one chunk, in the persistent form
static void smxx_chunk_token_done(const torch::Tensor& row_to_token,
                                  const torch::Tensor& token_done,
                                  const torch::Tensor& task_queue,
                                  const int& task_idx) {
    constexpr int kNumThreads = 256;

    const auto num_sms = device_runtime->get_num_sms();
    const SMXXChunkTokenDoneRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .task_queue = task_queue.data_ptr(),
        .task_idx = static_cast<uint32_t>(task_idx),
        .row_to_token = row_to_token.data_ptr(),
        .token_done = token_done.data_ptr(),
        .num_tokens = static_cast<uint32_t>(token_done.numel())
    };
    const auto code = SMXXChunkTokenDoneRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_token_done", code);
    SMXXChunkTokenDoneRuntime::launch(runtime, args);
}

} // namespace deep_gemm
