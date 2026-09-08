#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXTokenGatherRuntime final: public LaunchRuntime<SMXXTokenGatherRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_sms, num_threads, num_vecs_per_row;
        void* x;
        void* index;
        void* out;
        uint32_t num_rows;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_token_gather.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_token_gather_impl<
        {}, {}, {}
    >);
}};
)",
        args.num_sms, args.num_threads, args.num_vecs_per_row);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.x, args.index, args.out,
            args.num_rows));
    }
};

// Row gather with `-1` meaning a zeroed row, in the persistent form
static void smxx_token_gather(const torch::Tensor& x,
                              const torch::Tensor& index,
                              const torch::Tensor& out,
                              const int& row_bytes) {
    constexpr int kNumThreads = 1024;
    constexpr int kNumBytesPerAccess = 16;
    DG_HOST_ASSERT(row_bytes % kNumBytesPerAccess == 0);
    for (const auto& t: {x, out})
        DG_HOST_ASSERT(reinterpret_cast<uint64_t>(t.data_ptr()) % kNumBytesPerAccess == 0);

    const auto num_sms = device_runtime->get_num_sms();
    const SMXXTokenGatherRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .num_vecs_per_row = row_bytes / kNumBytesPerAccess,
        .x = x.data_ptr(),
        .index = index.data_ptr(),
        .out = out.data_ptr(),
        .num_rows = static_cast<uint32_t>(index.numel())
    };
    const auto code = SMXXTokenGatherRuntime::generate(args);
    const auto runtime = compiler->build("smxx_token_gather", code);
    SMXXTokenGatherRuntime::launch(runtime, args);
}

} // namespace deep_gemm
