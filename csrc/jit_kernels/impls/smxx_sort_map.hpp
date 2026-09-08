#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXSortMapRuntime final: public LaunchRuntime<SMXXSortMapRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads, num_topk;
        bool output_atomic;
        void* zip_to_atomic;
        void* m_start;
        void* out;
        uint32_t num_recv_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_sort_map.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_sort_map_impl<
        {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_topk, args.output_atomic);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.zip_to_atomic, args.m_start, args.out,
            args.num_recv_tokens));
    }
};

// Build one of the standard unzip order maps, with one CTA per expert
static void smxx_sort_map(const torch::Tensor& zip_to_atomic,
                          const torch::Tensor& m_start,
                          const torch::Tensor& out,
                          const bool& output_atomic) {
    constexpr int kNumThreads = 1024;

    const auto num_experts = static_cast<int>(m_start.numel()) - 1;
    const SMXXSortMapRuntime::Args args = {
        .launch_args = LaunchArgs(num_experts, kNumThreads),
        .num_threads = kNumThreads,
        .num_topk = static_cast<int>(zip_to_atomic.size(1)),
        .output_atomic = output_atomic,
        .zip_to_atomic = zip_to_atomic.data_ptr(),
        .m_start = m_start.data_ptr(),
        .out = out.data_ptr(),
        .num_recv_tokens = static_cast<uint32_t>(zip_to_atomic.size(0))
    };
    const auto code = SMXXSortMapRuntime::generate(args);
    const auto runtime = compiler->build("smxx_sort_map", code);
    SMXXSortMapRuntime::launch(runtime, args);
}

} // namespace deep_gemm
