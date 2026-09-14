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
        void* zip_to_atomic;
        void* m_start;
        void* m_start_out;
        void* out_zip;
        void* out_atomic;
        uint32_t num_recv_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_sort_map.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_sort_map_impl<
        {}, {}
    >);
}};
)",
        args.num_threads, args.num_topk);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.zip_to_atomic, args.m_start, args.m_start_out,
            args.out_zip, args.out_atomic,
            args.num_recv_tokens));
    }
};

// Build both standard unzip order maps in one scan, with one CTA per expert
static void smxx_sort_map(const torch::Tensor& zip_to_atomic,
                          const torch::Tensor& m_start,
                          const torch::Tensor& m_start_out,
                          const torch::Tensor& out_zip,
                          const torch::Tensor& out_atomic) {
    constexpr int kNumThreads = 1024;

    const auto num_experts = static_cast<int>(m_start.numel()) - 1;
    const SMXXSortMapRuntime::Args args = {
        .launch_args = LaunchArgs(num_experts, kNumThreads),
        .num_threads = kNumThreads,
        .num_topk = static_cast<int>(zip_to_atomic.size(1)),
        .zip_to_atomic = zip_to_atomic.data_ptr(),
        .m_start = m_start.data_ptr(),
        .m_start_out = m_start_out.data_ptr(),
        .out_zip = out_zip.data_ptr(),
        .out_atomic = out_atomic.data_ptr(),
        .num_recv_tokens = static_cast<uint32_t>(zip_to_atomic.size(0))
    };
    const auto code = SMXXSortMapRuntime::generate(args);
    const auto runtime = compiler->build("smxx_sort_map", code);
    SMXXSortMapRuntime::launch(runtime, args);
}

} // namespace deep_gemm
