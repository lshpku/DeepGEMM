#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXRequantRuntime final: public LaunchRuntime<SMXXRequantRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads, block_h, num_elems_per_access;
        void* src;
        void* src_scales;
        void* index;
        void* out;
        void* out_scales;
        uint32_t hidden, src_scale_stride;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_requant.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_requant_impl<
        {}, {}, {}
    >);
}};
)",
        args.num_threads, args.block_h, args.num_elems_per_access);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.src, args.src_scales, args.index,
            args.out, args.out_scales,
            args.hidden, args.src_scale_stride));
    }
};

// Requantize the gathered rows along the token dim, one block per `[512 token, kBlockH channel]`
static void smxx_requant(const torch::Tensor& src,
                         const torch::Tensor& src_scales,
                         const torch::Tensor& index,
                         const torch::Tensor& out,
                         const torch::Tensor& out_scales,
                         const int& num_dst_rows, const int& hidden) {
    constexpr int kNumThreads = 256;
    constexpr int kBlockH = kNumThreads;
    constexpr int kNumElemsPerAccess = 16;
    constexpr int kNumTokensPerPack = 512;

    DG_HOST_ASSERT(hidden % kBlockH == 0);
    DG_HOST_ASSERT(num_dst_rows % kNumTokensPerPack == 0);

    const SMXXRequantRuntime::Args args = {
        .launch_args = LaunchArgs({static_cast<int>(num_dst_rows / kNumTokensPerPack),
                                   static_cast<int>(hidden / kBlockH)}, kNumThreads),
        .num_threads = kNumThreads,
        .block_h = kBlockH,
        .num_elems_per_access = kNumElemsPerAccess,
        .src = src.data_ptr(),
        .src_scales = src_scales.data_ptr(),
        .index = index.data_ptr(),
        .out = out.data_ptr(),
        .out_scales = out_scales.data_ptr(),
        .hidden = static_cast<uint32_t>(hidden),
        .src_scale_stride = static_cast<uint32_t>(src_scales.stride(-1))
    };
    const auto code = SMXXRequantRuntime::generate(args);
    const auto runtime = compiler->build("smxx_requant", code);
    SMXXRequantRuntime::launch(runtime, args);
}

} // namespace deep_gemm
