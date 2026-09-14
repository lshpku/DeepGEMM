#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class SMXXChunkWeightedSwigluGradRuntime final: public LaunchRuntime<SMXXChunkWeightedSwigluGradRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_threads, num_topk;
        int num_vecs_per_row, num_elems_per_access;
        bool precise, interleaved, quant;
        void* task_queue;
        uint32_t task_idx;
        void* o1;
        void* probs;
        void* do2;
        void* o2_bwd;
        void* do1;
        void* drecv_probs;
        void* atomic_to_zip;
        void* zip_to_atomic;
        void* recv_token_indices;
        void* o2_bwd_scales;
        void* do1_scales;
        uint32_t scale_stride;
        uint32_t m_alignment;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_chunk_weighted_swiglu_grad.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_chunk_weighted_swiglu_grad_impl<
        {}, {}, {}, {}, {}, {}, {}
    >);
}};
)",
        args.num_threads, args.num_topk, args.num_vecs_per_row, args.num_elems_per_access,
        args.precise, args.interleaved, args.quant);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.task_idx,
            args.o1, args.probs, args.do2,
            args.o2_bwd, args.do1, args.drecv_probs,
            args.atomic_to_zip, args.zip_to_atomic, args.recv_token_indices,
            args.o2_bwd_scales, args.do1_scales, args.scale_stride,
            args.m_alignment));
    }
};

// Backward of the weighted SwiGLU over one chunk task, in the persistent form
static void smxx_chunk_weighted_swiglu_grad(const torch::Tensor& o1,
                                            const torch::Tensor& probs,
                                            const torch::Tensor& do2,
                                            const torch::Tensor& o2_bwd,
                                            const torch::Tensor& do1,
                                            const torch::Tensor& drecv_probs,
                                            const torch::Tensor& atomic_to_zip,
                                            const torch::Tensor& zip_to_atomic,
                                            const torch::Tensor& recv_token_indices,
                                            const std::optional<torch::Tensor>& o2_bwd_scales,
                                            const std::optional<torch::Tensor>& do1_scales,
                                            const torch::Tensor& task_queue,
                                            const int& task_idx,
                                            const int& chunk_size,
                                            const bool& precise,
                                            const bool& interleaved) {
    constexpr int kNumElemsPerAccess = 8;

    const auto quant = o2_bwd_scales.has_value();
    const auto num_blocks = chunk_size;
    // Align to paddle reduce stride: BF16 uses 256 threads and vec8; FP8 uses 256 threads and vec4
    const auto num_threads = quant ? 128 : 256;
    const auto shape_n = static_cast<int>(do2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;

    const SMXXChunkWeightedSwigluGradRuntime::Args args = {
        .launch_args = LaunchArgs(num_blocks, num_threads),
        .num_threads = num_threads,
        .num_topk = static_cast<int>(zip_to_atomic.size(-1)),
        .num_vecs_per_row = num_vecs_per_row,
        .num_elems_per_access = kNumElemsPerAccess,
        .precise = precise,
        .interleaved = interleaved,
        .quant = quant,
        .task_queue = task_queue.data_ptr(),
        .task_idx = static_cast<uint32_t>(task_idx),
        .o1 = o1.data_ptr(),
        .probs = probs.data_ptr(),
        .do2 = do2.data_ptr(),
        .o2_bwd = o2_bwd.data_ptr(),
        .do1 = do1.data_ptr(),
        .drecv_probs = drecv_probs.data_ptr(),
        .atomic_to_zip = atomic_to_zip.data_ptr(),
        .zip_to_atomic = zip_to_atomic.data_ptr(),
        .recv_token_indices = recv_token_indices.data_ptr(),
        .o2_bwd_scales = quant ? o2_bwd_scales->data_ptr() : nullptr,
        .do1_scales = do1_scales.has_value() ? do1_scales->data_ptr() : nullptr,
        .scale_stride = quant ? static_cast<uint32_t>(o2_bwd_scales->stride(-1)) : 0,
        .m_alignment = static_cast<uint32_t>(
            heuristics_runtime->get_mk_alignment_for_contiguous_layout())
    };
    const auto code = SMXXChunkWeightedSwigluGradRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_weighted_swiglu_grad", code);
    SMXXChunkWeightedSwigluGradRuntime::launch(runtime, args);
}

} // namespace deep_gemm
