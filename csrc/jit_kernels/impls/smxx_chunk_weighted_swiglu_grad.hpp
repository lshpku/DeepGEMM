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

        int num_sms, num_threads, num_topk;
        int num_vecs_per_row, num_elems_per_access;
        bool precise, interleaved;
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
        args.num_sms, args.num_threads, args.num_topk, args.num_vecs_per_row,
        args.num_elems_per_access, args.precise, args.interleaved);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.task_idx,
            args.o1, args.probs, args.do2,
            args.o2_bwd, args.do1, args.drecv_probs,
            args.atomic_to_zip, args.zip_to_atomic, args.recv_token_indices,
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
                                            const torch::Tensor& task_queue,
                                            const int& task_idx,
                                            const bool& precise,
                                            const bool& interleaved) {
    constexpr int kNumElemsPerAccess = 8;

    const auto num_sms = device_runtime->get_num_sms();
    const auto num_threads = precise ? 512 : 1024;
    const auto shape_n = static_cast<int>(do2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;

    const SMXXChunkWeightedSwigluGradRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, num_threads),
        .num_sms = num_sms,
        .num_threads = num_threads,
        .num_topk = static_cast<int>(zip_to_atomic.size(-1)),
        .num_vecs_per_row = num_vecs_per_row,
        .num_elems_per_access = kNumElemsPerAccess,
        .precise = precise,
        .interleaved = interleaved,
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
        .m_alignment = static_cast<uint32_t>(heuristics_runtime->get_mk_alignment_for_contiguous_layout())
    };
    const auto code = SMXXChunkWeightedSwigluGradRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_weighted_swiglu_grad", code);
    SMXXChunkWeightedSwigluGradRuntime::launch(runtime, args);
}

} // namespace deep_gemm
