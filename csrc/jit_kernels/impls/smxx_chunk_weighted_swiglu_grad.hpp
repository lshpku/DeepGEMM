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

        int num_sms, num_threads, num_warps_per_row, num_topk;
        int num_vecs_per_row, num_elems_per_access;
        bool precise;
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
        args.num_sms, args.num_threads, args.num_warps_per_row, args.num_topk,
        args.num_vecs_per_row, args.num_elems_per_access, args.precise);
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
                                            const bool& precise) {
    constexpr int kNumThreads = 1024;
    constexpr int kNumElemsPerAccess = 8;
    // NOTES: more vectors per lane would put more loads in flight and more rows in flight per
    //        CTA, and it did measure faster, but 2 and 4 both hit a `cudaErrorIllegalAddress`
    //        that only the `precise == false` instantiation shows, that `compute-sanitizer`
    //        cannot see, and that survives having every index bound checked on the device; it
    //        moves with unrelated codegen changes, so keep one vector per lane until it is
    //        understood, and re-run `tests_overlap/test_backward.py` in both modes if it changes
    constexpr int kNumVecsPerThread = 1;

    const auto num_sms = device_runtime->get_num_sms();
    const auto shape_n = static_cast<int>(do2.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);
    const auto num_vecs_per_row = shape_n / kNumElemsPerAccess;

    // One row is owned by as many warps as cover it in `kNumVecsPerThread` steps, so that the
    // `probs` gradient reduction stays inside one group and the row's traffic stays contiguous
    DG_HOST_ASSERT(num_vecs_per_row % 32 == 0);
    const auto num_warps_per_row = std::max(1, std::min(kNumThreads / 32,
                                                        num_vecs_per_row / 32 / kNumVecsPerThread));
    DG_HOST_ASSERT(num_vecs_per_row % (num_warps_per_row * 32) == 0);
    DG_HOST_ASSERT((kNumThreads / 32) % num_warps_per_row == 0);

    const SMXXChunkWeightedSwigluGradRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .num_warps_per_row = num_warps_per_row,
        .num_topk = static_cast<int>(zip_to_atomic.size(-1)),
        .num_vecs_per_row = num_vecs_per_row,
        .num_elems_per_access = kNumElemsPerAccess,
        .precise = precise,
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
