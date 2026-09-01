#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

namespace deep_gemm {

class SMXXChunkZipRuntime final: public LaunchRuntime<SMXXChunkZipRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;

        int num_sms, num_threads, max_rows_per_cta;
        int num_topk, num_vecs_per_row, num_elems_per_access;
        void* task_queue;
        uint32_t task_idx;
        void* o3;
        void* combine_input;
        void* atomic_to_zip;
        void* zip_to_atomic;
        void* recv_token_indices;
        void* num_valid_topk;
        void* token_done;
        void* zip_done;
        uint32_t num_tokens;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_chunk_zip.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_chunk_zip_impl<
        {}, {}, {}, {}, {}, {}
    >);
}};
)",
        args.num_sms, args.num_threads, args.max_rows_per_cta,
        args.num_topk, args.num_vecs_per_row, args.num_elems_per_access);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.task_queue, args.task_idx,
            args.o3, args.combine_input,
            args.atomic_to_zip, args.zip_to_atomic,
            args.recv_token_indices, args.num_valid_topk,
            args.token_done, args.zip_done,
            args.num_tokens));
    }
};
// Fused token completion and zip over one chunk task, in the persistent form
static void smxx_chunk_zip(const torch::Tensor& o3, const torch::Tensor& combine_input,
                           const torch::Tensor& atomic_to_zip, const torch::Tensor& zip_to_atomic,
                           const torch::Tensor& recv_token_indices,
                           const torch::Tensor& num_valid_topk,
                           const torch::Tensor& token_done, const torch::Tensor& zip_done,
                           const torch::Tensor& task_queue,
                           const int& task_idx, const int& chunk_size) {
    constexpr int kNumThreads = 512;
    constexpr int kNumElemsPerAccess = 8;

    const auto num_sms = device_runtime->get_num_sms();
    const auto shape_n = static_cast<int>(o3.size(-1));
    DG_HOST_ASSERT(shape_n % kNumElemsPerAccess == 0);

    // The local queue must hold every token a CTA can possibly finish, i.e. all of its rows
    auto max_rows_per_cta = 1;
    while (max_rows_per_cta < (chunk_size + num_sms - 1) / num_sms)
        max_rows_per_cta *= 2;

    const SMXXChunkZipRuntime::Args args = {
        .launch_args = LaunchArgs(num_sms, kNumThreads),
        .num_sms = num_sms,
        .num_threads = kNumThreads,
        .max_rows_per_cta = max_rows_per_cta,
        .num_topk = static_cast<int>(zip_to_atomic.size(1)),
        .num_vecs_per_row = shape_n / kNumElemsPerAccess,
        .num_elems_per_access = kNumElemsPerAccess,
        .task_queue = task_queue.data_ptr(),
        .task_idx = static_cast<uint32_t>(task_idx),
        .o3 = o3.data_ptr(),
        .combine_input = combine_input.data_ptr(),
        .atomic_to_zip = atomic_to_zip.data_ptr(),
        .zip_to_atomic = zip_to_atomic.data_ptr(),
        .recv_token_indices = recv_token_indices.data_ptr(),
        .num_valid_topk = num_valid_topk.data_ptr(),
        .token_done = token_done.data_ptr(),
        .zip_done = zip_done.data_ptr(),
        .num_tokens = static_cast<uint32_t>(token_done.numel())
    };
    const auto code = SMXXChunkZipRuntime::generate(args);
    const auto runtime = compiler->build("smxx_chunk_zip", code);
    SMXXChunkZipRuntime::launch(runtime, args);
}

} // namespace deep_gemm
