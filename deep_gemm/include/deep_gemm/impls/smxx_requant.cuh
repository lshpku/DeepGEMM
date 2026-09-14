#pragma once

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/fp8_quant.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Requantize a row-gathered FP8 tensor from the hidden dim to the token dim, for the FP8 wgrad
//
// The main branch quantizes along the hidden dim, while `k_grouped_fp8_gemm_tn` contracts over
// the token dim and therefore wants one scale per `(128 token, 1 channel)` block, so the four
// wgrad operands must be quantized a second time. Both scales are powers of two, hence the
// round trip only shifts exponents and is lossless up to saturation.
//
// One block owns `[512 tokens, kBlockH channels]`, i.e. exactly one packed scale int32 per
// channel. The destination rows are the standard (sorted) order padded to 512 per expert, and
// `index` maps them to the source rows, `-1` meaning a zeroed row.
//
// NOTES: the reduction runs along tokens, so the tile is staged in shared memory: the loads and
//        the stores stay coalesced along channels, while a thread owns one whole channel of the
//        tile when reducing
template <uint32_t kNumThreads, uint32_t kBlockH, uint32_t kNumElemsPerAccess>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
smxx_requant_impl(const uint8_t* __restrict__ src, const uint8_t* __restrict__ src_scales,
                  const int* __restrict__ index,
                  uint8_t* __restrict__ out, uint8_t* __restrict__ out_scales,
                  uint32_t hidden, uint32_t src_scale_stride) {
    DG_STATIC_ASSERT(kBlockH == kNumThreads, "A thread must own exactly one channel");
    DG_STATIC_ASSERT(kBlockH % kNumElemsPerAccess == 0, "Invalid channel block");
    DG_STATIC_ASSERT(quant::kGranK % kNumElemsPerAccess == 0, "Invalid granularity");

    using vec_t = int4;
    constexpr uint32_t kNumVecsPerRow = kBlockH / kNumElemsPerAccess;
    constexpr uint32_t kNumVecsPerTile = quant::kGranK * kNumVecsPerRow;
    constexpr uint32_t kNumGranKPerTile = kBlockH / quant::kGranK;

    // Wait for the producer when PDL is enabled
    cudaGridDependencySynchronize();

    // The tile is one quantization block of tokens wide
    __shared__ uint8_t smem_tile[quant::kGranK][kBlockH];
    __shared__ uint8_t smem_src_exp[quant::kGranK][kNumGranKPerTile];
    __shared__ float smem_scale[kBlockH];

    const auto ch_base = blockIdx.y * kBlockH;
    const auto pack_base = blockIdx.x * quant::kGranK * quant::kNumGranKPerPack;

    // The four blocks of a pack share the destination int32, so they are kept in registers
    uint8_t exponents[quant::kNumGranKPerPack];

    #pragma unroll 1
    for (uint32_t b = 0; b < quant::kNumGranKPerPack; ++ b) {
        const auto row_base = pack_base + b * quant::kGranK;

        // Stage the tile and the source exponents
        for (uint32_t i = threadIdx.x; i < kNumVecsPerTile; i += kNumThreads) {
            const auto token = i / kNumVecsPerRow;
            const auto vec_idx = i % kNumVecsPerRow;
            const auto src_row = index[row_base + token];
            vec_t vec = {};
            if (src_row >= 0)
                vec = __ldg(reinterpret_cast<const vec_t*>(
                    src + static_cast<uint64_t>(src_row) * hidden + ch_base +
                    vec_idx * kNumElemsPerAccess));
            *reinterpret_cast<vec_t*>(&smem_tile[token][vec_idx * kNumElemsPerAccess]) = vec;
        }
        for (uint32_t i = threadIdx.x; i < quant::kGranK * kNumGranKPerTile; i += kNumThreads) {
            const auto token = i / kNumGranKPerTile;
            const auto gran_k_idx = ch_base / quant::kGranK + i % kNumGranKPerTile;
            const auto src_row = index[row_base + token];
            uint8_t exponent = 0;
            if (src_row >= 0)
                exponent = src_scales[
                    (static_cast<uint64_t>(gran_k_idx / quant::kNumGranKPerPack) *
                     src_scale_stride + src_row) * 4 + gran_k_idx % quant::kNumGranKPerPack];
            smem_src_exp[token][i % kNumGranKPerTile] = exponent;
        }
        __syncthreads();

        // Reduce along the tokens, one channel per thread
        // NOTES: a source exponent byte is the biased exponent of the dequantization
        //        multiplier, so the byte shifted into place is that power of two
        float amax = 0.0f;
        const auto gran_k_in_tile = threadIdx.x / quant::kGranK;
        #pragma unroll 8
        for (uint32_t token = 0; token < quant::kGranK; ++ token) {
            const auto value = quant::fp8_to_float(smem_tile[token][threadIdx.x]) *
                               __uint_as_float(static_cast<uint32_t>(
                                   smem_src_exp[token][gran_k_in_tile]) << 23);
            amax = fmaxf(amax, fabsf(value));
        }
        const auto block = quant::compute_pow2_scale(amax);
        smem_scale[threadIdx.x] = block.scale;
        exponents[b] = block.exponent;
        __syncthreads();

        // Quantize and store, coalesced along the channels again
        for (uint32_t i = threadIdx.x; i < kNumVecsPerTile; i += kNumThreads) {
            const auto token = i / kNumVecsPerRow;
            const auto ch = (i % kNumVecsPerRow) * kNumElemsPerAccess;
            __nv_fp8x2_e4m3 vec[kNumElemsPerAccess / 2];
            #pragma unroll
            for (uint32_t j = 0; j < kNumElemsPerAccess / 2; ++ j) {
                const auto multiplier = __uint_as_float(static_cast<uint32_t>(
                    smem_src_exp[token][(ch + j * 2) / quant::kGranK]) << 23);
                vec[j] = quant::scale_to_fp8x2(
                    quant::fp8_to_float(smem_tile[token][ch + j * 2]) * multiplier,
                    quant::fp8_to_float(smem_tile[token][ch + j * 2 + 1]) * multiplier,
                    smem_scale[ch + j * 2], smem_scale[ch + j * 2 + 1]);
            }
            *reinterpret_cast<vec_t*>(out + static_cast<uint64_t>(row_base + token) * hidden +
                                      ch_base + ch) = *reinterpret_cast<const vec_t*>(vec);
        }
        __syncthreads();
    }

    // One int32 per channel, so the whole block writes `kBlockH` consecutive int32
    uint32_t pack = 0;
    #pragma unroll
    for (uint32_t b = 0; b < quant::kNumGranKPerPack; ++ b)
        pack |= static_cast<uint32_t>(exponents[b]) << (b * 8);
    reinterpret_cast<uint32_t*>(out_scales)[
        static_cast<uint64_t>(blockIdx.x) * hidden + ch_base + threadIdx.x] = pack;
}

}  // namespace deep_gemm
