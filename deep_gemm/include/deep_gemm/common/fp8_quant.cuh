#pragma once

#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::quant {

// The 1D1D recipe: one UE8M0 scale per 128 elements along the contraction dim
constexpr uint32_t kGranK = 128;
// Four UE8M0 exponents are packed into one int32, so one scale column spans 512 elements
constexpr uint32_t kNumGranKPerPack = 4;
// A vector is 8 elements, i.e. one 8B FP8 store, so a block is 16 lanes wide
constexpr uint32_t kNumElemsPerVec = 8;
constexpr uint32_t kNumLanesPerBlock = kGranK / kNumElemsPerVec;

// The scale of a zeroed quantization block is the exponent byte of `1.0f`, so a whole pack of
// four zeroed blocks is this
constexpr uint32_t kZeroBlockPack = 0x7f7f7f7fu;

// The scale of one quantization block: a power of two, plus the UE8M0 exponent byte of its
// reciprocal, which is what the GEMM reads back
struct BlockScale {
    float scale;
    uint8_t exponent;
};

// `ComputeScale<float, __nv_fp8_e4m3, true>` of Paddle's `quant_utils.h`, bit for bit
// NOTES: the scale is rounded down to a power of two, so its reciprocal is exact and both are
//        built straight from the exponent field instead of calling `ldexpf` and `__frcp_rn`
// NOTES: `RoundPower2Scale` maps the saturated scale back to `1.0f` on everything but SM90,
//        and it applies to all the paths, including `amax == 0` and the overflow one
// NOTES: an infinite `amax` is out of contract, as it is for the BF16 path
CUTLASS_DEVICE BlockScale compute_pow2_scale(const float& amax) {
    constexpr float kFP8Max = 448.0f;
    int32_t biased_exp = 0;
    if (amax != 0.0f) {
        const auto scale = kFP8Max / amax;
        if (not isinf(scale))
            biased_exp = static_cast<int32_t>(__float_as_uint(scale) >> 23) - 127;
        if (biased_exp == 127)
            biased_exp = 0;
    }
    return {__uint_as_float(static_cast<uint32_t>(biased_exp + 127) << 23),
            static_cast<uint8_t>(127 - biased_exp)};
}

// The MN-major (transposed) packed UE8M0 layout that DeepGEMM's TMA descriptor requires: the
// exponent of `[row, gran_k_idx]` is the `gran_k_idx % 4`-th byte of the int32 at
// `[gran_k_idx / 4, row]`, whose row stride is the whole buffer's (TMA aligned) row count
CUTLASS_DEVICE uint32_t* get_scale_pack(uint8_t* scales, const uint32_t& scale_stride,
                                        const uint32_t& row, const uint32_t& pack_idx) {
    return reinterpret_cast<uint32_t*>(scales) +
           static_cast<uint64_t>(pack_idx) * scale_stride + row;
}

// Publish a row's staged exponents as whole int32 packs
// NOTES: a byte store makes L2 read-modify-write a whole sector, and one row's 128-column
//        blocks are 4 bytes apart at best, so the exponents are staged in shared memory and
//        flushed as packs; the packs of consecutive rows are consecutive int32 as well
CUTLASS_DEVICE void store_scale_packs(uint8_t* scales, const uint32_t& scale_stride,
                                      const uint32_t& row, const uint8_t* staged,
                                      const uint32_t& num_packs, const uint32_t& pack_idx) {
    if (pack_idx < num_packs)
        *get_scale_pack(scales, scale_stride, row, pack_idx) =
            *reinterpret_cast<const uint32_t*>(staged + pack_idx * 4);
}

// The scales of an all-zero row, which is what a padded row must publish
CUTLASS_DEVICE void store_zero_packs(uint8_t* scales, const uint32_t& scale_stride,
                                     const uint32_t& row, const uint32_t& num_packs,
                                     const uint32_t& pack_idx) {
    if (pack_idx < num_packs)
        *get_scale_pack(scales, scale_stride, row, pack_idx) = kZeroBlockPack;
}

// One exponent byte, for the callers that cannot stage a whole row
// NOTES: measured to be no worse than the staged version, as the packs of the rows that run
//        concurrently end up in the same sectors anyway
CUTLASS_DEVICE void store_scale(uint8_t* scales, const uint32_t& scale_stride,
                                const uint32_t& row, const uint32_t& gran_k_idx,
                                const uint8_t exponent) {
    auto* pack = get_scale_pack(scales, scale_stride, row, gran_k_idx / kNumGranKPerPack);
    reinterpret_cast<uint8_t*>(pack)[gran_k_idx % kNumGranKPerPack] = exponent;
}

// `static_cast<__nv_fp8_e4m3>(value * scale)` on a pair, which is one `cvt.rn.satfinite`
CUTLASS_DEVICE __nv_fp8x2_e4m3 scale_to_fp8x2(const float& lhs, const float& rhs,
                                              const float& scale) {
    const float2 scaled = {lhs * scale, rhs * scale};
    __nv_fp8x2_e4m3 out;
    out.__x = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    return out;
}

// The same, for the requant pass where every channel has its own scale
CUTLASS_DEVICE __nv_fp8x2_e4m3 scale_to_fp8x2(const float& lhs, const float& rhs,
                                              const float& scale_lhs, const float& scale_rhs) {
    const float2 scaled = {lhs * scale_lhs, rhs * scale_rhs};
    __nv_fp8x2_e4m3 out;
    out.__x = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    return out;
}

// FP8 to FP32 is exact, as E4M3 has fewer mantissa bits than either FP32 or BF16
CUTLASS_DEVICE float fp8_to_float(const uint8_t byte) {
    __nv_fp8_e4m3 value;
    value.__x = byte;
    return static_cast<float>(value);
}

// All-reduce `amax` over the `kNumLanes` lanes that share one quantization block
// NOTES: a butterfly, so every lane ends up with the block's `amax`; `fmaxf` is exact,
//        hence the reduction order does not have to match the reference kernel's
template <uint32_t kNumLanes>
CUTLASS_DEVICE float group_amax(float amax) {
    DG_STATIC_ASSERT(kNumLanes == 8 or kNumLanes == 16 or kNumLanes == 32, "Invalid lane number");
    #pragma unroll
    for (uint32_t offset = kNumLanes / 2; offset > 0; offset /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    return amax;
}

// Quantize one 8B vector of a row and return its block's exponent, to be staged by the caller
// NOTES: `active` only masks the store, as every lane must join the `amax` butterfly; the
//        offsets are in units of one vector, and `num_vecs_per_row` is the output row's
CUTLASS_DEVICE uint8_t store_quantized(void* out, const float (&act)[kNumElemsPerVec],
                                       const uint32_t& row, const uint32_t& vec_idx,
                                       const bool& active,
                                       const uint32_t& num_vecs_per_row) {
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kNumElemsPerVec; ++ i)
        amax = fmaxf(amax, fabsf(act[i]));
    const auto block = compute_pow2_scale(group_amax<kNumLanesPerBlock>(amax));

    if (active) {
        __nv_fp8x2_e4m3 vec[kNumElemsPerVec / 2];
        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerVec / 2; ++ i)
            vec[i] = scale_to_fp8x2(act[i * 2], act[i * 2 + 1], block.scale);

        const auto off = static_cast<uint64_t>(row) * num_vecs_per_row + vec_idx;
        *(reinterpret_cast<uint2*>(out) + off) = *reinterpret_cast<const uint2*>(vec);
    }
    return block.exponent;
}

}  // namespace deep_gemm::quant
