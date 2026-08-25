#include <pybind11/pybind11.h>
#include <torch/python.h>

// Paddle only registers `phi::DataType` (exposed as `paddle.dtype`) with pybind11,
// so bindings using `at::ScalarType` (a standalone enum in the compat layer) need
// a caster that converts through `phi::DataType`.
namespace pybind11::detail {
template <>
struct type_caster<c10::ScalarType> {
    PYBIND11_TYPE_CASTER(c10::ScalarType, const_name("paddle.dtype"));

    bool load(handle src, bool) {
        try {
            value = compat::_PD_PhiDataTypeToAtenScalarType(src.cast<phi::DataType>());
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle cast(const c10::ScalarType& src, return_value_policy, handle) {
        return handle(paddle::pybind::ToPyObject(compat::_PD_AtenScalarTypeToPhiDataType(src)));
    }
};
} // namespace pybind11::detail

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/overlap.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::overlap::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
