#pragma once

#include <pybind11/pybind11.h>

#include <cmath>
#include <utility>
#include <vector>

#include "_cpu_binary.hpp"

#include "_cpu_carrier.hpp"

namespace py = pybind11;

namespace strideweave::carrier {

// Maximum/minimum share one comparison rule and one VJP: NaN wins, and an
// exact tie routes the cotangent to the first operand.

template <bool IsMaximum> float cpu_extrema_value(float lhs, float rhs) {
    if (std::isnan(lhs)) {
        return lhs;
    }
    if (std::isnan(rhs)) {
        return rhs;
    }
    if (lhs == rhs) {
        // NumPy resolves mixed signed zeros toward +0 for maximum and -0 for
        // minimum, while preserving the sign when both zeros have that sign.
        if (lhs == 0.0f) {
            const bool negative = IsMaximum ? std::signbit(lhs) && std::signbit(rhs)
                                            : std::signbit(lhs) || std::signbit(rhs);
            return std::copysign(0.0f, negative ? -1.0f : 1.0f);
        }
        return lhs;
    }
    return IsMaximum ? (lhs > rhs ? lhs : rhs) : (lhs < rhs ? lhs : rhs);
}

template <bool IsMaximum>
void cpu_extrema_vjp(float lhs, float rhs, float gradient, float& lhs_gradient,
                     float& rhs_gradient) {
    lhs_gradient = 0.0f;
    rhs_gradient = 0.0f;
    if (std::isnan(lhs) || std::isnan(rhs)) {
        lhs_gradient = rhs_gradient = std::numeric_limits<float>::quiet_NaN();
    } else if (lhs == rhs && std::isfinite(lhs)) {
        lhs_gradient = rhs_gradient = 0.5f * gradient;
    } else if ((IsMaximum && lhs > rhs) || (!IsMaximum && lhs < rhs)) {
        lhs_gradient = gradient;
    } else {
        rhs_gradient = gradient;
    }
}

template <bool IsMaximum>
class CpuExtremaOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, IsMaximum ? "maximum" : "minimum",
            IsMaximum ? "CPU maximum requires lhs and rhs tensors"
                      : "CPU minimum requires lhs and rhs tensors",
            [](float lhs, float rhs) { return cpu_extrema_value<IsMaximum>(lhs, rhs); },
            [](long long lhs, long long rhs) {
                return IsMaximum ? std::max(lhs, rhs) : std::min(lhs, rhs);
            });
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_same_shape(lhs, gradient);
        require_same_shape(rhs, gradient);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation lhs_result = allocate_gradient_for(lhs);
        CpuTensorAllocation rhs_result = allocate_gradient_for(rhs);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                const float x = lhs_view.read_float_expanded(key);
                const float y = rhs_view.read_float_expanded(key);
                const float g = gradient_view.read_float_expanded(key);
                float gx = 0.0f;
                float gy = 0.0f;
                cpu_extrema_vjp<IsMaximum>(x, y, g, gx, gy);
                lhs_result.view.write_float_expanded(key, gx);
                rhs_result.view.write_float_expanded(key, gy);
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }
};

}  // namespace strideweave::carrier
