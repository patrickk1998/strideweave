#pragma once

#include <pybind11/pybind11.h>

#include <cmath>
#include <utility>
#include <vector>

#include "_cpu_data.hpp"
#include "_operation.hpp"

namespace py = pybind11;

namespace neotorch::data {

inline float sigmoid_value(float value) {
    if (value >= 0.0f) {
        const float inverse = std::exp(-value);
        return 1.0f / (1.0f + inverse);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

inline float softplus_value(float value) {
    return std::log1p(std::exp(-std::fabs(value))) + std::max(value, 0.0f);
}

constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
constexpr float kLeakyReluNegativeSlope = 0.01f;

inline float gelu_value(float value) {
    return 0.5f * value * (1.0f + std::erf(value * kInvSqrt2));
}

inline float gelu_gradient_multiplier(float value) {
    return 0.5f * (1.0f + std::erf(value * kInvSqrt2)) +
           value * std::exp(-0.5f * value * value) * kInvSqrt2Pi;
}

// Unary elementwise CPU operation parameterized on a scalar policy:
//
//     struct Scalar {
//         static constexpr const char* kForwardError = "CPU op requires a tensor";
//         static float value(float input);
//         static float gradient_multiplier(float input);
//     };
//
// forward writes value(x) elementwise into a Float32 result; backward writes
// gradient * gradient_multiplier(x).
template <typename Scalar>
class CpuUnaryElementwiseOperation : public neotorch::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error(Scalar::kForwardError);
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");

        CpuTensorAllocation result =
            allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key, Scalar::value(tensor_view.read_float_expanded(key))
                );
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(
            std::move(result.data_object), std::move(result.layout_object)
        );
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_layout(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation result =
            allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key,
                    gradient_view.read_float_expanded(key) *
                        Scalar::gradient_multiplier(
                            tensor_view.read_float_expanded(key)
                        )
                );
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.data_object), std::move(result.layout_object))
        );
    }
};

template <typename Derived>
void bind_cpu_operation(py::module_& module, const char* name) {
    py::class_<Derived, neotorch::operation::Operation>(module, name)
        .def(py::init<>());
}

}  // namespace neotorch::data
