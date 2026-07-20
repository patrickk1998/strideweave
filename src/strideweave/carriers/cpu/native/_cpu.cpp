#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "_cpu_carrier.hpp"
#include "_cpu_operation.hpp"
#include "_carrier.hpp"

namespace py = pybind11;

namespace strideweave::carrier {
namespace {

using CpuOperationFactory = std::function<py::object()>;

std::unordered_map<std::string, CpuOperationFactory>& cpu_operation_registry() {
    static std::unordered_map<std::string, CpuOperationFactory> registry;
    return registry;
}

void register_cpu_operation(
    std::string operation_name, CpuOperationFactory operation_factory
) {
    cpu_operation_registry().insert_or_assign(
        std::move(operation_name), std::move(operation_factory)
    );
}

template <typename Operation>
void register_native_cpu_operation(const char* operation_name) {
    register_cpu_operation(operation_name, [] {
        return py::cast(new Operation(), py::return_value_policy::take_ownership);
    });
}

void register_python_cpu_operation(
    const char* operation_name,
    const char* operation_module_name,
    const char* operation_type_name
) {
    register_cpu_operation(operation_name, [operation_module_name, operation_type_name] {
        return py::module_::import(operation_module_name)
            .attr(operation_type_name)();
    });
}

bool exponent_preserves_int32(float exponent) {
    if (!std::isfinite(exponent)) {
        return false;
    }
    const float rounded = std::round(exponent);
    return exponent == rounded && exponent >= 0.0f &&
           rounded <= static_cast<float>(std::numeric_limits<int>::max());
}

struct CpuExpScalar {
    static constexpr const char* kForwardError = "CPU exp requires a tensor";
    static float value(float input) { return std::exp(input); }
    static float gradient_multiplier(float input) { return std::exp(input); }
};

struct CpuSigmoidScalar {
    static constexpr const char* kForwardError = "CPU sigmoid requires a tensor";
    static float value(float input) { return sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid * (1.0f - sigmoid);
    }
};

struct CpuTanhScalar {
    static constexpr const char* kForwardError = "CPU tanh requires a tensor";
    static float value(float input) { return std::tanh(input); }
    static float gradient_multiplier(float input) {
        const float output = std::tanh(input);
        return 1.0f - output * output;
    }
};

struct CpuGELUScalar {
    static constexpr const char* kForwardError = "CPU GELU requires a tensor";
    static float value(float input) { return gelu_value(input); }
    static float gradient_multiplier(float input) {
        return gelu_gradient_multiplier(input);
    }
};

struct CpuSiLUScalar {
    static constexpr const char* kForwardError = "CPU SiLU requires a tensor";
    static float value(float input) { return input * sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid + input * sigmoid * (1.0f - sigmoid);
    }
};

struct CpuSoftplusScalar {
    static constexpr const char* kForwardError = "CPU softplus requires a tensor";
    static float value(float input) { return softplus_value(input); }
    static float gradient_multiplier(float input) { return sigmoid_value(input); }
};

struct CpuELUScalar {
    static constexpr const char* kForwardError = "CPU ELU requires a tensor";
    static float value(float input) {
        return input > 0.0f ? input : std::expm1(input);
    }
    static float gradient_multiplier(float input) {
        return input > 0.0f ? 1.0f : std::exp(input);
    }
};

struct CpuLeakyReLUScalar {
    static constexpr const char* kForwardError = "CPU leaky ReLU requires a tensor";
    static float value(float input) {
        return input >= 0.0f ? input : kLeakyReluNegativeSlope * input;
    }
    static float gradient_multiplier(float input) {
        return input >= 0.0f ? 1.0f : kLeakyReluNegativeSlope;
    }
};

using CpuExpOperation = CpuUnaryElementwiseOperation<CpuExpScalar>;
using CpuSigmoidOperation = CpuUnaryElementwiseOperation<CpuSigmoidScalar>;
using CpuTanhOperation = CpuUnaryElementwiseOperation<CpuTanhScalar>;
using CpuGELUOperation = CpuUnaryElementwiseOperation<CpuGELUScalar>;
using CpuSiLUOperation = CpuUnaryElementwiseOperation<CpuSiLUScalar>;
using CpuSoftplusOperation = CpuUnaryElementwiseOperation<CpuSoftplusScalar>;
using CpuELUOperation = CpuUnaryElementwiseOperation<CpuELUScalar>;
using CpuLeakyReLUOperation = CpuUnaryElementwiseOperation<CpuLeakyReLUScalar>;

class CpuAddOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU add requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        require_same_layout(lhs, rhs);

        const CpuDType output_dtype = promote_cpu_binary_dtype(lhs, rhs);
        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(lhs), output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                if (output_dtype == CpuDType::Float32) {
                    result.view.write_float_expanded(
                        key,
                        lhs_view.read_float_expanded(key) +
                            rhs_view.read_float_expanded(key)
                    );
                } else {
                    write_int_result(
                        result.view,
                        key,
                        static_cast<long long>(lhs_view.read_int_expanded(key)) +
                            static_cast<long long>(rhs_view.read_int_expanded(key))
                    );
                }
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        return py::make_tuple(
            copy_gradient_for(lhs, gradient),
            copy_gradient_for(rhs, gradient)
        );
    }
};

class CpuSubOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU subtract requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        require_same_layout(lhs, rhs);

        const CpuDType output_dtype = promote_cpu_binary_dtype(lhs, rhs);
        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(lhs), output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                if (output_dtype == CpuDType::Float32) {
                    result.view.write_float_expanded(
                        key,
                        lhs_view.read_float_expanded(key) -
                            rhs_view.read_float_expanded(key)
                    );
                } else {
                    write_int_result(
                        result.view,
                        key,
                        static_cast<long long>(lhs_view.read_int_expanded(key)) -
                            static_cast<long long>(rhs_view.read_int_expanded(key))
                    );
                }
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        return py::make_tuple(
            copy_gradient_for(lhs, gradient),
            copy_negated_gradient_for(rhs, gradient)
        );
    }
};

class CpuScalarMulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU scalar multiply requires a tensor and scalar");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        scalar_ = require_float(inputs[1], "scalar");
        ctx_["scalar"] = py::float_(scalar_);
        const bool scalar_is_integral = is_integral_scalar(inputs[1]);
        const std::int32_t int_scalar =
            scalar_is_integral ? require_int32_scalar(inputs[1], "scalar") : 0;
        const CpuDType output_dtype =
            tensor_view.carrier->cpu_dtype() == CpuDType::Float32 || !scalar_is_integral
                ? CpuDType::Float32
                : CpuDType::Int32;

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (output_dtype == CpuDType::Float32) {
                    result.view.write_float_expanded(
                        key, tensor_view.read_float_expanded(key) * scalar_
                    );
                } else {
                    write_int_result(
                        result.view,
                        key,
                        static_cast<long long>(tensor_view.read_int_expanded(key)) *
                            static_cast<long long>(int_scalar)
                    );
                }
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_layout(tensor, gradient);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key, gradient_view.read_float_expanded(key) * scalar_
                );
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.carrier_object), std::move(result.layout_object))
        );
    }

private:
    float scalar_ = 0.0f;
};

class CpuElementwiseMulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error(
                "CPU elementwise multiply requires lhs and rhs tensors"
            );
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        require_same_layout(lhs, rhs);

        const CpuDType output_dtype = promote_cpu_binary_dtype(lhs, rhs);
        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(lhs), output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                if (output_dtype == CpuDType::Float32) {
                    result.view.write_float_expanded(
                        key,
                        lhs_view.read_float_expanded(key) *
                            rhs_view.read_float_expanded(key)
                    );
                } else {
                    write_int_result(
                        result.view,
                        key,
                        static_cast<long long>(lhs_view.read_int_expanded(key)) *
                            static_cast<long long>(rhs_view.read_int_expanded(key))
                    );
                }
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_same_layout(lhs, gradient);
        require_same_layout(rhs, gradient);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation lhs_result =
            allocate_cpu_tensor(tensor_layout(lhs), CpuDType::Float32);
        CpuTensorAllocation rhs_result =
            allocate_cpu_tensor(tensor_layout(rhs), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                const float gradient_value = gradient_view.read_float_expanded(key);
                lhs_result.view.write_float_expanded(
                    key, gradient_value * rhs_view.read_float_expanded(key)
                );
                rhs_result.view.write_float_expanded(
                    key, gradient_value * lhs_view.read_float_expanded(key)
                );
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }

        return py::make_tuple(
            make_tensor(
                std::move(lhs_result.carrier_object), std::move(lhs_result.layout_object)
            ),
            make_tensor(
                std::move(rhs_result.carrier_object), std::move(rhs_result.layout_object)
            )
        );
    }
};

class CpuDivOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU division requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        require_same_layout(lhs, rhs);

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(lhs), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key,
                    lhs_view.read_float_expanded(key) / rhs_view.read_float_expanded(key)
                );
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_same_layout(lhs, gradient);
        require_same_layout(rhs, gradient);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation lhs_result =
            allocate_cpu_tensor(tensor_layout(lhs), CpuDType::Float32);
        CpuTensorAllocation rhs_result =
            allocate_cpu_tensor(tensor_layout(rhs), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                const float lhs_value = lhs_view.read_float_expanded(key);
                const float rhs_value = rhs_view.read_float_expanded(key);
                const float gradient_value = gradient_view.read_float_expanded(key);
                lhs_result.view.write_float_expanded(key, gradient_value / rhs_value);
                rhs_result.view.write_float_expanded(
                    key, -gradient_value * lhs_value / (rhs_value * rhs_value)
                );
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }

        return py::make_tuple(
            make_tensor(
                std::move(lhs_result.carrier_object), std::move(lhs_result.layout_object)
            ),
            make_tensor(
                std::move(rhs_result.carrier_object), std::move(rhs_result.layout_object)
            )
        );
    }
};

class CpuReLUOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU ReLU requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");

        CpuTensorAllocation result =
            allocate_cpu_tensor(tensor_layout(tensor), tensor_view.carrier->cpu_dtype());
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (tensor_view.carrier->cpu_dtype() == CpuDType::Float32) {
                    const float value = tensor_view.read_float_expanded(key);
                    result.view.write_float_expanded(
                        key, value > 0.0f ? value : 0.0f
                    );
                } else {
                    const std::int32_t value = tensor_view.read_int_expanded(key);
                    result.view.write_int_expanded(key, value > 0 ? value : 0);
                }
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_layout(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                const float value = tensor_view.read_float_expanded(key);
                result.view.write_float_expanded(
                    key, value > 0.0f ? gradient_view.read_float_expanded(key) : 0.0f
                );
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.carrier_object), std::move(result.layout_object))
        );
    }
};

class CpuPowOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU power requires a tensor and exponent");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        exponent_ = require_float(inputs[1], "exponent");
        ctx_["exponent"] = py::float_(exponent_);
        const bool int_output = tensor_view.carrier->cpu_dtype() == CpuDType::Int32 &&
                                exponent_preserves_int32(exponent_);
        const CpuDType output_dtype = int_output ? CpuDType::Int32 : CpuDType::Float32;
        const int int_exponent = int_output ? static_cast<int>(std::round(exponent_)) : 0;

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (output_dtype == CpuDType::Float32) {
                    result.view.write_float_expanded(
                        key,
                        std::pow(tensor_view.read_float_expanded(key), exponent_)
                    );
                } else {
                    write_int_result(
                        result.view,
                        key,
                        checked_int32_pow(
                            tensor_view.read_int_expanded(key), int_exponent
                        )
                    );
                }
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_layout(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key,
                    gradient_view.read_float_expanded(key) * exponent_ *
                        std::pow(
                            tensor_view.read_float_expanded(key), exponent_ - 1.0f
                        )
                );
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.carrier_object), std::move(result.layout_object))
        );
    }

private:
    float exponent_ = 0.0f;
};

class CpuReduceSumOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU reduce requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        require_two_mode_tensor(tensor, "tensor");
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        output_layout_ = canonical_layout_from_modes(
            {mode_shape(tensor_layout(tensor), 0)}
        );
        ctx_["output_layout"] = output_layout_;

        CpuTensorAllocation result =
            allocate_cpu_tensor(output_layout_, tensor_view.carrier->cpu_dtype());
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> input_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                input_key = row_key;
                if (tensor_view.carrier->cpu_dtype() == CpuDType::Float32) {
                    float sum = 0.0f;
                    for (Index j = 0; j < m_size; ++j) {
                        sum += tensor_view.read_float_expanded(input_key);
                        tensor_view.cache->increment_mode(
                            input_key.data(), input_key.size(), 1
                        );
                    }
                    result.view.write_float_expanded(output_key, sum);
                } else {
                    long long sum = 0;
                    for (Index j = 0; j < m_size; ++j) {
                        sum = checked_add(sum, tensor_view.read_int_expanded(input_key));
                        tensor_view.cache->increment_mode(
                            input_key.data(), input_key.size(), 1
                        );
                    }
                    write_int_result(result.view, output_key, sum);
                }
                tensor_view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                result.view.cache->increment_key(
                    output_key.data(), output_key.size()
                );
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        CpuTensorAllocation result = allocate_cpu_tensor(tensor_layout(tensor), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(result.view.leaf_rank(), 0);
            std::vector<Index> input_key(result.view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                const float gradient_value = gradient_view.read_float_expanded(gradient_key);
                input_key = row_key;
                for (Index j = 0; j < m_size; ++j) {
                    result.view.write_float_expanded(input_key, gradient_value);
                    result.view.cache->increment_mode(
                        input_key.data(), input_key.size(), 1
                    );
                }
                result.view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                gradient_view.cache->increment_key(
                    gradient_key.data(), gradient_key.size()
                );
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.carrier_object), std::move(result.layout_object))
        );
    }

private:
    py::object output_layout_ = py::none();
};

class CpuMatmulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU matmul requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        require_two_mode_tensor(lhs, "lhs");
        require_two_mode_tensor(rhs, "rhs");
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index lhs_k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);
        const Index rhs_k_size = mode_logical_size(tensor_layout(rhs), 1);
        if (lhs_k_size != rhs_k_size) {
            throw py::value_error("Matmul inner dimensions must match");
        }

        output_layout_ = canonical_layout_from_modes(
            {mode_shape(tensor_layout(lhs), 0), mode_shape(tensor_layout(rhs), 0)}
        );
        ctx_["output_layout"] = output_layout_;

        const CpuDType output_dtype = promote_cpu_binary_dtype(lhs, rhs);
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, output_dtype);
        {
            py::gil_scoped_release release;
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            std::vector<Index> rhs_j_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_i_base(lhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index j = 0; j < m_size; ++j) {
                std::fill(lhs_i_base.begin(), lhs_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    lhs_key = lhs_i_base;
                    rhs_key = rhs_j_base;
                    if (output_dtype == CpuDType::Float32) {
                        float sum = 0.0f;
                        for (Index k = 0; k < lhs_k_size; ++k) {
                            sum += lhs_view.read_float_expanded(lhs_key) *
                                   rhs_view.read_float_expanded(rhs_key);
                            lhs_view.cache->increment_mode(
                                lhs_key.data(), lhs_key.size(), 1
                            );
                            rhs_view.cache->increment_mode(
                                rhs_key.data(), rhs_key.size(), 1
                            );
                        }
                        result.view.write_float_expanded(output_key, sum);
                    } else {
                        long long sum = 0;
                        for (Index k = 0; k < lhs_k_size; ++k) {
                            sum = checked_add(
                                sum,
                                static_cast<long long>(
                                    lhs_view.read_int_expanded(lhs_key)
                                ) *
                                    static_cast<long long>(
                                        rhs_view.read_int_expanded(rhs_key)
                                    )
                            );
                            lhs_view.cache->increment_mode(
                                lhs_key.data(), lhs_key.size(), 1
                            );
                            rhs_view.cache->increment_mode(
                                rhs_key.data(), rhs_key.size(), 1
                            );
                        }
                        write_int_result(result.view, output_key, sum);
                    }
                    result.view.cache->increment_key(
                        output_key.data(), output_key.size()
                    );
                    lhs_view.cache->increment_mode(
                        lhs_i_base.data(), lhs_i_base.size(), 0
                    );
                }
                rhs_view.cache->increment_mode(
                    rhs_j_base.data(), rhs_j_base.size(), 0
                );
            }
        }
        return make_tensor(std::move(result.carrier_object), std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_layout(gradient, output_layout_);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);

        CpuTensorAllocation lhs_result =
            allocate_cpu_tensor(tensor_layout(lhs), CpuDType::Float32);
        CpuTensorAllocation rhs_result =
            allocate_cpu_tensor(tensor_layout(rhs), CpuDType::Float32);
        {
            py::gil_scoped_release release;
            std::vector<Index> lhs_k_base(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> rhs_k_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_output_key(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_i_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                lhs_output_key = lhs_k_base;
                std::fill(gradient_i_base.begin(), gradient_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    float sum = 0.0f;
                    gradient_key = gradient_i_base;
                    rhs_key = rhs_k_base;
                    for (Index j = 0; j < m_size; ++j) {
                        sum += gradient_view.read_float_expanded(gradient_key) *
                               rhs_view.read_float_expanded(rhs_key);
                        gradient_view.cache->increment_mode(
                            gradient_key.data(), gradient_key.size(), 1
                        );
                        rhs_view.cache->increment_mode(
                            rhs_key.data(), rhs_key.size(), 0
                        );
                    }
                    lhs_result.view.write_float_expanded(lhs_output_key, sum);
                    lhs_result.view.cache->increment_mode(
                        lhs_output_key.data(), lhs_output_key.size(), 0
                    );
                    gradient_view.cache->increment_mode(
                        gradient_i_base.data(), gradient_i_base.size(), 0
                    );
                }
                lhs_result.view.cache->increment_mode(
                    lhs_k_base.data(), lhs_k_base.size(), 1
                );
                rhs_view.cache->increment_mode(
                    rhs_k_base.data(), rhs_k_base.size(), 1
                );
            }

            std::vector<Index> rhs_k_output_base(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> lhs_k_base_for_rhs(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_output_key(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_j_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                rhs_output_key = rhs_k_output_base;
                std::fill(gradient_j_base.begin(), gradient_j_base.end(), 0);
                for (Index j = 0; j < m_size; ++j) {
                    float sum = 0.0f;
                    gradient_key = gradient_j_base;
                    lhs_key = lhs_k_base_for_rhs;
                    for (Index i = 0; i < n_size; ++i) {
                        sum += gradient_view.read_float_expanded(gradient_key) *
                               lhs_view.read_float_expanded(lhs_key);
                        gradient_view.cache->increment_mode(
                            gradient_key.data(), gradient_key.size(), 0
                        );
                        lhs_view.cache->increment_mode(
                            lhs_key.data(), lhs_key.size(), 0
                        );
                    }
                    rhs_result.view.write_float_expanded(rhs_output_key, sum);
                    rhs_result.view.cache->increment_mode(
                        rhs_output_key.data(), rhs_output_key.size(), 0
                    );
                    gradient_view.cache->increment_mode(
                        gradient_j_base.data(), gradient_j_base.size(), 1
                    );
                }
                rhs_result.view.cache->increment_mode(
                    rhs_k_output_base.data(), rhs_k_output_base.size(), 1
                );
                lhs_view.cache->increment_mode(
                    lhs_k_base_for_rhs.data(), lhs_k_base_for_rhs.size(), 1
                );
            }
        }

        return py::make_tuple(
            make_tensor(
                std::move(lhs_result.carrier_object), std::move(lhs_result.layout_object)
            ),
            make_tensor(
                std::move(rhs_result.carrier_object), std::move(rhs_result.layout_object)
            )
        );
    }

private:
    py::object output_layout_ = py::none();
};

}  // namespace

py::object CPU::dispatch_op(const std::string& operation_name) const {
    auto& registry = cpu_operation_registry();
    auto operation_factory = registry.find(operation_name);
    if (operation_factory == registry.end()) {
        PyErr_Format(
            PyExc_NotImplementedError,
            "CPU carrier does not support operation '%s'",
            operation_name.c_str()
        );
        throw py::error_already_set();
    }
    return operation_factory->second();
}

void bind_cpu(py::module_& module) {
    py::class_<CPU, Carrier>(module, "CPU")
        .def(
            py::init<Index, py::object, bool, py::object>(),
            py::arg("size"),
            py::arg("pointer") = py::none(),
            py::kw_only(),
            py::arg("mutable") = true,
            py::arg("dtype") = py::none()
        )
        .def(
            "new_like",
            &CPU::new_like_with_dtype,
            py::arg("values"),
            py::kw_only(),
            py::arg("mutable") = true,
            py::arg("dtype") = py::none()
        )
        .def(
            "empty_like",
            &CPU::empty_like,
            py::arg("size"),
            py::kw_only(),
            py::arg("mutable") = true,
            py::arg("dtype") = py::none()
        )
        .def("pointer", &CPU::pointer)
        .def("set_value", &CPU::set_value_public, py::arg("index"), py::arg("value"))
        .def("dispatch_op", &CPU::dispatch_op, py::arg("operation_name"));

    bind_cpu_operation<CpuAddOperation>(module, "_CPUAddOperation");
    bind_cpu_operation<CpuSubOperation>(module, "_CPUSubOperation");
    bind_cpu_operation<CpuScalarMulOperation>(module, "_CPUScalarMulOperation");
    bind_cpu_operation<CpuElementwiseMulOperation>(
        module, "_CPUElementwiseMulOperation"
    );
    bind_cpu_operation<CpuDivOperation>(module, "_CPUDivOperation");
    bind_cpu_operation<CpuExpOperation>(module, "_CPUExpOperation");
    bind_cpu_operation<CpuReLUOperation>(module, "_CPUReLUOperation");
    bind_cpu_operation<CpuSigmoidOperation>(module, "_CPUSigmoidOperation");
    bind_cpu_operation<CpuTanhOperation>(module, "_CPUTanhOperation");
    bind_cpu_operation<CpuGELUOperation>(module, "_CPUGELUOperation");
    bind_cpu_operation<CpuSiLUOperation>(module, "_CPUSiLUOperation");
    bind_cpu_operation<CpuSoftplusOperation>(module, "_CPUSoftplusOperation");
    bind_cpu_operation<CpuELUOperation>(module, "_CPUELUOperation");
    bind_cpu_operation<CpuLeakyReLUOperation>(module, "_CPULeakyReLUOperation");
    bind_cpu_operation<CpuPowOperation>(module, "_CPUPowOperation");
    bind_cpu_operation<CpuReduceSumOperation>(module, "_CPUReduceSumOperation");
    bind_cpu_operation<CpuMatmulOperation>(module, "_CPUMatmulOperation");

    register_native_cpu_operation<CpuAddOperation>("add");
    register_native_cpu_operation<CpuDivOperation>("div");
    register_native_cpu_operation<CpuELUOperation>("elu");
    register_native_cpu_operation<CpuElementwiseMulOperation>("elementwise_mul");
    register_native_cpu_operation<CpuExpOperation>("exp");
    register_native_cpu_operation<CpuGELUOperation>("gelu");
    register_native_cpu_operation<CpuLeakyReLUOperation>("leaky_relu");
    register_native_cpu_operation<CpuMatmulOperation>("matmul");
    register_native_cpu_operation<CpuScalarMulOperation>("mul");
    register_native_cpu_operation<CpuPowOperation>("pow");
    register_native_cpu_operation<CpuReduceSumOperation>("reduce");
    register_native_cpu_operation<CpuReLUOperation>("relu");
    register_native_cpu_operation<CpuSigmoidOperation>("sigmoid");
    register_native_cpu_operation<CpuSiLUOperation>("silu");
    register_native_cpu_operation<CpuSoftplusOperation>("softplus");
    register_native_cpu_operation<CpuSubOperation>("sub");
    register_native_cpu_operation<CpuTanhOperation>("tanh");
    register_python_cpu_operation(
        "permute", "strideweave.carriers.shared_ops", "PermuteOperation"
    );
    register_python_cpu_operation(
        "rearrange", "strideweave.carriers.shared_ops", "RearrangeOperation"
    );
    register_python_cpu_operation(
        "view", "strideweave.carriers.shared_ops", "GenericViewOperation"
    );
}

}  // namespace strideweave::carrier
