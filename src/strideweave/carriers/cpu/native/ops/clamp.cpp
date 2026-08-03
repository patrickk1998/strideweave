#include "_cpu_binary.hpp"
#include "_cpu_extrema.hpp"
#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuClampOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 3) {
            throw py::type_error(
                "CPU clamp requires a tensor, lower bound, and upper bound");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object lower = py::reinterpret_borrow<py::object>(inputs[1]);
        py::object upper = py::reinterpret_borrow<py::object>(inputs[2]);
        lower_is_tensor_ = py::isinstance(lower, tensor_type());
        upper_is_tensor_ = py::isinstance(upper, tensor_type());

        CpuTensorView original_tensor = cpu_tensor_view(tensor, "tensor");
        if (lower_is_tensor_) {
            require_same_carrier_class(tensor, lower,
                                       "Tensor backing carriers must match");
        }
        if (upper_is_tensor_) {
            require_same_carrier_class(tensor, upper,
                                       "Tensor backing carriers must match");
        }

        const CpuPlan plan = [&]() {
            if (lower_is_tensor_ && upper_is_tensor_) {
                CpuTensorView lower_view = cpu_tensor_view(lower, "lower");
                CpuTensorView upper_view = cpu_tensor_view(upper, "upper");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()),
                    cpu_dtype_object(lower_view.carrier->cpu_dtype()),
                    cpu_dtype_object(upper_view.carrier->cpu_dtype()));
            }
            if (lower_is_tensor_) {
                CpuTensorView lower_view = cpu_tensor_view(lower, "lower");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()),
                    cpu_dtype_object(lower_view.carrier->cpu_dtype()), upper);
            }
            if (upper_is_tensor_) {
                CpuTensorView upper_view = cpu_tensor_view(upper, "upper");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()), lower,
                    cpu_dtype_object(upper_view.carrier->cpu_dtype()));
            }
            return resolve_cpu_plan(
                executing_carrier_class(tensor), "clamp",
                cpu_dtype_object(original_tensor.carrier->cpu_dtype()), lower, upper);
        }();

        if (!lower_is_tensor_) {
            lower_scalar_ = require_float(lower, "lower");
        }
        if (!upper_is_tensor_) {
            upper_scalar_ = require_float(upper, "upper");
        }

        if (lower_is_tensor_ && upper_is_tensor_) {
            AlignedTernaryOperands aligned = align_ternary_operands(
                std::move(tensor), std::move(lower), std::move(upper));
            tensor = std::move(aligned.first);
            lower = std::move(aligned.second);
            upper = std::move(aligned.third);
            output_layout_ = std::move(aligned.result_layout);
        } else if (lower_is_tensor_) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(tensor), std::move(lower));
            tensor = std::move(aligned.lhs);
            lower = std::move(aligned.rhs);
            output_layout_ = std::move(aligned.result_layout);
        } else if (upper_is_tensor_) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(tensor), std::move(upper));
            tensor = std::move(aligned.lhs);
            upper = std::move(aligned.rhs);
            output_layout_ = std::move(aligned.result_layout);
        } else {
            py::object top_level =
                tensor_layout(tensor).attr("shape").attr("top_level");
            output_layout_ = canonical_layout_from_top_level(
                py::reinterpret_borrow<py::object>(top_level));
        }

        if (py::len(this->inputs()) != 0) {
            if (lower_is_tensor_ && upper_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, lower, upper));
            } else if (lower_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, lower));
            } else if (upper_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, upper));
            } else {
                this->store_inputs(py::make_tuple(tensor));
            }
        }

        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView lower_view =
            lower_is_tensor_ ? cpu_tensor_view(lower, "lower") : tensor_view;
        CpuTensorView upper_view =
            upper_is_tensor_ ? cpu_tensor_view(upper, "upper") : tensor_view;
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < result.view.logical_size; ++i) {
                const float value = tensor_view.read_float_expanded(key);
                const float lower_value = lower_is_tensor_
                                              ? lower_view.read_float_expanded(key)
                                              : lower_scalar_;
                const float upper_value = upper_is_tensor_
                                              ? upper_view.read_float_expanded(key)
                                              : upper_scalar_;
                const float middle = cpu_extrema_value<true>(value, lower_value);
                result.view.write_float_expanded(
                    key, cpu_extrema_value<false>(middle, upper_value));
                result.view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        std::size_t cursor = 1;
        py::object lower = py::none();
        py::object upper = py::none();
        if (lower_is_tensor_) {
            lower = py::reinterpret_borrow<py::object>(input_tensors[cursor++]);
        }
        if (upper_is_tensor_) {
            upper = py::reinterpret_borrow<py::object>(input_tensors[cursor]);
        }
        require_layout(gradient, output_layout_);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView lower_view =
            lower_is_tensor_ ? cpu_tensor_view(lower, "lower") : tensor_view;
        CpuTensorView upper_view =
            upper_is_tensor_ ? cpu_tensor_view(upper, "upper") : tensor_view;
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation tensor_result = allocate_gradient_for(tensor);
        std::optional<CpuTensorAllocation> lower_result;
        std::optional<CpuTensorAllocation> upper_result;
        if (lower_is_tensor_) {
            lower_result.emplace(allocate_gradient_for(lower));
        }
        if (upper_is_tensor_) {
            upper_result.emplace(allocate_gradient_for(upper));
        }
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                const float value = tensor_view.read_float_expanded(key);
                const float lower_value = lower_is_tensor_
                                              ? lower_view.read_float_expanded(key)
                                              : lower_scalar_;
                const float upper_value = upper_is_tensor_
                                              ? upper_view.read_float_expanded(key)
                                              : upper_scalar_;
                const float middle = cpu_extrema_value<true>(value, lower_value);
                float middle_gradient = 0.0f;
                float upper_gradient = 0.0f;
                cpu_extrema_vjp<false>(middle, upper_value,
                                       gradient_view.read_float_expanded(key),
                                       middle_gradient, upper_gradient);
                float tensor_gradient = 0.0f;
                float lower_gradient = 0.0f;
                cpu_extrema_vjp<true>(value, lower_value, middle_gradient,
                                      tensor_gradient, lower_gradient);
                tensor_result.view.write_float_expanded(key, tensor_gradient);
                if (lower_result.has_value()) {
                    lower_result->view.write_float_expanded(key, lower_gradient);
                }
                if (upper_result.has_value()) {
                    upper_result->view.write_float_expanded(key, upper_gradient);
                }
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }

        py::tuple gradients(1 + static_cast<std::size_t>(lower_is_tensor_) +
                            static_cast<std::size_t>(upper_is_tensor_));
        std::size_t output = 0;
        gradients[output++] = make_tensor(std::move(tensor_result.carrier_object),
                                          std::move(tensor_result.layout_object));
        if (lower_result.has_value()) {
            gradients[output++] = make_tensor(std::move(lower_result->carrier_object),
                                              std::move(lower_result->layout_object));
        }
        if (upper_result.has_value()) {
            gradients[output] = make_tensor(std::move(upper_result->carrier_object),
                                            std::move(upper_result->layout_object));
        }
        return gradients;
    }

private:
    bool lower_is_tensor_ = false;
    bool upper_is_tensor_ = false;
    float lower_scalar_ = 0.0f;
    float upper_scalar_ = 0.0f;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kMetadata{"clamp", "cpu.clamp", "default",
                                      "_CPUClampOperation"};

}  // namespace

void register_cpu_clamp(py::module_& module) {
    bind_and_register_cpu_operation<CpuClampOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
