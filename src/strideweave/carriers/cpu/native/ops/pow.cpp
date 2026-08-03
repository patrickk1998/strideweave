#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuPowOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU power requires a tensor and exponent");
        }
        py::object first = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object second = py::reinterpret_borrow<py::object>(inputs[1]);
        if (py::isinstance(first, tensor_type()) &&
            py::isinstance(second, tensor_type())) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(first), std::move(second));
            first = std::move(aligned.lhs);
            second = std::move(aligned.rhs);
            if (py::len(this->inputs()) != 0) {
                this->store_inputs(py::make_tuple(first, second));
            }
            CpuTensorView base = cpu_tensor_view(first, "base");
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            const CpuPlan plan = resolve_binary_plan(first, "pow", base, exponent);
            CpuTensorAllocation result =
                allocate_cpu_tensor(std::move(aligned.result_layout), plan.output);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(result.view.leaf_rank(), 0);
                for (Index i = 0; i < result.view.logical_size; ++i) {
                    result.view.write_float_expanded(
                        key, std::pow(base.read_float_expanded(key),
                                      exponent.read_float_expanded(key)));
                    result.view.cache->increment_key(key.data(), key.size());
                }
            }
            return make_tensor(std::move(result.carrier_object),
                               std::move(result.layout_object));
        }
        if (!py::isinstance(first, tensor_type()) &&
            py::isinstance(second, tensor_type())) {
            scalar_base_ = true;
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            const CpuPlan plan =
                resolve_cpu_plan(executing_carrier_class(second), "pow", first,
                                 cpu_dtype_object(exponent.carrier->cpu_dtype()));
            base_ = require_float(first, "base");
            ctx_["base"] = py::float_(base_);
            CpuTensorAllocation result =
                allocate_cpu_tensor(injective_layout_for(second), plan.output);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(exponent.leaf_rank(), 0);
                for (Index i = 0; i < exponent.logical_size; ++i) {
                    result.view.write_float_expanded(
                        key, std::pow(base_, exponent.read_float_expanded(key)));
                    exponent.cache->increment_key(key.data(), key.size());
                }
            }
            return make_tensor(std::move(result.carrier_object),
                               std::move(result.layout_object));
        }

        // Tensor raised to a weak scalar.  Whether an Int32 result is
        // preserved is central policy, so the exponent is materialized exactly
        // for the integer path and only once as binary32 otherwise.
        if (!py::isinstance(first, tensor_type())) {
            throw py::type_error("CPU power requires a tensor and exponent");
        }
        CpuTensorView tensor_view = cpu_tensor_view(first, "tensor");
        const CpuPlan plan = resolve_cpu_plan(
            executing_carrier_class(first), "pow",
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()), second);
        exponent_ = require_float(second, "exponent");
        ctx_["exponent"] = py::float_(exponent_);
        const std::int32_t int_exponent =
            plan.is_integer() ? require_int32_scalar(second, "exponent") : 0;

        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(first), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (plan.is_integer()) {
                    write_computed_int(
                        result.view, key,
                        checked_int32_pow(tensor_view.read_int_expanded(key),
                                          int_exponent),
                        plan.compute);
                } else {
                    result.view.write_float_expanded(
                        key, std::pow(tensor_view.read_float_expanded(key), exponent_));
                }
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object first = py::reinterpret_borrow<py::object>(input_tensors[0]);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        if (py::len(input_tensors) == 2) {
            py::object second = py::reinterpret_borrow<py::object>(input_tensors[1]);
            require_same_shape(first, gradient);
            require_same_shape(second, gradient);
            CpuTensorView base = cpu_tensor_view(first, "base");
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            CpuTensorAllocation base_result = allocate_gradient_for(first);
            CpuTensorAllocation exponent_result = allocate_gradient_for(second);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(base.leaf_rank(), 0);
                for (Index i = 0; i < base.logical_size; ++i) {
                    const float x = base.read_float_expanded(key);
                    const float p = exponent.read_float_expanded(key);
                    const float y = std::pow(x, p);
                    const float g = gradient_view.read_float_expanded(key);
                    base_result.view.write_float_expanded(
                        key, g * p * std::pow(x, p - 1.0f));
                    exponent_result.view.write_float_expanded(key, g * y * std::log(x));
                    base.cache->increment_key(key.data(), key.size());
                }
            }
            return py::make_tuple(
                make_tensor(std::move(base_result.carrier_object),
                            std::move(base_result.layout_object)),
                make_tensor(std::move(exponent_result.carrier_object),
                            std::move(exponent_result.layout_object)));
        }
        // In scalar-base form only the tensor exponent receives a gradient.
        if (py::len(input_tensors) == 1 && scalar_base_) {
            py::object exponent = py::reinterpret_borrow<py::object>(input_tensors[0]);
            require_same_shape(exponent, gradient);
            CpuTensorView exponent_view = cpu_tensor_view(exponent, "exponent");
            CpuTensorAllocation result = allocate_gradient_for(exponent);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(exponent_view.leaf_rank(), 0);
                for (Index i = 0; i < exponent_view.logical_size; ++i) {
                    const float y =
                        std::pow(base_, exponent_view.read_float_expanded(key));
                    result.view.write_float_expanded(
                        key,
                        gradient_view.read_float_expanded(key) * y * std::log(base_));
                    exponent_view.cache->increment_key(key.data(), key.size());
                }
            }
            return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                              std::move(result.layout_object)));
        }
        py::object tensor = first;
        require_same_shape(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorAllocation result = allocate_gradient_for(tensor);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key, gradient_view.read_float_expanded(key) * exponent_ *
                             std::pow(tensor_view.read_float_expanded(key),
                                      exponent_ - 1.0f));
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    float exponent_ = 0.0f;
    float base_ = 0.0f;
    bool scalar_base_ = false;
};

constexpr CpuKernelMetadata kMetadata{"pow", "cpu.pow", "default", "_CPUPowOperation"};

}  // namespace

void register_cpu_pow(py::module_& module) {
    bind_and_register_cpu_operation<CpuPowOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
