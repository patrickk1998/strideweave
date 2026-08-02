#include <cmath>
#include <cstdint>
#include <vector>

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
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        const CpuPlan plan = resolve_cpu_plan(
            executing_carrier_class(tensor), "pow",
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()), inputs[1]);
        exponent_ = require_float(inputs[1], "exponent");
        ctx_["exponent"] = py::float_(exponent_);
        const std::int32_t int_exponent =
            plan.is_integer() ? require_int32_scalar(inputs[1], "exponent") : 0;

        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(tensor), plan.output);
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
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_shape(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

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
};

constexpr CpuKernelMetadata kMetadata{"pow", "cpu.pow", "default", "_CPUPowOperation"};

}  // namespace

void register_cpu_pow(py::module_& module) {
    bind_and_register_cpu_operation<CpuPowOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
