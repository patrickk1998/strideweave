#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuReLUOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU ReLU requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");

        // ReLU preserves its input dtype, and selecting between an element and
        // zero cannot overflow, so the integer plan is exact and unchecked.
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), "relu",
                             cpu_dtype_object(tensor_view.carrier->cpu_dtype()));
        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(tensor), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (plan.is_integer()) {
                    const std::int32_t value = tensor_view.read_int_expanded(key);
                    write_computed_int(result.view, key, value > 0 ? value : 0,
                                       plan.compute);
                } else {
                    const float value = tensor_view.read_float_expanded(key);
                    result.view.write_float_expanded(key, value > 0.0f ? value : 0.0f);
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
                const float value = tensor_view.read_float_expanded(key);
                result.view.write_float_expanded(
                    key, value > 0.0f ? gradient_view.read_float_expanded(key) : 0.0f);
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }
};

constexpr CpuKernelMetadata kMetadata{"relu", "cpu.relu", "default",
                                      "_CPUReLUOperation"};

}  // namespace

void register_cpu_relu(py::module_& module) {
    bind_and_register_cpu_operation<CpuReLUOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
