#include <cstdint>
#include <vector>

#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuScalarMulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU scalar multiply requires a tensor and scalar");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        const CpuPlan plan = resolve_cpu_plan(
            executing_carrier_class(tensor), "mul",
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()), inputs[1]);
        scalar_ = require_float(inputs[1], "scalar");
        ctx_["scalar"] = py::float_(scalar_);
        const std::int32_t int_scalar =
            plan.is_integer() ? require_int32_scalar(inputs[1], "scalar") : 0;

        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(tensor), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                if (plan.is_integer()) {
                    write_computed_int(
                        result.view, key,
                        static_cast<long long>(tensor_view.read_int_expanded(key)) *
                            static_cast<long long>(int_scalar),
                        plan.compute);
                } else {
                    result.view.write_float_expanded(
                        key, tensor_view.read_float_expanded(key) * scalar_);
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
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        CpuTensorAllocation result = allocate_gradient_for(tensor);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                result.view.write_float_expanded(
                    key, gradient_view.read_float_expanded(key) * scalar_);
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    float scalar_ = 0.0f;
};

constexpr CpuKernelMetadata kMetadata{"mul", "cpu.scalar_mul", "default",
                                      "_CPUScalarMulOperation"};

}  // namespace

void register_cpu_scalar_mul(py::module_& module) {
    bind_and_register_cpu_operation<CpuScalarMulOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
