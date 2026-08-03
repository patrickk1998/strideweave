#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuDivOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU division requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        AlignedBinaryOperands aligned =
            align_binary_operands(std::move(lhs), std::move(rhs));
        lhs = std::move(aligned.lhs);
        rhs = std::move(aligned.rhs);
        if (py::len(this->inputs()) != 0) {
            py::tuple aligned_inputs = py::make_tuple(lhs, rhs);
            this->store_inputs(py::reinterpret_borrow<py::args>(aligned_inputs));
        }
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");

        // Division has no integer path: the plan converts two Int32 operands to
        // Float32 rather than truncating.
        const CpuPlan plan = resolve_binary_plan(lhs, "div", lhs_view, rhs_view);
        CpuTensorAllocation result =
            allocate_cpu_tensor(std::move(aligned.result_layout), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < result.view.logical_size; ++i) {
                result.view.write_float_expanded(key,
                                                 lhs_view.read_float_expanded(key) /
                                                     rhs_view.read_float_expanded(key));
                result.view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
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
                const float lhs_value = lhs_view.read_float_expanded(key);
                const float rhs_value = rhs_view.read_float_expanded(key);
                const float gradient_value = gradient_view.read_float_expanded(key);
                lhs_result.view.write_float_expanded(key, gradient_value / rhs_value);
                rhs_result.view.write_float_expanded(key, -gradient_value * lhs_value /
                                                              (rhs_value * rhs_value));
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }

        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }
};

constexpr CpuKernelMetadata kMetadata{"div", "cpu.div", "default", "_CPUDivOperation"};

}  // namespace

void register_cpu_div(py::module_& module) {
    bind_and_register_cpu_operation<CpuDivOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
