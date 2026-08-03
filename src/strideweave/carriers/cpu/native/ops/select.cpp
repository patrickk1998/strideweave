#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuSelectOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 3) {
            throw py::type_error(
                "CPU select requires condition, on_true, and on_false tensors");
        }
        py::object condition = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object on_true = py::reinterpret_borrow<py::object>(inputs[1]);
        py::object on_false = py::reinterpret_borrow<py::object>(inputs[2]);
        require_same_carrier_class(condition, on_true,
                                   "Tensor backing carriers must match");
        require_same_carrier_class(condition, on_false,
                                   "Tensor backing carriers must match");

        CpuTensorView original_condition = cpu_tensor_view(condition, "condition");
        CpuTensorView original_true = cpu_tensor_view(on_true, "on_true");
        CpuTensorView original_false = cpu_tensor_view(on_false, "on_false");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(condition), "select",
                             cpu_dtype_object(original_condition.carrier->cpu_dtype()),
                             cpu_dtype_object(original_true.carrier->cpu_dtype()),
                             cpu_dtype_object(original_false.carrier->cpu_dtype()));

        AlignedTernaryOperands aligned = align_ternary_operands(
            std::move(condition), std::move(on_true), std::move(on_false));
        condition = std::move(aligned.first);
        on_true = std::move(aligned.second);
        on_false = std::move(aligned.third);
        output_layout_ = std::move(aligned.result_layout);
        if (py::len(this->inputs()) != 0) {
            this->store_inputs(py::make_tuple(condition, on_true, on_false));
        }

        CpuTensorView condition_view = cpu_tensor_view(condition, "condition");
        CpuTensorView true_view = cpu_tensor_view(on_true, "on_true");
        CpuTensorView false_view = cpu_tensor_view(on_false, "on_false");
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < result.view.logical_size; ++i) {
                const bool choose_true = condition_view.read_bool_expanded(key);
                // Read exactly one value branch.  In particular an unselected
                // NaN never participates in this operation.
                const float selected = choose_true
                                           ? true_view.read_float_expanded(key)
                                           : false_view.read_float_expanded(key);
                result.view.write_float_expanded(key, selected);
                result.view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object condition = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object on_true = py::reinterpret_borrow<py::object>(input_tensors[1]);
        py::object on_false = py::reinterpret_borrow<py::object>(input_tensors[2]);
        require_layout(gradient, output_layout_);
        CpuTensorView condition_view = cpu_tensor_view(condition, "condition");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation true_result = allocate_gradient_for(on_true);
        CpuTensorAllocation false_result = allocate_gradient_for(on_false);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                const float value = gradient_view.read_float_expanded(key);
                if (condition_view.read_bool_expanded(key)) {
                    true_result.view.write_float_expanded(key, value);
                    false_result.view.write_float_expanded(key, 0.0f);
                } else {
                    true_result.view.write_float_expanded(key, 0.0f);
                    false_result.view.write_float_expanded(key, value);
                }
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(py::none(),
                              make_tensor(std::move(true_result.carrier_object),
                                          std::move(true_result.layout_object)),
                              make_tensor(std::move(false_result.carrier_object),
                                          std::move(false_result.layout_object)));
    }

private:
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kMetadata{"select", "cpu.select", "default",
                                      "_CPUSelectOperation"};

}  // namespace

void register_cpu_select(py::module_& module) {
    bind_and_register_cpu_operation<CpuSelectOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
