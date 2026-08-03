#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuCumsumOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU cumsum requires a tensor and dimension");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        if (py::isinstance<py::bool_>(inputs[1])) {
            throw py::type_error("cumsum dimension must be an integer top-level mode");
        }
        const CpuLayoutModes modes = cpu_layout_modes(tensor, "cumsum");
        if (modes.modes.empty()) {
            throw py::value_error("cumsum requires nonempty tensor modes");
        }
        const Index axis = normalize_axis(py::cast<Index>(inputs[1]),
                                          static_cast<Index>(modes.modes.size()));
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), "cumsum",
                             cpu_dtype_object(view.carrier->cpu_dtype()));
        py::object output_top = tensor_layout(tensor).attr("shape").attr("top_level");
        output_layout_ = canonical_layout_from_top_level(
            py::reinterpret_borrow<py::object>(output_top));
        ctx_["output_layout"] = output_layout_;
        axis_ = axis;
        modes_ = modes;
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        const Index outer_size = outer_product_without_axis(modes, axis);
        const std::vector<Index> offsets = mode_leaf_offsets(modes);
        const Index axis_offset = offsets[static_cast<std::size_t>(axis)];
        const Index axis_extent =
            modes.modes[static_cast<std::size_t>(axis)].logical_size();
        if (axis_extent <= 0) {
            throw py::value_error("cumsum requires nonempty tensor modes");
        }
        {
            py::gil_scoped_release release;
            for (Index outer = 0; outer < outer_size; ++outer) {
                const std::vector<Index> outer_key =
                    outer_key_for_ordinal(outer, modes, axis);
                float sum = 0.0f;
                for (Index position = 0; position < axis_extent; ++position) {
                    std::vector<Index> key = outer_key;
                    const std::vector<Index> axis_key = decode_ordinal(
                        position,
                        modes.modes[static_cast<std::size_t>(axis)].leaf_extents);
                    key.insert(key.begin() + axis_offset, axis_key.begin(),
                               axis_key.end());
                    sum += view.read_float_expanded(key);
                    result.view.write_float_expanded(key, sum);
                }
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView grad = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation result = allocate_gradient_for(tensor);
        const Index outer_size = outer_product_without_axis(modes_, axis_);
        const std::vector<Index> offsets = mode_leaf_offsets(modes_);
        const Index axis_offset = offsets[static_cast<std::size_t>(axis_)];
        const Index axis_extent =
            modes_.modes[static_cast<std::size_t>(axis_)].logical_size();
        {
            py::gil_scoped_release release;
            for (Index outer = 0; outer < outer_size; ++outer) {
                const std::vector<Index> outer_key =
                    outer_key_for_ordinal(outer, modes_, axis_);
                float sum = 0.0f;
                for (Index position = axis_extent; position-- > 0;) {
                    std::vector<Index> key = outer_key;
                    const std::vector<Index> axis_key = decode_ordinal(
                        position,
                        modes_.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                    key.insert(key.begin() + axis_offset, axis_key.begin(),
                               axis_key.end());
                    sum += grad.read_float_expanded(key);
                    result.view.write_float_expanded(key, sum);
                }
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    static Index outer_product_without_axis(const CpuLayoutModes& modes, Index axis) {
        Index result = 1;
        for (Index mode = 0; mode < static_cast<Index>(modes.modes.size()); ++mode) {
            if (mode != axis) {
                result *= modes.modes[static_cast<std::size_t>(mode)].logical_size();
            }
        }
        return result;
    }

    CpuLayoutModes modes_;
    Index axis_ = 0;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kMetadata{"cumsum", "cpu.cumsum", "default",
                                      "_CPUCumsumOperation"};

}  // namespace

void register_cpu_cumsum(py::module_& module) {
    bind_and_register_cpu_operation<CpuCumsumOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
