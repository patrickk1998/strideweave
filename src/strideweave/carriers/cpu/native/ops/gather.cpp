#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuGatherOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 3) {
            throw py::type_error("CPU gather requires tensor, indices, and axis");
        }
        py::object data = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object indices = py::reinterpret_borrow<py::object>(inputs[1]);
        data_modes_ = cpu_layout_modes(data, "gather");
        index_modes_ = cpu_layout_modes(indices, "gather indices");
        axis_ = normalize_axis(py::cast<Index>(inputs[2]),
                               static_cast<Index>(data_modes_.modes.size()));
        CpuTensorView data_view = cpu_tensor_view(data, "data");
        CpuTensorView index_view = cpu_tensor_view(indices, "indices");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(data), "gather",
                             cpu_dtype_object(data_view.carrier->cpu_dtype()),
                             cpu_dtype_object(index_view.carrier->cpu_dtype()));
        const Index axis_extent =
            data_modes_.modes[static_cast<std::size_t>(axis_)].logical_size();
        if (axis_extent <= 0) {
            throw py::value_error("gather axis extent must be positive");
        }
        validate_indices(index_view, axis_extent);
        py::list output_top;
        for (Index mode = 0; mode < axis_; ++mode) {
            output_top.append(data_modes_.modes[static_cast<std::size_t>(mode)].shape);
        }
        for (const CpuModeInfo& mode : index_modes_.modes) {
            output_top.append(mode.shape);
        }
        for (Index mode = axis_ + 1;
             mode < static_cast<Index>(data_modes_.modes.size()); ++mode) {
            output_top.append(data_modes_.modes[static_cast<std::size_t>(mode)].shape);
        }
        output_layout_ = canonical_layout_from_top_level(std::move(output_top));
        if (py::len(this->inputs()) != 0) {
            this->store_inputs(py::make_tuple(data, indices));
        }
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        CpuTensorView result_view = result.view;
        {
            py::gil_scoped_release release;
            const std::vector<Index> data_offsets = mode_leaf_offsets(data_modes_);
            const Index axis_leaf_offset =
                data_offsets[static_cast<std::size_t>(axis_)];
            Index index_leaf_count = 0;
            for (Index count : index_modes_.leaf_counts) {
                index_leaf_count += count;
            }
            std::vector<Index> output_key(result_view.leaf_rank(), 0);
            for (Index i = 0; i < result_view.logical_size; ++i) {
                const std::vector<Index> outer(output_key.begin(),
                                               output_key.begin() + axis_leaf_offset);
                const std::vector<Index> index_key(
                    output_key.begin() + axis_leaf_offset,
                    output_key.begin() + axis_leaf_offset + index_leaf_count);
                const Index index = index_view.read_int_expanded(index_key);
                std::vector<Index> suffix(output_key.begin() + axis_leaf_offset +
                                              index_leaf_count,
                                          output_key.end());
                std::vector<Index> data_key = outer;
                const std::vector<Index> axis_key = decode_ordinal(
                    index,
                    data_modes_.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                data_key.insert(data_key.end(), axis_key.begin(), axis_key.end());
                data_key.insert(data_key.end(), suffix.begin(), suffix.end());
                result_view.write_float_expanded(
                    output_key, data_view.read_float_expanded(data_key));
                result_view.cache->increment_key(output_key.data(), output_key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object data = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object indices = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_layout(gradient, output_layout_);
        CpuTensorView index_view = cpu_tensor_view(indices, "indices");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation result = allocate_gradient_for(data);
        std::vector<Index> output_key(gradient_view.leaf_rank(), 0);
        std::vector<Index> zero_key(result.view.leaf_rank(), 0);
        {
            py::gil_scoped_release release;
            for (Index i = 0; i < result.view.logical_size; ++i) {
                result.view.write_float_expanded(zero_key, 0.0f);
                result.view.cache->increment_key(zero_key.data(), zero_key.size());
            }
            const std::vector<Index> offsets = mode_leaf_offsets(data_modes_);
            const Index axis_offset = offsets[static_cast<std::size_t>(axis_)];
            Index index_count = 0;
            for (Index count : index_modes_.leaf_counts) {
                index_count += count;
            }
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                const std::vector<Index> outer(output_key.begin(),
                                               output_key.begin() + axis_offset);
                const std::vector<Index> index_key(output_key.begin() + axis_offset,
                                                   output_key.begin() + axis_offset +
                                                       index_count);
                const Index selected = index_view.read_int_expanded(index_key);
                std::vector<Index> data_key = outer;
                const std::vector<Index> axis_key = decode_ordinal(
                    selected,
                    data_modes_.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                data_key.insert(data_key.end(), axis_key.begin(), axis_key.end());
                const std::vector<Index> suffix(
                    output_key.begin() + axis_offset + index_count, output_key.end());
                data_key.insert(data_key.end(), suffix.begin(), suffix.end());
                result.view.write_float_expanded(
                    data_key, result.view.read_float_expanded(data_key) +
                                  gradient_view.read_float_expanded(output_key));
                gradient_view.cache->increment_key(output_key.data(),
                                                   output_key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)),
                              py::none());
    }

private:
    static void validate_indices(const CpuTensorView& indices, Index extent) {
        std::vector<Index> key(indices.leaf_rank(), 0);
        for (Index i = 0; i < indices.logical_size; ++i) {
            const Index value = indices.read_int_expanded(key);
            if (value < 0 || value >= extent) {
                throw py::value_error("gather index is out of range");
            }
            indices.cache->increment_key(key.data(), key.size());
        }
    }

    CpuLayoutModes data_modes_;
    CpuLayoutModes index_modes_;
    Index axis_ = 0;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kMetadata{"gather", "cpu.gather", "default",
                                      "_CPUGatherOperation"};

}  // namespace

void register_cpu_gather(py::module_& module) {
    bind_and_register_cpu_operation<CpuGatherOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
