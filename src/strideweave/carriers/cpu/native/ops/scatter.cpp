#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuScatterOperation : public strideweave::operation::Operation {
public:
    explicit CpuScatterOperation(bool add) : add_(add) {}

    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 4) {
            throw py::type_error(
                "CPU scatter requires base, indices, updates, and axis");
        }
        py::object base = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object indices = py::reinterpret_borrow<py::object>(inputs[1]);
        py::object updates = py::reinterpret_borrow<py::object>(inputs[2]);
        data_modes_ = cpu_layout_modes(base, "scatter");
        index_modes_ = cpu_layout_modes(indices, "scatter indices");
        CpuLayoutModes update_modes = cpu_layout_modes(updates, "scatter updates");
        axis_ = normalize_axis(py::cast<Index>(inputs[3]),
                               static_cast<Index>(data_modes_.modes.size()));
        py::list expected_top;
        for (Index mode = 0; mode < axis_; ++mode) {
            expected_top.append(
                data_modes_.modes[static_cast<std::size_t>(mode)].shape);
        }
        for (const CpuModeInfo& mode : index_modes_.modes) {
            expected_top.append(mode.shape);
        }
        for (Index mode = axis_ + 1;
             mode < static_cast<Index>(data_modes_.modes.size()); ++mode) {
            expected_top.append(
                data_modes_.modes[static_cast<std::size_t>(mode)].shape);
        }
        py::object expected_shape = py::module_::import("strideweave.layout")
                                        .attr("Shape")(std::move(expected_top));
        if (!layouts_equal(expected_shape, tensor_layout(updates).attr("shape"))) {
            throw py::value_error(
                "scatter updates shape must match gather result shape");
        }
        CpuTensorView base_view = cpu_tensor_view(base, "base");
        CpuTensorView index_view = cpu_tensor_view(indices, "indices");
        CpuTensorView update_view = cpu_tensor_view(updates, "updates");
        const CpuPlan plan = resolve_cpu_plan(
            executing_carrier_class(base), add_ ? "scatter_add" : "scatter",
            cpu_dtype_object(base_view.carrier->cpu_dtype()),
            cpu_dtype_object(index_view.carrier->cpu_dtype()),
            cpu_dtype_object(update_view.carrier->cpu_dtype()));
        const Index extent =
            data_modes_.modes[static_cast<std::size_t>(axis_)].logical_size();
        validate_indices(index_view, extent);
        if (!add_) {
            std::unordered_set<Index> seen;
            std::vector<Index> key(index_view.leaf_rank(), 0);
            for (Index i = 0; i < index_view.logical_size; ++i) {
                if (!seen.insert(index_view.read_int_expanded(key)).second) {
                    throw py::value_error("scatter indices must be distinct");
                }
                index_view.cache->increment_key(key.data(), key.size());
            }
        }
        py::object data_top = tensor_layout(base).attr("shape").attr("top_level");
        output_layout_ = canonical_layout_from_top_level(
            py::reinterpret_borrow<py::object>(data_top));
        if (py::len(this->inputs()) != 0) {
            this->store_inputs(py::make_tuple(base, indices, updates));
        }
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < result.view.logical_size; ++i) {
                result.view.write_float_expanded(key,
                                                 base_view.read_float_expanded(key));
                result.view.cache->increment_key(key.data(), key.size());
            }
            const std::vector<Index> data_offsets = mode_leaf_offsets(data_modes_);
            const Index axis_leaf_offset =
                data_offsets[static_cast<std::size_t>(axis_)];
            Index index_leaf_count = 0;
            for (Index count : index_modes_.leaf_counts) {
                index_leaf_count += count;
            }
            std::vector<Index> update_key(update_view.leaf_rank(), 0);
            for (Index i = 0; i < update_view.logical_size; ++i) {
                const std::vector<Index> outer(update_key.begin(),
                                               update_key.begin() + axis_leaf_offset);
                const std::vector<Index> index_key(
                    update_key.begin() + axis_leaf_offset,
                    update_key.begin() + axis_leaf_offset + index_leaf_count);
                const Index selected = index_view.read_int_expanded(index_key);
                std::vector<Index> suffix(update_key.begin() + axis_leaf_offset +
                                              index_leaf_count,
                                          update_key.end());
                std::vector<Index> destination = outer;
                const std::vector<Index> axis_key = decode_ordinal(
                    selected,
                    data_modes_.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                destination.insert(destination.end(), axis_key.begin(), axis_key.end());
                destination.insert(destination.end(), suffix.begin(), suffix.end());
                const float value = update_view.read_float_expanded(update_key);
                if (add_) {
                    result.view.write_float_expanded(
                        destination,
                        result.view.read_float_expanded(destination) + value);
                } else {
                    result.view.write_float_expanded(destination, value);
                }
                update_view.cache->increment_key(update_key.data(), update_key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        if (py::len(input_tensors) != 3) {
            return py::make_tuple();
        }
        py::object base = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object indices = py::reinterpret_borrow<py::object>(input_tensors[1]);
        py::object updates = py::reinterpret_borrow<py::object>(input_tensors[2]);
        require_layout(gradient, output_layout_);
        CpuTensorView index_view = cpu_tensor_view(indices, "indices");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorView updates_view = cpu_tensor_view(updates, "updates");
        CpuTensorAllocation base_result = allocate_gradient_for(base);
        CpuTensorAllocation updates_result = allocate_gradient_for(updates);
        const std::vector<Index> data_offsets = mode_leaf_offsets(data_modes_);
        const Index axis_leaf_offset = data_offsets[static_cast<std::size_t>(axis_)];
        Index index_leaf_count = 0;
        for (Index count : index_modes_.leaf_counts) {
            index_leaf_count += count;
        }
        {
            py::gil_scoped_release release;
            // The base cotangent is the output cotangent everywhere, except at
            // destinations overwritten by ``scatter``.  Scatter-add keeps the
            // output cotangent unchanged at all base positions.
            std::vector<Index> key(base_result.view.leaf_rank(), 0);
            for (Index i = 0; i < base_result.view.logical_size; ++i) {
                base_result.view.write_float_expanded(
                    key, gradient_view.read_float_expanded(key));
                base_result.view.cache->increment_key(key.data(), key.size());
            }
            if (!add_) {
                std::vector<Index> update_key(updates_view.leaf_rank(), 0);
                for (Index i = 0; i < updates_view.logical_size; ++i) {
                    const std::vector<Index> outer(
                        update_key.begin(), update_key.begin() + axis_leaf_offset);
                    const std::vector<Index> index_key(
                        update_key.begin() + axis_leaf_offset,
                        update_key.begin() + axis_leaf_offset + index_leaf_count);
                    const Index selected = index_view.read_int_expanded(index_key);
                    std::vector<Index> destination = outer;
                    const std::vector<Index> axis_key = decode_ordinal(
                        selected, data_modes_.modes[static_cast<std::size_t>(axis_)]
                                      .leaf_extents);
                    destination.insert(destination.end(), axis_key.begin(),
                                       axis_key.end());
                    destination.insert(destination.end(),
                                       update_key.begin() + axis_leaf_offset +
                                           index_leaf_count,
                                       update_key.end());
                    base_result.view.write_float_expanded(destination, 0.0f);
                    updates_view.cache->increment_key(update_key.data(),
                                                      update_key.size());
                }
            }

            // Every update is a direct read from the output cotangent at its
            // destination.  This naturally handles repeated indices for
            // scatter_add and preserves hierarchical axis coordinates.
            std::vector<Index> update_key(updates_view.leaf_rank(), 0);
            for (Index i = 0; i < updates_view.logical_size; ++i) {
                const std::vector<Index> outer(update_key.begin(),
                                               update_key.begin() + axis_leaf_offset);
                const std::vector<Index> index_key(
                    update_key.begin() + axis_leaf_offset,
                    update_key.begin() + axis_leaf_offset + index_leaf_count);
                const Index selected = index_view.read_int_expanded(index_key);
                std::vector<Index> destination = outer;
                const std::vector<Index> axis_key = decode_ordinal(
                    selected,
                    data_modes_.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                destination.insert(destination.end(), axis_key.begin(), axis_key.end());
                destination.insert(destination.end(),
                                   update_key.begin() + axis_leaf_offset +
                                       index_leaf_count,
                                   update_key.end());
                updates_result.view.write_float_expanded(
                    update_key, gradient_view.read_float_expanded(destination));
                updates_view.cache->increment_key(update_key.data(), update_key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(base_result.carrier_object),
                                          std::move(base_result.layout_object)),
                              py::none(),
                              make_tensor(std::move(updates_result.carrier_object),
                                          std::move(updates_result.layout_object)));
    }

private:
    static void validate_indices(const CpuTensorView& indices, Index extent) {
        std::vector<Index> key(indices.leaf_rank(), 0);
        for (Index i = 0; i < indices.logical_size; ++i) {
            const Index value = indices.read_int_expanded(key);
            if (value < 0 || value >= extent) {
                throw py::value_error("scatter index is out of range");
            }
            indices.cache->increment_key(key.data(), key.size());
        }
    }

    bool add_;
    CpuLayoutModes data_modes_;
    CpuLayoutModes index_modes_;
    Index axis_ = 0;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kScatterMetadata{"scatter", "cpu.scatter", "default",
                                             "_CPUScatterOperation"};
constexpr CpuKernelMetadata kScatterAddMetadata{"scatter_add", "cpu.scatter_add",
                                                "default", "_CPUScatterOperation"};

}  // namespace

void register_cpu_scatter(py::module_& module) {
    py::class_<CpuScatterOperation, strideweave::operation::Operation>(
        module, "_CPUScatterOperation")
        .def(py::init<bool>());
    register_cpu_native_operation(kScatterMetadata, [] {
        return py::cast(new CpuScatterOperation(false),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_native_operation(kScatterAddMetadata, [] {
        return py::cast(new CpuScatterOperation(true),
                        py::return_value_policy::take_ownership);
    });
}

}  // namespace strideweave::carrier
