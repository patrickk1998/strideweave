#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuSortOperation : public strideweave::operation::Operation {
public:
    CpuSortOperation(bool values, bool topk) : values_(values), topk_(topk) {}

    py::object _forward(py::args inputs) override {
        if ((topk_ && (py::len(inputs) < 2 || py::len(inputs) > 4)) ||
            (!topk_ && (py::len(inputs) < 1 || py::len(inputs) > 3))) {
            throw py::type_error(topk_ ? "CPU topk requires tensor and k"
                                       : "CPU sort requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuLayoutModes modes = cpu_layout_modes(tensor, topk_ ? "topk" : "sort");
        modes_ = modes;
        axis_ =
            modes.modes.empty()
                ? 0
                : normalize_axis(
                      topk_ ? (py::len(inputs) >= 3 ? py::cast<Index>(inputs[2]) : -1)
                            : (py::len(inputs) >= 2 ? py::cast<Index>(inputs[1]) : -1),
                      static_cast<Index>(modes.modes.size()));
        largest_ =
            topk_ ? (py::len(inputs) >= 4 ? require_bool(inputs[3], "largest") : true)
                  : (py::len(inputs) >= 3 ? require_bool(inputs[2], "descending")
                                          : false);
        k_ = topk_ ? py::cast<Index>(inputs[1])
                   : modes.modes[static_cast<std::size_t>(axis_)].logical_size();
        const Index axis_extent =
            modes.modes[static_cast<std::size_t>(axis_)].logical_size();
        if (axis_extent <= 0 || k_ <= 0 || k_ > axis_extent || k_ > INT32_MAX) {
            throw py::value_error(
                "sort axis extent and topk k must be positive and fit Int32");
        }
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        const char* operation = values_ ? (topk_ ? "_topk_values" : "_sort_values")
                                        : (topk_ ? "_topk_indices" : "_sort_indices");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), operation,
                             cpu_dtype_object(view.carrier->cpu_dtype()));
        py::list output_top;
        for (Index mode = 0; mode < static_cast<Index>(modes.modes.size()); ++mode) {
            if (mode == axis_ && topk_) {
                output_top.append(py::int_(k_));
            } else {
                output_top.append(modes.modes[static_cast<std::size_t>(mode)].shape);
            }
        }
        output_layout_ = canonical_layout_from_top_level(std::move(output_top));
        output_leaf_extents_.clear();
        for (Index mode = 0; mode < static_cast<Index>(modes.modes.size()); ++mode) {
            if (mode == axis_ && topk_) {
                output_leaf_extents_.push_back(k_);
            } else {
                const auto& leaves =
                    modes.modes[static_cast<std::size_t>(mode)].leaf_extents;
                output_leaf_extents_.insert(output_leaf_extents_.end(), leaves.begin(),
                                            leaves.end());
            }
        }
        if (values_) {
            source_keys_.assign(
                static_cast<std::size_t>(product_extents(output_leaf_extents_)), {});
        }
        if (values_ && py::len(this->inputs()) != 0) {
            this->store_inputs(py::make_tuple(tensor));
        }
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        const Index outer_size = product_modes_without_axis(modes, axis_);
        const std::vector<Index> offsets = mode_leaf_offsets(modes);
        const Index axis_leaf_offset = offsets[static_cast<std::size_t>(axis_)];
        {
            py::gil_scoped_release release;
            for (Index outer = 0; outer < outer_size; ++outer) {
                const std::vector<Index> outer_key =
                    outer_key_for_ordinal(outer, modes, axis_);
                std::vector<std::pair<float, Index>> entries;
                entries.reserve(static_cast<std::size_t>(axis_extent));
                for (Index position = 0; position < axis_extent; ++position) {
                    const std::vector<Index> axis_key = decode_ordinal(
                        position,
                        modes.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                    std::vector<Index> source_key = outer_key;
                    source_key.insert(source_key.begin() + axis_leaf_offset,
                                      axis_key.begin(), axis_key.end());
                    entries.emplace_back(view.read_float_expanded(source_key),
                                         position);
                }
                std::stable_sort(entries.begin(), entries.end(),
                                 [this](const auto& lhs, const auto& rhs) {
                                     const bool lhs_nan = std::isnan(lhs.first);
                                     const bool rhs_nan = std::isnan(rhs.first);
                                     if (lhs_nan != rhs_nan) {
                                         return largest_ ? lhs_nan : !lhs_nan;
                                     }
                                     if (lhs_nan || lhs.first == rhs.first) {
                                         return lhs.second < rhs.second;
                                     }
                                     return largest_ ? lhs.first > rhs.first
                                                     : lhs.first < rhs.first;
                                 });
                for (Index position = 0; position < k_; ++position) {
                    const auto& entry = entries[static_cast<std::size_t>(position)];
                    const std::vector<Index> output_axis_key =
                        topk_
                            ? std::vector<Index>{position}
                            : decode_ordinal(
                                  position, modes.modes[static_cast<std::size_t>(axis_)]
                                                .leaf_extents);
                    std::vector<Index> output_key = outer_key;
                    output_key.insert(output_key.begin() + axis_leaf_offset,
                                      output_axis_key.begin(), output_axis_key.end());
                    if (values_) {
                        const Index output_ordinal =
                            expanded_key_ordinal(output_key, output_leaf_extents_);
                        std::vector<Index> source_key = outer_key;
                        const std::vector<Index> source_axis_key = decode_ordinal(
                            entry.second,
                            modes.modes[static_cast<std::size_t>(axis_)].leaf_extents);
                        source_key.insert(source_key.begin() + axis_leaf_offset,
                                          source_axis_key.begin(),
                                          source_axis_key.end());
                        source_keys_[static_cast<std::size_t>(output_ordinal)] =
                            std::move(source_key);
                    }
                    if (values_) {
                        result.view.write_float_expanded(output_key, entry.first);
                    } else {
                        result.view.write_int_expanded(output_key,
                                                       checked_int32(entry.second));
                    }
                }
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        if (!values_) {
            return py::make_tuple();
        }
        py::tuple input_tensors = inputs();
        if (py::len(input_tensors) != 1) {
            return py::make_tuple();
        }
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation result = allocate_gradient_for(tensor);
        std::vector<Index> output_key(gradient_view.leaf_rank(), 0);
        std::vector<Index> zero_key(result.view.leaf_rank(), 0);
        {
            py::gil_scoped_release release;
            for (Index i = 0; i < result.view.logical_size; ++i) {
                result.view.write_float_expanded(zero_key, 0.0f);
                result.view.cache->increment_key(zero_key.data(), zero_key.size());
            }
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                const Index output_ordinal =
                    expanded_key_ordinal(output_key, output_leaf_extents_);
                const std::vector<Index>& source_key =
                    source_keys_[static_cast<std::size_t>(output_ordinal)];
                result.view.write_float_expanded(
                    source_key, result.view.read_float_expanded(source_key) +
                                    gradient_view.read_float_expanded(output_key));
                gradient_view.cache->increment_key(output_key.data(),
                                                   output_key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    static bool require_bool(py::handle value, const char* name) {
        if (!py::isinstance<py::bool_>(value)) {
            throw py::type_error(std::string(name) + " must be a bool");
        }
        return py::cast<bool>(value);
    }

    static Index product_modes_without_axis(const CpuLayoutModes& modes, Index axis) {
        Index result = 1;
        for (Index i = 0; i < static_cast<Index>(modes.modes.size()); ++i) {
            if (i != axis) {
                result *= modes.modes[static_cast<std::size_t>(i)].logical_size();
            }
        }
        return result;
    }

    bool values_;
    bool topk_;
    bool largest_ = false;
    Index axis_ = 0;
    Index k_ = 0;
    CpuLayoutModes modes_;
    std::vector<Index> output_extents_;
    std::vector<Index> output_leaf_extents_;
    std::vector<std::vector<Index>> source_keys_;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kSortValuesMetadata{"_sort_values", "cpu.sort_values",
                                                "default", "_CPUSortOperation"};
constexpr CpuKernelMetadata kSortIndicesMetadata{"_sort_indices", "cpu.sort_indices",
                                                 "default", "_CPUSortOperation"};
constexpr CpuKernelMetadata kTopKValuesMetadata{"_topk_values", "cpu.topk_values",
                                                "default", "_CPUSortOperation"};
constexpr CpuKernelMetadata kTopKIndicesMetadata{"_topk_indices", "cpu.topk_indices",
                                                 "default", "_CPUSortOperation"};

void register_sort_variant(const CpuKernelMetadata& metadata, bool values, bool topk) {
    register_cpu_native_operation(metadata, [values, topk] {
        return py::cast(new CpuSortOperation(values, topk),
                        py::return_value_policy::take_ownership);
    });
}

}  // namespace

void register_cpu_sort(py::module_& module) {
    py::class_<CpuSortOperation, strideweave::operation::Operation>(module,
                                                                    "_CPUSortOperation")
        .def(py::init<bool, bool>());
    register_sort_variant(kSortValuesMetadata, true, false);
    register_sort_variant(kSortIndicesMetadata, false, false);
    register_sort_variant(kTopKValuesMetadata, true, true);
    register_sort_variant(kTopKIndicesMetadata, false, true);
}

}  // namespace strideweave::carrier
