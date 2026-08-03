#include "_cpu_layout.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuConvGeneralOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) < 4 || py::len(inputs) > 10) {
            throw py::type_error(
                "CPU conv_general requires lhs, kernel, strides, and padding");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object kernel = py::reinterpret_borrow<py::object>(inputs[1]);
        lhs_modes_ = cpu_layout_modes(lhs, "conv_general lhs");
        kernel_modes_ = cpu_layout_modes(kernel, "conv_general kernel");
        const CpuLayoutModes& lhs_modes = lhs_modes_;
        const CpuLayoutModes& kernel_modes = kernel_modes_;
        if (lhs_modes.modes.size() < 3 ||
            kernel_modes.modes.size() != lhs_modes.modes.size()) {
            throw py::value_error(
                "conv_general requires batch, feature, and spatial modes");
        }
        spatial_rank_ = static_cast<Index>(lhs_modes.modes.size()) - 2;
        const std::size_t spatial_rank = static_cast<std::size_t>(spatial_rank_);
        lhs_roles_ = inputs.size() > 7 && !inputs[7].is_none()
                         ? parse_permutation(inputs[7],
                                             static_cast<Index>(lhs_modes.modes.size()),
                                             "lhs_dims")
                         : canonical_roles(static_cast<Index>(lhs_modes.modes.size()));
        kernel_roles_ =
            inputs.size() > 8 && !inputs[8].is_none()
                ? parse_permutation(inputs[8],
                                    static_cast<Index>(kernel_modes.modes.size()),
                                    "kernel_dims")
                : canonical_roles(static_cast<Index>(kernel_modes.modes.size()));
        const std::vector<Index> output_roles =
            inputs.size() > 9 && !inputs[9].is_none()
                ? parse_permutation(inputs[9],
                                    static_cast<Index>(lhs_modes.modes.size()),
                                    "output_dims")
                : canonical_roles(static_cast<Index>(lhs_modes.modes.size()));
        (void)output_roles;
        const std::vector<Index>& lhs_roles = lhs_roles_;
        const std::vector<Index>& kernel_roles = kernel_roles_;
        strides_ = parse_int_sequence(inputs[2], spatial_rank_, "strides");
        padding_ = parse_padding(inputs[3], spatial_rank_);
        lhs_dilation_ =
            inputs.size() > 4 && !inputs[4].is_none()
                ? parse_int_sequence(inputs[4], spatial_rank_, "lhs_dilation")
                : std::vector<Index>(spatial_rank, 1);
        kernel_dilation_ =
            inputs.size() > 5 && !inputs[5].is_none()
                ? parse_int_sequence(inputs[5], spatial_rank_, "kernel_dilation")
                : std::vector<Index>(spatial_rank, 1);
        if (inputs.size() > 6 && py::isinstance<py::bool_>(inputs[6])) {
            throw py::type_error("feature_groups must be an integer");
        }
        const Index groups = inputs.size() > 6 ? py::cast<Index>(inputs[6]) : 1;
        groups_ = groups;
        const std::vector<Index>& strides = strides_;
        const std::vector<std::pair<Index, Index>>& padding = padding_;
        const std::vector<Index>& lhs_dilation = lhs_dilation_;
        const std::vector<Index>& kernel_dilation = kernel_dilation_;
        if (groups <= 0) {
            throw py::value_error("feature_groups must be positive");
        }
        const Index n_size =
            lhs_modes.modes[static_cast<std::size_t>(lhs_roles[0])].logical_size();
        const Index cin_size =
            lhs_modes.modes[static_cast<std::size_t>(lhs_roles[1])].logical_size();
        const Index cout_size =
            kernel_modes.modes[static_cast<std::size_t>(kernel_roles[0])]
                .logical_size();
        const Index kernel_cin =
            kernel_modes.modes[static_cast<std::size_t>(kernel_roles[1])]
                .logical_size();
        for (std::size_t d = 0; d < spatial_rank; ++d) {
            if (lhs_modes.modes[static_cast<std::size_t>(lhs_roles[d + 2])]
                        .leaf_extents.size() != 1 ||
                kernel_modes.modes[static_cast<std::size_t>(kernel_roles[d + 2])]
                        .leaf_extents.size() != 1) {
                throw py::value_error("conv_general spatial role modes must be leaves");
            }
        }
        if (cin_size % groups != 0 || cout_size % groups != 0 ||
            kernel_cin != cin_size / groups) {
            throw py::value_error(
                "conv_general feature and group extents are incompatible");
        }
        std::vector<Index> x_sizes, k_sizes, y_sizes;
        for (std::size_t d = 0; d < spatial_rank; ++d) {
            const Index x = lhs_modes.modes[static_cast<std::size_t>(lhs_roles[d + 2])]
                                .logical_size();
            const Index k =
                kernel_modes.modes[static_cast<std::size_t>(kernel_roles[d + 2])]
                    .logical_size();
            const Index effective_x = (x - 1) * lhs_dilation[d] + 1;
            const Index effective_k = (k - 1) * kernel_dilation[d] + 1;
            const Index y = floor_division(padding[d].first + effective_x +
                                               padding[d].second - effective_k,
                                           strides[d]) +
                            1;
            if (x <= 0 || k <= 0 || y <= 0) {
                throw py::value_error("conv_general extents must be positive");
            }
            x_sizes.push_back(x);
            k_sizes.push_back(k);
            y_sizes.push_back(y);
        }
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView kernel_view = cpu_tensor_view(kernel, "kernel");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(lhs), "conv_general",
                             cpu_dtype_object(lhs_view.carrier->cpu_dtype()),
                             cpu_dtype_object(kernel_view.carrier->cpu_dtype()));
        py::list output_top;
        output_top.append(
            lhs_modes.modes[static_cast<std::size_t>(lhs_roles[0])].shape);
        output_top.append(
            kernel_modes.modes[static_cast<std::size_t>(kernel_roles[0])].shape);
        for (Index y : y_sizes) {
            output_top.append(py::int_(y));
        }
        output_layout_ = canonical_layout_from_top_level(std::move(output_top));
        if (py::len(this->inputs()) != 0) {
            this->store_inputs(py::make_tuple(lhs, kernel));
        }
        x_sizes_ = x_sizes;
        k_sizes_ = k_sizes;
        y_sizes_ = y_sizes;
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        const Index k_product = product_extents(k_sizes);
        const Index cin_group = cin_size / groups;
        {
            py::gil_scoped_release release;
            const Index spatial_output_size = product_extents(y_sizes);
            for (Index n = 0; n < n_size; ++n) {
                const std::vector<Index> n_key = decode_ordinal(
                    n, lhs_modes.modes[static_cast<std::size_t>(lhs_roles[0])]
                           .leaf_extents);
                for (Index o = 0; o < cout_size; ++o) {
                    const std::vector<Index> o_key = decode_ordinal(
                        o, kernel_modes.modes[static_cast<std::size_t>(kernel_roles[0])]
                               .leaf_extents);
                    for (Index y_ordinal = 0; y_ordinal < spatial_output_size;
                         ++y_ordinal) {
                        const std::vector<Index> y = decode_ordinal(y_ordinal, y_sizes);
                        float sum = 0.0f;
                        for (Index k_ordinal = 0; k_ordinal < k_product; ++k_ordinal) {
                            const std::vector<Index> k =
                                decode_ordinal(k_ordinal, k_sizes);
                            for (Index c = 0; c < cin_group; ++c) {
                                std::vector<Index> kernel_key;
                                const std::vector<Index> kernel_channel =
                                    decode_ordinal(c,
                                                   kernel_modes
                                                       .modes[static_cast<std::size_t>(
                                                           kernel_roles[1])]
                                                       .leaf_extents);
                                std::vector<std::vector<Index>> kernel_role_keys(
                                    kernel_modes.modes.size());
                                kernel_role_keys[static_cast<std::size_t>(
                                    kernel_roles[0])] = o_key;
                                kernel_role_keys[static_cast<std::size_t>(
                                    kernel_roles[1])] = kernel_channel;
                                for (std::size_t d = 0; d < spatial_rank; ++d) {
                                    kernel_role_keys[static_cast<std::size_t>(
                                        kernel_roles[d + 2])] =
                                        decode_ordinal(
                                            k[d], kernel_modes
                                                      .modes[static_cast<std::size_t>(
                                                          kernel_roles[d + 2])]
                                                      .leaf_extents);
                                }
                                for (const auto& role_key : kernel_role_keys) {
                                    kernel_key.insert(kernel_key.end(),
                                                      role_key.begin(), role_key.end());
                                }
                                std::vector<Index> lhs_key;
                                const Index channel =
                                    (o / (cout_size / groups)) * cin_group + c;
                                const std::vector<Index> channel_key = decode_ordinal(
                                    channel,
                                    lhs_modes
                                        .modes[static_cast<std::size_t>(lhs_roles[1])]
                                        .leaf_extents);
                                std::vector<Index> input_coord;
                                bool valid = true;
                                for (std::size_t d = 0; d < spatial_rank; ++d) {
                                    const Index u = y[d] * strides[d] +
                                                    k[d] * kernel_dilation[d] -
                                                    padding[d].first;
                                    if (u < 0 || u % lhs_dilation[d] != 0 ||
                                        u / lhs_dilation[d] >= x_sizes[d]) {
                                        valid = false;
                                        break;
                                    }
                                    input_coord.push_back(u / lhs_dilation[d]);
                                }
                                float lhs_value = 0.0f;
                                if (valid) {
                                    std::vector<std::vector<Index>> lhs_role_keys(
                                        lhs_modes.modes.size());
                                    lhs_role_keys[static_cast<std::size_t>(
                                        lhs_roles[0])] = n_key;
                                    lhs_role_keys[static_cast<std::size_t>(
                                        lhs_roles[1])] = channel_key;
                                    for (std::size_t d = 0; d < spatial_rank; ++d) {
                                        lhs_role_keys[static_cast<std::size_t>(
                                            lhs_roles[d + 2])] =
                                            decode_ordinal(
                                                input_coord[d],
                                                lhs_modes
                                                    .modes[static_cast<std::size_t>(
                                                        lhs_roles[d + 2])]
                                                    .leaf_extents);
                                    }
                                    for (const auto& role_key : lhs_role_keys) {
                                        lhs_key.insert(lhs_key.end(), role_key.begin(),
                                                       role_key.end());
                                    }
                                    lhs_value = lhs_view.read_float_expanded(lhs_key);
                                }
                                // Padding/input-dilation holes use an explicit
                                // binary32 +0 lhs value; ordinary IEEE
                                // multiplication is intentional (including
                                // 0*inf -> NaN).
                                sum += lhs_value *
                                       kernel_view.read_float_expanded(kernel_key);
                            }
                        }
                        std::vector<Index> output_key = n_key;
                        output_key.insert(output_key.end(), o_key.begin(), o_key.end());
                        output_key.insert(output_key.end(), y.begin(), y.end());
                        result.view.write_float_expanded(output_key, sum);
                    }
                }
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        if (py::len(input_tensors) != 2) {
            return py::make_tuple();
        }
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object kernel = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_layout(gradient, output_layout_);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView kernel_view = cpu_tensor_view(kernel, "kernel");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation lhs_result = allocate_gradient_for(lhs);
        CpuTensorAllocation kernel_result = allocate_gradient_for(kernel);
        const Index n_size =
            lhs_modes_.modes[static_cast<std::size_t>(lhs_roles_[0])].logical_size();
        const Index cin_size =
            lhs_modes_.modes[static_cast<std::size_t>(lhs_roles_[1])].logical_size();
        const Index cout_size =
            kernel_modes_.modes[static_cast<std::size_t>(kernel_roles_[0])]
                .logical_size();
        const Index cin_group = cin_size / groups_;
        const Index k_product = product_extents(k_sizes_);
        const Index y_product = product_extents(y_sizes_);
        const std::size_t spatial_rank = static_cast<std::size_t>(spatial_rank_);
        {
            py::gil_scoped_release release;
            for (Index n = 0; n < n_size; ++n) {
                const std::vector<Index> n_key = decode_ordinal(
                    n, lhs_modes_.modes[static_cast<std::size_t>(lhs_roles_[0])]
                           .leaf_extents);
                for (Index o = 0; o < cout_size; ++o) {
                    const std::vector<Index> o_key = decode_ordinal(
                        o,
                        kernel_modes_.modes[static_cast<std::size_t>(kernel_roles_[0])]
                            .leaf_extents);
                    for (Index y_ord = 0; y_ord < y_product; ++y_ord) {
                        const std::vector<Index> y = decode_ordinal(y_ord, y_sizes_);
                        std::vector<Index> output_key = n_key;
                        output_key.insert(output_key.end(), o_key.begin(), o_key.end());
                        output_key.insert(output_key.end(), y.begin(), y.end());
                        const float g = gradient_view.read_float_expanded(output_key);
                        for (Index k_ord = 0; k_ord < k_product; ++k_ord) {
                            const std::vector<Index> k =
                                decode_ordinal(k_ord, k_sizes_);
                            for (Index c = 0; c < cin_group; ++c) {
                                const Index channel =
                                    (o / (cout_size / groups_)) * cin_group + c;
                                const std::vector<Index> channel_key = decode_ordinal(
                                    channel,
                                    lhs_modes_
                                        .modes[static_cast<std::size_t>(lhs_roles_[1])]
                                        .leaf_extents);
                                const std::vector<Index> kernel_channel =
                                    decode_ordinal(c,
                                                   kernel_modes_
                                                       .modes[static_cast<std::size_t>(
                                                           kernel_roles_[1])]
                                                       .leaf_extents);
                                std::vector<std::vector<Index>> kernel_role_keys(
                                    kernel_modes_.modes.size());
                                kernel_role_keys[static_cast<std::size_t>(
                                    kernel_roles_[0])] = o_key;
                                kernel_role_keys[static_cast<std::size_t>(
                                    kernel_roles_[1])] = kernel_channel;
                                for (std::size_t d = 0; d < spatial_rank; ++d) {
                                    kernel_role_keys[static_cast<std::size_t>(
                                        kernel_roles_[d + 2])] =
                                        decode_ordinal(
                                            k[d], kernel_modes_
                                                      .modes[static_cast<std::size_t>(
                                                          kernel_roles_[d + 2])]
                                                      .leaf_extents);
                                }
                                std::vector<Index> kernel_key;
                                for (const auto& role_key : kernel_role_keys) {
                                    kernel_key.insert(kernel_key.end(),
                                                      role_key.begin(), role_key.end());
                                }
                                std::vector<Index> input_coord;
                                bool valid = true;
                                for (std::size_t d = 0; d < spatial_rank; ++d) {
                                    const Index u = y[d] * strides_[d] +
                                                    k[d] * kernel_dilation_[d] -
                                                    padding_[d].first;
                                    if (u < 0 || u % lhs_dilation_[d] != 0 ||
                                        u / lhs_dilation_[d] >= x_sizes_[d]) {
                                        valid = false;
                                        break;
                                    }
                                    input_coord.push_back(u / lhs_dilation_[d]);
                                }
                                std::vector<Index> lhs_key;
                                std::vector<std::vector<Index>> lhs_role_keys(
                                    lhs_modes_.modes.size());
                                lhs_role_keys[static_cast<std::size_t>(lhs_roles_[0])] =
                                    n_key;
                                lhs_role_keys[static_cast<std::size_t>(lhs_roles_[1])] =
                                    channel_key;
                                if (valid) {
                                    for (std::size_t d = 0; d < spatial_rank; ++d) {
                                        lhs_role_keys[static_cast<std::size_t>(
                                            lhs_roles_[d + 2])] =
                                            decode_ordinal(
                                                input_coord[d],
                                                lhs_modes_
                                                    .modes[static_cast<std::size_t>(
                                                        lhs_roles_[d + 2])]
                                                    .leaf_extents);
                                    }
                                    for (const auto& role_key : lhs_role_keys) {
                                        lhs_key.insert(lhs_key.end(), role_key.begin(),
                                                       role_key.end());
                                    }
                                }
                                const float kernel_value =
                                    kernel_view.read_float_expanded(kernel_key);
                                const float lhs_value =
                                    valid ? lhs_view.read_float_expanded(lhs_key)
                                          : 0.0f;
                                if (valid) {
                                    lhs_result.view.write_float_expanded(
                                        lhs_key,
                                        lhs_result.view.read_float_expanded(lhs_key) +
                                            g * kernel_value);
                                }
                                kernel_result.view.write_float_expanded(
                                    kernel_key,
                                    kernel_result.view.read_float_expanded(kernel_key) +
                                        g * lhs_value);
                            }
                        }
                    }
                }
            }
        }
        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(kernel_result.carrier_object),
                                          std::move(kernel_result.layout_object)));
    }

private:
    static std::vector<Index> canonical_roles(Index rank) {
        std::vector<Index> result;
        for (Index i = 0; i < rank; ++i) {
            result.push_back(i);
        }
        return result;
    }

    static std::vector<Index> parse_permutation(py::handle value, Index rank,
                                                const char* name) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (py::len(sequence) != static_cast<std::size_t>(rank)) {
            throw py::value_error(std::string(name) +
                                  " must list every top-level mode");
        }
        std::vector<Index> result;
        std::unordered_set<Index> seen;
        for (py::handle item : sequence) {
            if (py::isinstance<py::bool_>(item)) {
                throw py::type_error(std::string(name) + " entries must be integers");
            }
            const Index mode = py::cast<Index>(item);
            if (mode < 0 || mode >= rank || !seen.insert(mode).second) {
                throw py::value_error(std::string(name) + " must be a permutation");
            }
            result.push_back(mode);
        }
        return result;
    }

    static std::vector<Index> parse_int_sequence(py::handle value, Index rank,
                                                 const char* name) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (py::len(sequence) != static_cast<std::size_t>(rank)) {
            throw py::value_error(std::string(name) + " must match spatial rank");
        }
        std::vector<Index> result;
        for (py::handle item : sequence) {
            if (py::isinstance<py::bool_>(item)) {
                throw py::type_error(std::string(name) + " entries must be integers");
            }
            Index parsed = py::cast<Index>(item);
            if (parsed <= 0) {
                throw py::value_error(std::string(name) + " entries must be positive");
            }
            result.push_back(parsed);
        }
        return result;
    }

    static std::vector<std::pair<Index, Index>> parse_padding(py::handle value,
                                                              Index rank) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (py::len(sequence) != static_cast<std::size_t>(rank)) {
            throw py::value_error("padding must match spatial rank");
        }
        std::vector<std::pair<Index, Index>> result;
        for (py::handle item : sequence) {
            py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
            if (py::len(pair) != 2) {
                throw py::value_error("padding entries must be (low, high) pairs");
            }
            if (py::isinstance<py::bool_>(pair[0]) ||
                py::isinstance<py::bool_>(pair[1])) {
                throw py::type_error("padding entries must be integers");
            }
            Index low = py::cast<Index>(pair[0]);
            Index high = py::cast<Index>(pair[1]);
            if (low < 0 || high < 0) {
                throw py::value_error("padding entries must be non-negative");
            }
            result.emplace_back(low, high);
        }
        return result;
    }

    // Conv metadata is retained for the deterministic VJP pass.
    CpuLayoutModes lhs_modes_;
    CpuLayoutModes kernel_modes_;
    std::vector<Index> lhs_roles_;
    std::vector<Index> kernel_roles_;
    std::vector<Index> strides_;
    std::vector<std::pair<Index, Index>> padding_;
    std::vector<Index> lhs_dilation_;
    std::vector<Index> kernel_dilation_;
    std::vector<Index> x_sizes_;
    std::vector<Index> k_sizes_;
    std::vector<Index> y_sizes_;
    Index spatial_rank_ = 0;
    Index groups_ = 1;
    py::object output_layout_ = py::none();
};

constexpr CpuKernelMetadata kMetadata{"conv_general", "cpu.conv_general", "default",
                                      "_CPUConvGeneralOperation"};

}  // namespace

void register_cpu_conv_general(py::module_& module) {
    bind_and_register_cpu_operation<CpuConvGeneralOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
