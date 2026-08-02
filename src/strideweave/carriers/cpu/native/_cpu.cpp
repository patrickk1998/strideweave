#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "_carrier.hpp"
#include "_cpu_carrier.hpp"
#include "_cpu_operation.hpp"

namespace py = pybind11;

namespace strideweave::carrier {
namespace {

using CpuOperationFactory = std::function<py::object()>;

std::unordered_map<std::string, CpuOperationFactory>& cpu_operation_registry() {
    static std::unordered_map<std::string, CpuOperationFactory> registry;
    return registry;
}

void register_cpu_operation(std::string operation_name,
                            CpuOperationFactory operation_factory) {
    cpu_operation_registry().insert_or_assign(std::move(operation_name),
                                              std::move(operation_factory));
}

template <typename Operation>
void register_native_cpu_operation(const char* operation_name) {
    register_cpu_operation(operation_name, [] {
        return py::cast(new Operation(), py::return_value_policy::take_ownership);
    });
}

void register_python_cpu_operation(const char* operation_name,
                                   const char* operation_module_name,
                                   const char* operation_type_name) {
    register_cpu_operation(operation_name, [operation_module_name,
                                            operation_type_name] {
        return py::module_::import(operation_module_name).attr(operation_type_name)();
    });
}

// CPU is a closed carrier implementation, so there is no Python subclass for a
// trampoline to dispatch back into. The wording matches
// `strideweave.carriers.base.CLOSED_CARRIER_MESSAGE`, which
// `tests/test_carrier.py::test_a_closed_carrier_states_one_refusal` pins.
[[noreturn]] void reject_cpu_subclass() {
    throw py::type_error(
        "CPU is a closed carrier implementation and cannot be subclassed; "
        "implement a sibling Carrier instead, normally by composing existing "
        "carriers and lowering operations onto them the way Evictable does");
}

struct CpuExpScalar {
    static constexpr const char* kOperation = "exp";
    static constexpr const char* kForwardError = "CPU exp requires a tensor";
    static float value(float input) { return std::exp(input); }
    static float gradient_multiplier(float input) { return std::exp(input); }
};

struct CpuSigmoidScalar {
    static constexpr const char* kOperation = "sigmoid";
    static constexpr const char* kForwardError = "CPU sigmoid requires a tensor";
    static float value(float input) { return sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid * (1.0f - sigmoid);
    }
};

struct CpuTanhScalar {
    static constexpr const char* kOperation = "tanh";
    static constexpr const char* kForwardError = "CPU tanh requires a tensor";
    static float value(float input) { return std::tanh(input); }
    static float gradient_multiplier(float input) {
        const float output = std::tanh(input);
        return 1.0f - output * output;
    }
};

struct CpuGELUScalar {
    static constexpr const char* kOperation = "gelu";
    static constexpr const char* kForwardError = "CPU GELU requires a tensor";
    static float value(float input) { return gelu_value(input); }
    static float gradient_multiplier(float input) {
        return gelu_gradient_multiplier(input);
    }
};

struct CpuSiLUScalar {
    static constexpr const char* kOperation = "silu";
    static constexpr const char* kForwardError = "CPU SiLU requires a tensor";
    static float value(float input) { return input * sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid + input * sigmoid * (1.0f - sigmoid);
    }
};

struct CpuSoftplusScalar {
    static constexpr const char* kOperation = "softplus";
    static constexpr const char* kForwardError = "CPU softplus requires a tensor";
    static float value(float input) { return softplus_value(input); }
    static float gradient_multiplier(float input) { return sigmoid_value(input); }
};

struct CpuELUScalar {
    static constexpr const char* kOperation = "elu";
    static constexpr const char* kForwardError = "CPU ELU requires a tensor";
    static float value(float input) { return input > 0.0f ? input : std::expm1(input); }
    static float gradient_multiplier(float input) {
        return input > 0.0f ? 1.0f : std::exp(input);
    }
};

struct CpuLeakyReLUScalar {
    static constexpr const char* kOperation = "leaky_relu";
    static constexpr const char* kForwardError = "CPU leaky ReLU requires a tensor";
    static float value(float input) {
        return input >= 0.0f ? input : kLeakyReluNegativeSlope * input;
    }
    static float gradient_multiplier(float input) {
        return input >= 0.0f ? 1.0f : kLeakyReluNegativeSlope;
    }
};

struct CpuNegScalar {
    static constexpr const char* kOperation = "neg";
    static constexpr const char* kForwardError = "CPU neg requires a tensor";
    static float value(float x) { return -x; }
    static float gradient_multiplier(float) { return -1.0f; }
};
struct CpuAbsScalar {
    static constexpr const char* kOperation = "abs";
    static constexpr const char* kForwardError = "CPU abs requires a tensor";
    static float value(float x) { return std::fabs(x); }
    static float gradient_multiplier(float x) {
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    }
};
struct CpuSignScalar {
    static constexpr const char* kOperation = "sign";
    static constexpr const char* kForwardError = "CPU sign requires a tensor";
    static float value(float x) {
        return std::isnan(x) ? std::numeric_limits<float>::quiet_NaN()
                             : (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f));
    }
    static float gradient_multiplier(float) { return 0.0f; }
};
struct CpuRecipScalar {
    static constexpr const char* kOperation = "recip";
    static constexpr const char* kForwardError = "CPU recip requires a tensor";
    static float value(float x) { return 1.0f / x; }
    static float gradient_multiplier(float x) { return -1.0f / (x * x); }
};
struct CpuSqrtScalar {
    static constexpr const char* kOperation = "sqrt";
    static constexpr const char* kForwardError = "CPU sqrt requires a tensor";
    static float value(float x) { return std::sqrt(x); }
    static float gradient_multiplier(float x) { return 1.0f / (2.0f * std::sqrt(x)); }
};
struct CpuRsqrtScalar {
    static constexpr const char* kOperation = "rsqrt";
    static constexpr const char* kForwardError = "CPU rsqrt requires a tensor";
    static float value(float x) { return 1.0f / std::sqrt(x); }
    static float gradient_multiplier(float x) {
        const float y = value(x);
        return -0.5f * y * y * y;
    }
};
struct CpuExp2Scalar {
    static constexpr const char* kOperation = "exp2";
    static constexpr const char* kForwardError = "CPU exp2 requires a tensor";
    static float value(float x) { return std::exp2(x); }
    static float gradient_multiplier(float x) { return std::log(2.0f) * std::exp2(x); }
};
struct CpuLogScalar {
    static constexpr const char* kOperation = "log";
    static constexpr const char* kForwardError = "CPU log requires a tensor";
    static float value(float x) { return std::log(x); }
    static float gradient_multiplier(float x) { return 1.0f / x; }
};
struct CpuLog2Scalar {
    static constexpr const char* kOperation = "log2";
    static constexpr const char* kForwardError = "CPU log2 requires a tensor";
    static float value(float x) { return std::log2(x); }
    static float gradient_multiplier(float x) { return 1.0f / (x * std::log(2.0f)); }
};
struct CpuSinScalar {
    static constexpr const char* kOperation = "sin";
    static constexpr const char* kForwardError = "CPU sin requires a tensor";
    static float value(float x) { return std::sin(x); }
    static float gradient_multiplier(float x) { return std::cos(x); }
};
struct CpuCosScalar {
    static constexpr const char* kOperation = "cos";
    static constexpr const char* kForwardError = "CPU cos requires a tensor";
    static float value(float x) { return std::cos(x); }
    static float gradient_multiplier(float x) { return -std::sin(x); }
};
struct CpuErfScalar {
    static constexpr const char* kOperation = "erf";
    static constexpr const char* kForwardError = "CPU erf requires a tensor";
    static float value(float x) { return std::erf(x); }
    static float gradient_multiplier(float x) {
        // Generic's NumPy binary32 scalar forces the 2/sqrt(pi) constant to
        // Float32 before multiplying the binary32 exponential.
        const float coefficient =
            static_cast<float>(2.0 / std::sqrt(3.14159265358979323846));
        return coefficient * std::exp(-(x * x));
    }
};
struct CpuFloorScalar {
    static constexpr const char* kOperation = "floor";
    static constexpr const char* kForwardError = "CPU floor requires a tensor";
    static float value(float x) { return std::floor(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};
struct CpuCeilScalar {
    static constexpr const char* kOperation = "ceil";
    static constexpr const char* kForwardError = "CPU ceil requires a tensor";
    static float value(float x) { return std::ceil(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};
struct CpuRoundScalar {
    static constexpr const char* kOperation = "round";
    static constexpr const char* kForwardError = "CPU round requires a tensor";
    static float value(float x) { return std::nearbyint(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};

using CpuExpOperation = CpuUnaryElementwiseOperation<CpuExpScalar>;
using CpuSigmoidOperation = CpuUnaryElementwiseOperation<CpuSigmoidScalar>;
using CpuTanhOperation = CpuUnaryElementwiseOperation<CpuTanhScalar>;
using CpuGELUOperation = CpuUnaryElementwiseOperation<CpuGELUScalar>;
using CpuSiLUOperation = CpuUnaryElementwiseOperation<CpuSiLUScalar>;
using CpuSoftplusOperation = CpuUnaryElementwiseOperation<CpuSoftplusScalar>;
using CpuELUOperation = CpuUnaryElementwiseOperation<CpuELUScalar>;
using CpuLeakyReLUOperation = CpuUnaryElementwiseOperation<CpuLeakyReLUScalar>;
using CpuNegOperation = CpuUnaryElementwiseOperation<CpuNegScalar>;
using CpuAbsOperation = CpuUnaryElementwiseOperation<CpuAbsScalar>;
using CpuSignOperation = CpuUnaryElementwiseOperation<CpuSignScalar>;
using CpuRecipOperation = CpuUnaryElementwiseOperation<CpuRecipScalar>;
using CpuSqrtOperation = CpuUnaryElementwiseOperation<CpuSqrtScalar>;
using CpuRsqrtOperation = CpuUnaryElementwiseOperation<CpuRsqrtScalar>;
using CpuExp2Operation = CpuUnaryElementwiseOperation<CpuExp2Scalar>;
using CpuLogOperation = CpuUnaryElementwiseOperation<CpuLogScalar>;
using CpuLog2Operation = CpuUnaryElementwiseOperation<CpuLog2Scalar>;
using CpuSinOperation = CpuUnaryElementwiseOperation<CpuSinScalar>;
using CpuCosOperation = CpuUnaryElementwiseOperation<CpuCosScalar>;
using CpuErfOperation = CpuUnaryElementwiseOperation<CpuErfScalar>;
using CpuFloorOperation = CpuUnaryElementwiseOperation<CpuFloorScalar>;
using CpuCeilOperation = CpuUnaryElementwiseOperation<CpuCeilScalar>;
using CpuRoundOperation = CpuUnaryElementwiseOperation<CpuRoundScalar>;

// Resolve the plan for a binary operation over two CPU tensors. The GIL is
// still held here, before any kernel releases it.
CpuPlan resolve_binary_plan(py::handle lhs, const char* operation,
                            const CpuTensorView& lhs_view,
                            const CpuTensorView& rhs_view) {
    return resolve_cpu_plan(executing_carrier_class(lhs), operation,
                            cpu_dtype_object(lhs_view.carrier->cpu_dtype()),
                            cpu_dtype_object(rhs_view.carrier->cpu_dtype()));
}

struct AlignedBinaryOperands {
    py::object lhs;
    py::object rhs;
    py::object result_layout;
};

struct AlignedTernaryOperands {
    py::object first;
    py::object second;
    py::object third;
    py::object result_layout;
};

AlignedBinaryOperands align_binary_operands(py::object lhs, py::object rhs) {
    py::tuple aligned = py::cast<py::tuple>(
        py::module_::import("strideweave.carriers.operation_helpers")
            .attr("_align_binary_operands")(lhs, rhs));
    return {
        py::reinterpret_borrow<py::object>(aligned[0]),
        py::reinterpret_borrow<py::object>(aligned[1]),
        py::reinterpret_borrow<py::object>(aligned[2]),
    };
}

void require_same_carrier_class(py::handle first, py::handle second,
                                const char* message) {
    if (!executing_carrier_class(first).is(executing_carrier_class(second))) {
        throw py::type_error(message);
    }
}

AlignedTernaryOperands align_ternary_operands(py::object first, py::object second,
                                              py::object third) {
    // Each binary alignment computes the same unique per-leaf common extent.
    // Re-aligning the first two results to the third operand widens every
    // original tensor directly to the final common shape, so no pairwise
    // ordering can choose a different result.
    AlignedBinaryOperands first_second =
        align_binary_operands(std::move(first), std::move(second));
    AlignedBinaryOperands first_third =
        align_binary_operands(std::move(first_second.lhs), std::move(third));
    AlignedBinaryOperands second_first =
        align_binary_operands(std::move(first_second.rhs), std::move(first_third.lhs));
    return {
        std::move(second_first.rhs),
        std::move(second_first.lhs),
        std::move(first_third.rhs),
        std::move(second_first.result_layout),
    };
}

Index product_extents(const std::vector<Index>& extents);

struct CpuModeInfo {
    py::object shape;
    std::vector<Index> leaf_extents;

    Index logical_size() const { return product_extents(leaf_extents); }
};

struct CpuLayoutModes {
    std::vector<CpuModeInfo> modes;
    std::vector<Index> leaf_counts;
};

void flatten_mode_shape(py::handle shape, std::vector<Index>& leaves) {
    if (strideweave::layout_index::is_int(shape)) {
        leaves.push_back(py::cast<Index>(shape));
        return;
    }
    for (py::handle child : py::reinterpret_borrow<py::sequence>(shape)) {
        flatten_mode_shape(child, leaves);
    }
}

CpuLayoutModes cpu_layout_modes(py::handle tensor, const char*) {
    py::object top_level = tensor_layout(tensor).attr("shape").attr("top_level");
    CpuLayoutModes result;
    for (py::handle shape : py::reinterpret_borrow<py::sequence>(top_level)) {
        CpuModeInfo mode;
        mode.shape = py::reinterpret_borrow<py::object>(shape);
        flatten_mode_shape(shape, mode.leaf_extents);
        result.leaf_counts.push_back(static_cast<Index>(mode.leaf_extents.size()));
        result.modes.push_back(std::move(mode));
    }
    return result;
}

Index normalize_axis(Index axis, Index rank) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw py::value_error("axis is out of range");
    }
    return axis;
}

Index product_extents(const std::vector<Index>& extents) {
    Index result = 1;
    for (Index extent : extents) {
        if (extent < 0 ||
            (extent != 0 && result > std::numeric_limits<Index>::max() / extent)) {
            throw py::value_error("layout extent is too large");
        }
        result *= extent;
    }
    return result;
}

Index floor_division(Index numerator, Index denominator) {
    Index quotient = numerator / denominator;
    const Index remainder = numerator % denominator;
    if (remainder < 0) {
        --quotient;
    }
    return quotient;
}

std::vector<Index> decode_ordinal(Index ordinal, const std::vector<Index>& leaves) {
    std::vector<Index> key(leaves.size(), 0);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const Index extent = leaves[i];
        key[i] = extent == 0 ? 0 : ordinal % extent;
        if (extent != 0) {
            ordinal /= extent;
        }
    }
    return key;
}

// Encode expanded (leaf-level) coordinates in StrideWeave's first-mode-fast
// order.  This is used by selection VJPs to retain a permutation keyed by the
// logical output ordinal, including nested modes.
Index expanded_key_ordinal(const std::vector<Index>& key,
                           const std::vector<Index>& extents) {
    if (key.size() != extents.size()) {
        throw py::value_error("expanded coordinate rank does not match layout");
    }
    Index ordinal = 0;
    Index factor = 1;
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (key[i] < 0 || key[i] >= extents[i]) {
            throw py::value_error("expanded coordinate is outside layout");
        }
        ordinal += key[i] * factor;
        factor *= extents[i];
    }
    return ordinal;
}

std::vector<Index> mode_leaf_offsets(const CpuLayoutModes& modes) {
    std::vector<Index> offsets(modes.modes.size(), 0);
    Index offset = 0;
    for (std::size_t i = 0; i < modes.modes.size(); ++i) {
        offsets[i] = offset;
        offset += static_cast<Index>(modes.modes[i].leaf_extents.size());
    }
    return offsets;
}

py::object canonical_layout_from_top_level(py::object top_level) {
    py::object layout_module = py::module_::import("strideweave.layout");
    py::object shape_type = layout_module.attr("Shape");
    py::object stride_type = layout_module.attr("Stride");
    py::object layout_type = layout_module.attr("Layout");
    py::object shape = shape_type(std::move(top_level));
    auto [stride_level, _] = canonical_stride_level(shape.attr("top_level"), 1);
    return layout_type(std::move(shape), stride_type(std::move(stride_level)));
}

std::vector<Index> outer_key_for_ordinal(Index ordinal, const CpuLayoutModes& modes,
                                         Index axis) {
    std::vector<Index> key;
    for (Index i = 0; i < static_cast<Index>(modes.modes.size()); ++i) {
        if (i == axis) {
            continue;
        }
        const Index size = modes.modes[static_cast<std::size_t>(i)].logical_size();
        const Index mode_ordinal = size == 0 ? 0 : ordinal % size;
        if (size != 0) {
            ordinal /= size;
        }
        std::vector<Index> mode_key = decode_ordinal(
            mode_ordinal, modes.modes[static_cast<std::size_t>(i)].leaf_extents);
        key.insert(key.end(), mode_key.begin(), mode_key.end());
    }
    return key;
}

template <typename FloatOperation, typename IntegerOperation>
py::object cpu_binary_elementwise_forward(strideweave::operation::Operation& owner,
                                          py::args inputs, const char* operation,
                                          const char* error_message,
                                          FloatOperation float_operation,
                                          IntegerOperation integer_operation) {
    if (py::len(inputs) != 2) {
        throw py::type_error(error_message);
    }
    py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
    py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
    AlignedBinaryOperands aligned =
        align_binary_operands(std::move(lhs), std::move(rhs));
    lhs = std::move(aligned.lhs);
    rhs = std::move(aligned.rhs);
    if (py::len(owner.inputs()) != 0) {
        py::tuple aligned_inputs = py::make_tuple(lhs, rhs);
        owner.store_inputs(py::reinterpret_borrow<py::args>(aligned_inputs));
    }
    CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
    CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");

    const CpuPlan plan = resolve_binary_plan(lhs, operation, lhs_view, rhs_view);
    CpuTensorAllocation result =
        allocate_cpu_tensor(std::move(aligned.result_layout), plan.output);
    {
        py::gil_scoped_release release;
        std::vector<Index> key(result.view.leaf_rank(), 0);
        for (Index i = 0; i < result.view.logical_size; ++i) {
            if (plan.is_integer()) {
                write_computed_int(
                    result.view, key,
                    integer_operation(
                        static_cast<long long>(lhs_view.read_int_expanded(key)),
                        static_cast<long long>(rhs_view.read_int_expanded(key))),
                    plan.compute);
            } else {
                result.view.write_float_expanded(
                    key, float_operation(lhs_view.read_float_expanded(key),
                                         rhs_view.read_float_expanded(key)));
            }
            result.view.cache->increment_key(key.data(), key.size());
        }
    }
    return make_tensor(std::move(result.carrier_object),
                       std::move(result.layout_object));
}

class CpuAddOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, "add", "CPU add requires lhs and rhs tensors",
            [](float lhs, float rhs) { return lhs + rhs; },
            [](long long lhs, long long rhs) { return lhs + rhs; });
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        return py::make_tuple(copy_gradient_for(lhs, gradient),
                              copy_gradient_for(rhs, gradient));
    }
};

class CpuSubOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, "sub", "CPU subtract requires lhs and rhs tensors",
            [](float lhs, float rhs) { return lhs - rhs; },
            [](long long lhs, long long rhs) { return lhs - rhs; });
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        return py::make_tuple(copy_gradient_for(lhs, gradient),
                              copy_negated_gradient_for(rhs, gradient));
    }
};

class CpuScalarMulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU scalar multiply requires a tensor and scalar");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object other = py::reinterpret_borrow<py::object>(inputs[1]);
        // ``mul`` has both tensor-tensor and tensor/weak-scalar policy
        // overloads.  Keep the historical scalar operation class as the
        // dispatch target, but route tensor-tensor through the same structural
        // alignment and integer-overflow checks as elementwise_mul.
        if (py::isinstance(other, tensor_type())) {
            return cpu_binary_elementwise_forward(
                *this, inputs, "mul", "CPU multiply requires lhs and rhs tensors",
                [](float lhs, float rhs) { return lhs * rhs; },
                [](long long lhs, long long rhs) { return lhs * rhs; });
        }
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        // The scalar is weak: it selects the plan but never forces a width, so
        // the plan — not the scalar's Python type — decides the result dtype.
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
        if (py::len(input_tensors) == 2) {
            py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
            require_same_shape(tensor, gradient);
            require_same_shape(rhs, gradient);
            CpuTensorView lhs_view = cpu_tensor_view(tensor, "lhs");
            CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
            CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
            CpuTensorAllocation lhs_result = allocate_gradient_for(tensor);
            CpuTensorAllocation rhs_result = allocate_gradient_for(rhs);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(lhs_view.leaf_rank(), 0);
                for (Index i = 0; i < lhs_view.logical_size; ++i) {
                    const float g = gradient_view.read_float_expanded(key);
                    lhs_result.view.write_float_expanded(
                        key, g * rhs_view.read_float_expanded(key));
                    rhs_result.view.write_float_expanded(
                        key, g * lhs_view.read_float_expanded(key));
                    lhs_view.cache->increment_key(key.data(), key.size());
                }
            }
            return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                              std::move(lhs_result.layout_object)),
                                  make_tensor(std::move(rhs_result.carrier_object),
                                              std::move(rhs_result.layout_object)));
        }
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

class CpuElementwiseMulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, "elementwise_mul",
            "CPU elementwise multiply requires lhs and rhs tensors",
            [](float lhs, float rhs) { return lhs * rhs; },
            [](long long lhs, long long rhs) { return lhs * rhs; });
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
                const float gradient_value = gradient_view.read_float_expanded(key);
                lhs_result.view.write_float_expanded(
                    key, gradient_value * rhs_view.read_float_expanded(key));
                rhs_result.view.write_float_expanded(
                    key, gradient_value * lhs_view.read_float_expanded(key));
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }

        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }
};

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

template <typename Predicate>
py::object cpu_binary_predicate_forward(strideweave::operation::Operation& owner,
                                        py::args inputs, const char* operation,
                                        const char* error_message,
                                        Predicate predicate) {
    if (py::len(inputs) != 2) {
        throw py::type_error(error_message);
    }
    py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
    py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
    AlignedBinaryOperands aligned =
        align_binary_operands(std::move(lhs), std::move(rhs));
    lhs = std::move(aligned.lhs);
    rhs = std::move(aligned.rhs);
    if (py::len(owner.inputs()) != 0) {
        owner.store_inputs(py::make_tuple(lhs, rhs));
    }
    CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
    CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
    const CpuPlan plan = resolve_binary_plan(lhs, operation, lhs_view, rhs_view);
    CpuTensorAllocation result =
        allocate_cpu_tensor(std::move(aligned.result_layout), plan.output);
    {
        py::gil_scoped_release release;
        std::vector<Index> key(result.view.leaf_rank(), 0);
        for (Index i = 0; i < result.view.logical_size; ++i) {
            result.view.write_bool_expanded(
                key, predicate(lhs_view.read_float_expanded(key),
                               rhs_view.read_float_expanded(key)));
            result.view.cache->increment_key(key.data(), key.size());
        }
    }
    return make_tensor(std::move(result.carrier_object),
                       std::move(result.layout_object));
}

template <bool IsMaximum> float cpu_extrema_value(float lhs, float rhs) {
    if (std::isnan(lhs)) {
        return lhs;
    }
    if (std::isnan(rhs)) {
        return rhs;
    }
    if (lhs == rhs) {
        // NumPy resolves mixed signed zeros toward +0 for maximum and -0 for
        // minimum, while preserving the sign when both zeros have that sign.
        if (lhs == 0.0f) {
            const bool negative = IsMaximum ? std::signbit(lhs) && std::signbit(rhs)
                                            : std::signbit(lhs) || std::signbit(rhs);
            return std::copysign(0.0f, negative ? -1.0f : 1.0f);
        }
        return lhs;
    }
    return IsMaximum ? (lhs > rhs ? lhs : rhs) : (lhs < rhs ? lhs : rhs);
}

template <bool IsMaximum>
void cpu_extrema_vjp(float lhs, float rhs, float gradient, float& lhs_gradient,
                     float& rhs_gradient) {
    lhs_gradient = 0.0f;
    rhs_gradient = 0.0f;
    if (std::isnan(lhs) || std::isnan(rhs)) {
        lhs_gradient = rhs_gradient = std::numeric_limits<float>::quiet_NaN();
    } else if (lhs == rhs && std::isfinite(lhs)) {
        lhs_gradient = rhs_gradient = 0.5f * gradient;
    } else if ((IsMaximum && lhs > rhs) || (!IsMaximum && lhs < rhs)) {
        lhs_gradient = gradient;
    } else {
        rhs_gradient = gradient;
    }
}

template <bool IsMaximum>
class CpuExtremaOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, IsMaximum ? "maximum" : "minimum",
            IsMaximum ? "CPU maximum requires lhs and rhs tensors"
                      : "CPU minimum requires lhs and rhs tensors",
            [](float lhs, float rhs) { return cpu_extrema_value<IsMaximum>(lhs, rhs); },
            [](long long lhs, long long rhs) {
                return IsMaximum ? std::max(lhs, rhs) : std::min(lhs, rhs);
            });
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
                const float x = lhs_view.read_float_expanded(key);
                const float y = rhs_view.read_float_expanded(key);
                const float g = gradient_view.read_float_expanded(key);
                float gx = 0.0f;
                float gy = 0.0f;
                cpu_extrema_vjp<IsMaximum>(x, y, g, gx, gy);
                lhs_result.view.write_float_expanded(key, gx);
                rhs_result.view.write_float_expanded(key, gy);
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }
};

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

class CpuClampOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 3) {
            throw py::type_error(
                "CPU clamp requires a tensor, lower bound, and upper bound");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object lower = py::reinterpret_borrow<py::object>(inputs[1]);
        py::object upper = py::reinterpret_borrow<py::object>(inputs[2]);
        lower_is_tensor_ = py::isinstance(lower, tensor_type());
        upper_is_tensor_ = py::isinstance(upper, tensor_type());

        CpuTensorView original_tensor = cpu_tensor_view(tensor, "tensor");
        if (lower_is_tensor_) {
            require_same_carrier_class(tensor, lower,
                                       "Tensor backing carriers must match");
        }
        if (upper_is_tensor_) {
            require_same_carrier_class(tensor, upper,
                                       "Tensor backing carriers must match");
        }

        const CpuPlan plan = [&]() {
            if (lower_is_tensor_ && upper_is_tensor_) {
                CpuTensorView lower_view = cpu_tensor_view(lower, "lower");
                CpuTensorView upper_view = cpu_tensor_view(upper, "upper");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()),
                    cpu_dtype_object(lower_view.carrier->cpu_dtype()),
                    cpu_dtype_object(upper_view.carrier->cpu_dtype()));
            }
            if (lower_is_tensor_) {
                CpuTensorView lower_view = cpu_tensor_view(lower, "lower");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()),
                    cpu_dtype_object(lower_view.carrier->cpu_dtype()), upper);
            }
            if (upper_is_tensor_) {
                CpuTensorView upper_view = cpu_tensor_view(upper, "upper");
                return resolve_cpu_plan(
                    executing_carrier_class(tensor), "clamp",
                    cpu_dtype_object(original_tensor.carrier->cpu_dtype()), lower,
                    cpu_dtype_object(upper_view.carrier->cpu_dtype()));
            }
            return resolve_cpu_plan(
                executing_carrier_class(tensor), "clamp",
                cpu_dtype_object(original_tensor.carrier->cpu_dtype()), lower, upper);
        }();

        if (!lower_is_tensor_) {
            lower_scalar_ = require_float(lower, "lower");
        }
        if (!upper_is_tensor_) {
            upper_scalar_ = require_float(upper, "upper");
        }

        if (lower_is_tensor_ && upper_is_tensor_) {
            AlignedTernaryOperands aligned = align_ternary_operands(
                std::move(tensor), std::move(lower), std::move(upper));
            tensor = std::move(aligned.first);
            lower = std::move(aligned.second);
            upper = std::move(aligned.third);
            output_layout_ = std::move(aligned.result_layout);
        } else if (lower_is_tensor_) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(tensor), std::move(lower));
            tensor = std::move(aligned.lhs);
            lower = std::move(aligned.rhs);
            output_layout_ = std::move(aligned.result_layout);
        } else if (upper_is_tensor_) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(tensor), std::move(upper));
            tensor = std::move(aligned.lhs);
            upper = std::move(aligned.rhs);
            output_layout_ = std::move(aligned.result_layout);
        } else {
            py::object top_level =
                tensor_layout(tensor).attr("shape").attr("top_level");
            output_layout_ = canonical_layout_from_top_level(
                py::reinterpret_borrow<py::object>(top_level));
        }

        if (py::len(this->inputs()) != 0) {
            if (lower_is_tensor_ && upper_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, lower, upper));
            } else if (lower_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, lower));
            } else if (upper_is_tensor_) {
                this->store_inputs(py::make_tuple(tensor, upper));
            } else {
                this->store_inputs(py::make_tuple(tensor));
            }
        }

        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView lower_view =
            lower_is_tensor_ ? cpu_tensor_view(lower, "lower") : tensor_view;
        CpuTensorView upper_view =
            upper_is_tensor_ ? cpu_tensor_view(upper, "upper") : tensor_view;
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < result.view.logical_size; ++i) {
                const float value = tensor_view.read_float_expanded(key);
                const float lower_value = lower_is_tensor_
                                              ? lower_view.read_float_expanded(key)
                                              : lower_scalar_;
                const float upper_value = upper_is_tensor_
                                              ? upper_view.read_float_expanded(key)
                                              : upper_scalar_;
                const float middle = cpu_extrema_value<true>(value, lower_value);
                result.view.write_float_expanded(
                    key, cpu_extrema_value<false>(middle, upper_value));
                result.view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        std::size_t cursor = 1;
        py::object lower = py::none();
        py::object upper = py::none();
        if (lower_is_tensor_) {
            lower = py::reinterpret_borrow<py::object>(input_tensors[cursor++]);
        }
        if (upper_is_tensor_) {
            upper = py::reinterpret_borrow<py::object>(input_tensors[cursor]);
        }
        require_layout(gradient, output_layout_);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView lower_view =
            lower_is_tensor_ ? cpu_tensor_view(lower, "lower") : tensor_view;
        CpuTensorView upper_view =
            upper_is_tensor_ ? cpu_tensor_view(upper, "upper") : tensor_view;
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        CpuTensorAllocation tensor_result = allocate_gradient_for(tensor);
        std::optional<CpuTensorAllocation> lower_result;
        std::optional<CpuTensorAllocation> upper_result;
        if (lower_is_tensor_) {
            lower_result.emplace(allocate_gradient_for(lower));
        }
        if (upper_is_tensor_) {
            upper_result.emplace(allocate_gradient_for(upper));
        }
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                const float value = tensor_view.read_float_expanded(key);
                const float lower_value = lower_is_tensor_
                                              ? lower_view.read_float_expanded(key)
                                              : lower_scalar_;
                const float upper_value = upper_is_tensor_
                                              ? upper_view.read_float_expanded(key)
                                              : upper_scalar_;
                const float middle = cpu_extrema_value<true>(value, lower_value);
                float middle_gradient = 0.0f;
                float upper_gradient = 0.0f;
                cpu_extrema_vjp<false>(middle, upper_value,
                                       gradient_view.read_float_expanded(key),
                                       middle_gradient, upper_gradient);
                float tensor_gradient = 0.0f;
                float lower_gradient = 0.0f;
                cpu_extrema_vjp<true>(value, lower_value, middle_gradient,
                                      tensor_gradient, lower_gradient);
                tensor_result.view.write_float_expanded(key, tensor_gradient);
                if (lower_result.has_value()) {
                    lower_result->view.write_float_expanded(key, lower_gradient);
                }
                if (upper_result.has_value()) {
                    upper_result->view.write_float_expanded(key, upper_gradient);
                }
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }

        py::tuple gradients(1 + static_cast<std::size_t>(lower_is_tensor_) +
                            static_cast<std::size_t>(upper_is_tensor_));
        std::size_t output = 0;
        gradients[output++] = make_tensor(std::move(tensor_result.carrier_object),
                                          std::move(tensor_result.layout_object));
        if (lower_result.has_value()) {
            gradients[output++] = make_tensor(std::move(lower_result->carrier_object),
                                              std::move(lower_result->layout_object));
        }
        if (upper_result.has_value()) {
            gradients[output] = make_tensor(std::move(upper_result->carrier_object),
                                            std::move(upper_result->layout_object));
        }
        return gradients;
    }

private:
    bool lower_is_tensor_ = false;
    bool upper_is_tensor_ = false;
    float lower_scalar_ = 0.0f;
    float upper_scalar_ = 0.0f;
    py::object output_layout_ = py::none();
};

class CpuRemOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_elementwise_forward(
            *this, inputs, "rem", "CPU remainder requires lhs and rhs tensors",
            [](float lhs, float rhs) { return std::fmod(lhs, rhs); },
            [](long long lhs, long long rhs) { return lhs % rhs; });
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
                const float x = lhs_view.read_float_expanded(key);
                const float y = rhs_view.read_float_expanded(key);
                const float g = gradient_view.read_float_expanded(key);
                const float q = std::trunc(x / y);
                lhs_result.view.write_float_expanded(key, g);
                rhs_result.view.write_float_expanded(key, -g * q);
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }
};

class CpuEqOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_predicate_forward(
            *this, inputs, "eq", "CPU eq requires lhs and rhs tensors",
            [](float lhs, float rhs) {
                return !std::isnan(lhs) && !std::isnan(rhs) && lhs == rhs;
            });
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

class CpuNeOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_predicate_forward(
            *this, inputs, "ne", "CPU ne requires lhs and rhs tensors",
            [](float lhs, float rhs) {
                return std::isnan(lhs) || std::isnan(rhs) || lhs != rhs;
            });
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

class CpuLtOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_predicate_forward(
            *this, inputs, "lt", "CPU lt requires lhs and rhs tensors",
            [](float lhs, float rhs) {
                return !std::isnan(lhs) && !std::isnan(rhs) && lhs < rhs;
            });
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

class CpuLeOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        return cpu_binary_predicate_forward(
            *this, inputs, "le", "CPU le requires lhs and rhs tensors",
            [](float lhs, float rhs) {
                return !std::isnan(lhs) && !std::isnan(rhs) && lhs <= rhs;
            });
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

class CpuLogicalNotOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU logical_not requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), "logical_not",
                             cpu_dtype_object(view.carrier->cpu_dtype()));
        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(tensor), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(view.leaf_rank(), 0);
            for (Index i = 0; i < view.logical_size; ++i) {
                // logical_not is Float32-only.  NaN is nonzero (therefore
                // logically true), while either signed zero is false.
                result.view.write_bool_expanded(key,
                                                view.read_float_expanded(key) == 0.0f);
                view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

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

inline float cpu_reduce_max_value(float lhs, float rhs) {
    if (std::isnan(lhs)) {
        return lhs;
    }
    if (std::isnan(rhs)) {
        return rhs;
    }
    if (lhs == rhs && lhs == 0.0f) {
        return std::copysign(0.0f, 1.0f);
    }
    return lhs > rhs ? lhs : rhs;
}

inline float cpu_reduce_min_value(float lhs, float rhs) {
    if (std::isnan(lhs)) {
        return lhs;
    }
    if (std::isnan(rhs)) {
        return rhs;
    }
    if (lhs == rhs && lhs == 0.0f) {
        return std::copysign(0.0f, -1.0f);
    }
    return lhs < rhs ? lhs : rhs;
}

template <typename Kind>
class CpuFloatReductionOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error(Kind::error_message());
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        require_two_mode_tensor(tensor, "tensor");
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        const Index n = mode_logical_size(tensor_layout(tensor), 0);
        const Index m = mode_logical_size(tensor_layout(tensor), 1);
        if (m <= 0) {
            throw py::value_error("reduction fibers must be non-empty");
        }
        output_layout_ =
            canonical_layout_from_modes({mode_shape(tensor_layout(tensor), 0)});
        ctx_["output_layout"] = output_layout_;
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), Kind::operation(),
                             cpu_dtype_object(view.carrier->cpu_dtype()));
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(view.leaf_rank(), 0);
            std::vector<Index> input_key(view.leaf_rank(), 0);
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < n; ++i) {
                input_key = row_key;
                if constexpr (Kind::kArg) {
                    float first = view.read_float_expanded(input_key);
                    Index best = 0;
                    for (Index j = 1; j < m; ++j) {
                        view.cache->increment_mode(input_key.data(), input_key.size(),
                                                   1);
                        const float candidate = view.read_float_expanded(input_key);
                        const bool candidate_nan = std::isnan(candidate);
                        const bool best_nan = std::isnan(first);
                        if ((candidate_nan && !best_nan) ||
                            (candidate_nan == best_nan &&
                             ((Kind::kMax && candidate > first) ||
                              (!Kind::kMax && candidate < first)))) {
                            first = candidate;
                            best = j;
                        }
                    }
                    result.view.write_int_expanded(output_key, checked_int32(best));
                } else if constexpr (Kind::kProd) {
                    float accumulator = 1.0f;
                    for (Index j = 0; j < m; ++j) {
                        accumulator *= view.read_float_expanded(input_key);
                        view.cache->increment_mode(input_key.data(), input_key.size(),
                                                   1);
                    }
                    result.view.write_float_expanded(output_key, accumulator);
                } else {
                    float accumulator = view.read_float_expanded(input_key);
                    for (Index j = 1; j < m; ++j) {
                        view.cache->increment_mode(input_key.data(), input_key.size(),
                                                   1);
                        const float candidate = view.read_float_expanded(input_key);
                        accumulator =
                            Kind::kMax ? cpu_reduce_max_value(accumulator, candidate)
                                       : cpu_reduce_min_value(accumulator, candidate);
                    }
                    result.view.write_float_expanded(output_key, accumulator);
                }
                view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                result.view.cache->increment_key(output_key.data(), output_key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        if constexpr (Kind::kArg) {
            return py::make_tuple();
        }
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        const Index n = mode_logical_size(tensor_layout(tensor), 0);
        const Index m = mode_logical_size(tensor_layout(tensor), 1);
        CpuTensorAllocation result = allocate_gradient_for(tensor);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(view.leaf_rank(), 0);
            std::vector<Index> key(view.leaf_rank(), 0);
            std::vector<Index> grad_key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < n; ++i) {
                const float g = gradient_view.read_float_expanded(grad_key);
                if constexpr (Kind::kProd) {
                    key = row_key;
                    for (Index j = 0; j < m; ++j) {
                        float product = 1.0f;
                        std::vector<Index> other = row_key;
                        for (Index k = 0; k < m; ++k) {
                            if (k != j) {
                                product *= view.read_float_expanded(other);
                            }
                            view.cache->increment_mode(other.data(), other.size(), 1);
                        }
                        result.view.write_float_expanded(key, g * product);
                        result.view.cache->increment_mode(key.data(), key.size(), 1);
                    }
                } else {
                    key = row_key;
                    float reduced = view.read_float_expanded(key);
                    for (Index j = 1; j < m; ++j) {
                        view.cache->increment_mode(key.data(), key.size(), 1);
                        const float candidate = view.read_float_expanded(key);
                        reduced = Kind::kMax ? cpu_reduce_max_value(reduced, candidate)
                                             : cpu_reduce_min_value(reduced, candidate);
                    }
                    const bool nan_result = std::isnan(reduced);
                    Index winners = 0;
                    if (!nan_result) {
                        key = row_key;
                        for (Index j = 0; j < m; ++j) {
                            winners += (view.read_float_expanded(key) == reduced);
                            view.cache->increment_mode(key.data(), key.size(), 1);
                        }
                    }
                    const float each = nan_result
                                           ? std::numeric_limits<float>::quiet_NaN()
                                           : g / static_cast<float>(winners);
                    key = row_key;
                    for (Index j = 0; j < m; ++j) {
                        const float value = view.read_float_expanded(key);
                        result.view.write_float_expanded(
                            key, nan_result ? each : (value == reduced ? each : 0.0f));
                        view.cache->increment_mode(key.data(), key.size(), 1);
                    }
                }
                view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                gradient_view.cache->increment_key(grad_key.data(), grad_key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    py::object output_layout_ = py::none();
};

struct CpuReduceProdKind {
    static constexpr bool kProd = true;
    [[maybe_unused]] static constexpr bool kMax = false;
    static constexpr bool kArg = false;
    static constexpr const char* operation() { return "reduce_prod"; }
    static constexpr const char* error_message() {
        return "CPU reduce_prod requires a tensor";
    }
};
struct CpuReduceMaxKind {
    static constexpr bool kProd = false, kMax = true, kArg = false;
    static constexpr const char* operation() { return "reduce_max"; }
    static constexpr const char* error_message() {
        return "CPU reduce_max requires a tensor";
    }
};
struct CpuReduceMinKind {
    static constexpr bool kProd = false, kMax = false, kArg = false;
    static constexpr const char* operation() { return "reduce_min"; }
    static constexpr const char* error_message() {
        return "CPU reduce_min requires a tensor";
    }
};
struct CpuArgmaxKind {
    static constexpr bool kProd = false, kMax = true, kArg = true;
    static constexpr const char* operation() { return "argmax"; }
    static constexpr const char* error_message() {
        return "CPU argmax requires a tensor";
    }
};
struct CpuArgminKind {
    static constexpr bool kProd = false, kMax = false, kArg = true;
    static constexpr const char* operation() { return "argmin"; }
    static constexpr const char* error_message() {
        return "CPU argmin requires a tensor";
    }
};

using CpuReduceProdOperation = CpuFloatReductionOperation<CpuReduceProdKind>;
using CpuReduceMaxOperation = CpuFloatReductionOperation<CpuReduceMaxKind>;
using CpuReduceMinOperation = CpuFloatReductionOperation<CpuReduceMinKind>;
using CpuArgmaxOperation = CpuFloatReductionOperation<CpuArgmaxKind>;
using CpuArgminOperation = CpuFloatReductionOperation<CpuArgminKind>;

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

class CpuPowOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU power requires a tensor and exponent");
        }
        py::object first = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object second = py::reinterpret_borrow<py::object>(inputs[1]);
        if (py::isinstance(first, tensor_type()) &&
            py::isinstance(second, tensor_type())) {
            AlignedBinaryOperands aligned =
                align_binary_operands(std::move(first), std::move(second));
            first = std::move(aligned.lhs);
            second = std::move(aligned.rhs);
            if (py::len(this->inputs()) != 0) {
                this->store_inputs(py::make_tuple(first, second));
            }
            CpuTensorView base = cpu_tensor_view(first, "base");
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            const CpuPlan plan = resolve_binary_plan(first, "pow", base, exponent);
            CpuTensorAllocation result =
                allocate_cpu_tensor(std::move(aligned.result_layout), plan.output);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(result.view.leaf_rank(), 0);
                for (Index i = 0; i < result.view.logical_size; ++i) {
                    result.view.write_float_expanded(
                        key, std::pow(base.read_float_expanded(key),
                                      exponent.read_float_expanded(key)));
                    result.view.cache->increment_key(key.data(), key.size());
                }
            }
            return make_tensor(std::move(result.carrier_object),
                               std::move(result.layout_object));
        }
        if (!py::isinstance(first, tensor_type()) &&
            py::isinstance(second, tensor_type())) {
            scalar_base_ = true;
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            const CpuPlan plan =
                resolve_cpu_plan(executing_carrier_class(second), "pow", first,
                                 cpu_dtype_object(exponent.carrier->cpu_dtype()));
            base_ = require_float(first, "base");
            ctx_["base"] = py::float_(base_);
            CpuTensorAllocation result =
                allocate_cpu_tensor(injective_layout_for(second), plan.output);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(exponent.leaf_rank(), 0);
                for (Index i = 0; i < exponent.logical_size; ++i) {
                    result.view.write_float_expanded(
                        key, std::pow(base_, exponent.read_float_expanded(key)));
                    exponent.cache->increment_key(key.data(), key.size());
                }
            }
            return make_tensor(std::move(result.carrier_object),
                               std::move(result.layout_object));
        }

        // Tensor raised to a weak scalar.  Whether an Int32 result is
        // preserved is central policy, so the exponent is materialized exactly
        // for the integer path and only once as binary32 otherwise.
        if (!py::isinstance(first, tensor_type())) {
            throw py::type_error("CPU power requires a tensor and exponent");
        }
        CpuTensorView tensor_view = cpu_tensor_view(first, "tensor");
        const CpuPlan plan = resolve_cpu_plan(
            executing_carrier_class(first), "pow",
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()), second);
        exponent_ = require_float(second, "exponent");
        ctx_["exponent"] = py::float_(exponent_);
        const std::int32_t int_exponent =
            plan.is_integer() ? require_int32_scalar(second, "exponent") : 0;

        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(first), plan.output);
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
        py::object first = py::reinterpret_borrow<py::object>(input_tensors[0]);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
        if (py::len(input_tensors) == 2) {
            py::object second = py::reinterpret_borrow<py::object>(input_tensors[1]);
            require_same_shape(first, gradient);
            require_same_shape(second, gradient);
            CpuTensorView base = cpu_tensor_view(first, "base");
            CpuTensorView exponent = cpu_tensor_view(second, "exponent");
            CpuTensorAllocation base_result = allocate_gradient_for(first);
            CpuTensorAllocation exponent_result = allocate_gradient_for(second);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(base.leaf_rank(), 0);
                for (Index i = 0; i < base.logical_size; ++i) {
                    const float x = base.read_float_expanded(key);
                    const float p = exponent.read_float_expanded(key);
                    const float y = std::pow(x, p);
                    const float g = gradient_view.read_float_expanded(key);
                    base_result.view.write_float_expanded(
                        key, g * p * std::pow(x, p - 1.0f));
                    exponent_result.view.write_float_expanded(key, g * y * std::log(x));
                    base.cache->increment_key(key.data(), key.size());
                }
            }
            return py::make_tuple(
                make_tensor(std::move(base_result.carrier_object),
                            std::move(base_result.layout_object)),
                make_tensor(std::move(exponent_result.carrier_object),
                            std::move(exponent_result.layout_object)));
        }
        // In scalar-base form only the tensor exponent receives a gradient.
        if (py::len(input_tensors) == 1 && scalar_base_) {
            py::object exponent = py::reinterpret_borrow<py::object>(input_tensors[0]);
            require_same_shape(exponent, gradient);
            CpuTensorView exponent_view = cpu_tensor_view(exponent, "exponent");
            CpuTensorAllocation result = allocate_gradient_for(exponent);
            {
                py::gil_scoped_release release;
                std::vector<Index> key(exponent_view.leaf_rank(), 0);
                for (Index i = 0; i < exponent_view.logical_size; ++i) {
                    const float y =
                        std::pow(base_, exponent_view.read_float_expanded(key));
                    result.view.write_float_expanded(
                        key,
                        gradient_view.read_float_expanded(key) * y * std::log(base_));
                    exponent_view.cache->increment_key(key.data(), key.size());
                }
            }
            return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                              std::move(result.layout_object)));
        }
        py::object tensor = first;
        require_same_shape(tensor, gradient);
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");
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
    float base_ = 0.0f;
    bool scalar_base_ = false;
};

class CpuReduceSumOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU reduce requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        require_two_mode_tensor(tensor, "tensor");
        CpuTensorView tensor_view = cpu_tensor_view(tensor, "tensor");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        output_layout_ =
            canonical_layout_from_modes({mode_shape(tensor_layout(tensor), 0)});
        ctx_["output_layout"] = output_layout_;

        // The plan's accumulation decides how the terms combine: an exact
        // integer accumulator whose only checked step is the final narrowing,
        // or a floating accumulator in the dtype the plan declares. This
        // backend currently declares only the Float32 floating accumulator.
        const CpuPlan plan = resolve_cpu_plan_with_options(
            executing_carrier_class(tensor), "reduce_sum", execution_options(),
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()));
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> input_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                input_key = row_key;
                if (plan.accumulation == CpuAccumulation::ExactInteger) {
                    ExactIntegerSum sum;
                    for (Index j = 0; j < m_size; ++j) {
                        sum.add(tensor_view.read_int_expanded(input_key));
                        tensor_view.cache->increment_mode(input_key.data(),
                                                          input_key.size(), 1);
                    }
                    write_accumulated_int(result.view, output_key, sum);
                } else {
                    float sum = 0.0f;
                    for (Index j = 0; j < m_size; ++j) {
                        sum += tensor_view.read_float_expanded(input_key);
                        tensor_view.cache->increment_mode(input_key.data(),
                                                          input_key.size(), 1);
                    }
                    result.view.write_float_expanded(output_key, sum);
                }
                tensor_view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                result.view.cache->increment_key(output_key.data(), output_key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        CpuTensorAllocation result = allocate_gradient_for(tensor);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(result.view.leaf_rank(), 0);
            std::vector<Index> input_key(result.view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                const float gradient_value =
                    gradient_view.read_float_expanded(gradient_key);
                input_key = row_key;
                for (Index j = 0; j < m_size; ++j) {
                    result.view.write_float_expanded(input_key, gradient_value);
                    result.view.cache->increment_mode(input_key.data(),
                                                      input_key.size(), 1);
                }
                result.view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                gradient_view.cache->increment_key(gradient_key.data(),
                                                   gradient_key.size());
            }
        }
        return py::make_tuple(make_tensor(std::move(result.carrier_object),
                                          std::move(result.layout_object)));
    }

private:
    py::object output_layout_ = py::none();
};

class CpuMatmulOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU matmul requires lhs and rhs tensors");
        }
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        require_two_mode_tensor(lhs, "lhs");
        require_two_mode_tensor(rhs, "rhs");
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index lhs_k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);
        const Index rhs_k_size = mode_logical_size(tensor_layout(rhs), 1);
        if (lhs_k_size != rhs_k_size) {
            throw py::value_error("Matmul inner dimensions must match");
        }

        output_layout_ = canonical_layout_from_modes(
            {mode_shape(tensor_layout(lhs), 0), mode_shape(tensor_layout(rhs), 0)});
        ctx_["output_layout"] = output_layout_;

        // Matmul promotes like binary arithmetic but accumulates: two Int32
        // operands compute each product exactly and check only the narrowing of
        // the finished sum, so terms that legitimately cancel are not rejected.
        const CpuPlan plan = resolve_cpu_plan_with_options(
            executing_carrier_class(lhs), "matmul", execution_options(),
            cpu_dtype_object(lhs_view.carrier->cpu_dtype()),
            cpu_dtype_object(rhs_view.carrier->cpu_dtype()));
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            std::vector<Index> rhs_j_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_i_base(lhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index j = 0; j < m_size; ++j) {
                std::fill(lhs_i_base.begin(), lhs_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    lhs_key = lhs_i_base;
                    rhs_key = rhs_j_base;
                    if (plan.accumulation == CpuAccumulation::ExactInteger) {
                        ExactIntegerSum sum;
                        for (Index k = 0; k < lhs_k_size; ++k) {
                            // Each product is exact in int64; only their sum
                            // needs the wider accumulator.
                            sum.add(static_cast<std::int64_t>(
                                        lhs_view.read_int_expanded(lhs_key)) *
                                    static_cast<std::int64_t>(
                                        rhs_view.read_int_expanded(rhs_key)));
                            lhs_view.cache->increment_mode(lhs_key.data(),
                                                           lhs_key.size(), 1);
                            rhs_view.cache->increment_mode(rhs_key.data(),
                                                           rhs_key.size(), 1);
                        }
                        write_accumulated_int(result.view, output_key, sum);
                    } else {
                        float sum = 0.0f;
                        for (Index k = 0; k < lhs_k_size; ++k) {
                            sum += lhs_view.read_float_expanded(lhs_key) *
                                   rhs_view.read_float_expanded(rhs_key);
                            lhs_view.cache->increment_mode(lhs_key.data(),
                                                           lhs_key.size(), 1);
                            rhs_view.cache->increment_mode(rhs_key.data(),
                                                           rhs_key.size(), 1);
                        }
                        result.view.write_float_expanded(output_key, sum);
                    }
                    result.view.cache->increment_key(output_key.data(),
                                                     output_key.size());
                    lhs_view.cache->increment_mode(lhs_i_base.data(), lhs_i_base.size(),
                                                   0);
                }
                rhs_view.cache->increment_mode(rhs_j_base.data(), rhs_j_base.size(), 0);
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }

    py::object backward(py::object gradient) override {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_layout(gradient, output_layout_);
        CpuTensorView lhs_view = cpu_tensor_view(lhs, "lhs");
        CpuTensorView rhs_view = cpu_tensor_view(rhs, "rhs");
        CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);

        CpuTensorAllocation lhs_result = allocate_gradient_for(lhs);
        CpuTensorAllocation rhs_result = allocate_gradient_for(rhs);
        {
            py::gil_scoped_release release;
            std::vector<Index> lhs_k_base(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> rhs_k_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_output_key(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_i_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                lhs_output_key = lhs_k_base;
                std::fill(gradient_i_base.begin(), gradient_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    float sum = 0.0f;
                    gradient_key = gradient_i_base;
                    rhs_key = rhs_k_base;
                    for (Index j = 0; j < m_size; ++j) {
                        sum += gradient_view.read_float_expanded(gradient_key) *
                               rhs_view.read_float_expanded(rhs_key);
                        gradient_view.cache->increment_mode(gradient_key.data(),
                                                            gradient_key.size(), 1);
                        rhs_view.cache->increment_mode(rhs_key.data(), rhs_key.size(),
                                                       0);
                    }
                    lhs_result.view.write_float_expanded(lhs_output_key, sum);
                    lhs_result.view.cache->increment_mode(lhs_output_key.data(),
                                                          lhs_output_key.size(), 0);
                    gradient_view.cache->increment_mode(gradient_i_base.data(),
                                                        gradient_i_base.size(), 0);
                }
                lhs_result.view.cache->increment_mode(lhs_k_base.data(),
                                                      lhs_k_base.size(), 1);
                rhs_view.cache->increment_mode(rhs_k_base.data(), rhs_k_base.size(), 1);
            }

            std::vector<Index> rhs_k_output_base(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> lhs_k_base_for_rhs(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_output_key(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_j_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                rhs_output_key = rhs_k_output_base;
                std::fill(gradient_j_base.begin(), gradient_j_base.end(), 0);
                for (Index j = 0; j < m_size; ++j) {
                    float sum = 0.0f;
                    gradient_key = gradient_j_base;
                    lhs_key = lhs_k_base_for_rhs;
                    for (Index i = 0; i < n_size; ++i) {
                        sum += gradient_view.read_float_expanded(gradient_key) *
                               lhs_view.read_float_expanded(lhs_key);
                        gradient_view.cache->increment_mode(gradient_key.data(),
                                                            gradient_key.size(), 0);
                        lhs_view.cache->increment_mode(lhs_key.data(), lhs_key.size(),
                                                       0);
                    }
                    rhs_result.view.write_float_expanded(rhs_output_key, sum);
                    rhs_result.view.cache->increment_mode(rhs_output_key.data(),
                                                          rhs_output_key.size(), 0);
                    gradient_view.cache->increment_mode(gradient_j_base.data(),
                                                        gradient_j_base.size(), 1);
                }
                rhs_result.view.cache->increment_mode(rhs_k_output_base.data(),
                                                      rhs_k_output_base.size(), 1);
                lhs_view.cache->increment_mode(lhs_k_base_for_rhs.data(),
                                               lhs_k_base_for_rhs.size(), 1);
            }
        }

        return py::make_tuple(make_tensor(std::move(lhs_result.carrier_object),
                                          std::move(lhs_result.layout_object)),
                              make_tensor(std::move(rhs_result.carrier_object),
                                          std::move(rhs_result.layout_object)));
    }

private:
    py::object output_layout_ = py::none();
};

}  // namespace

py::object CPU::_dispatch_op(const std::string& operation_name) const {
    auto& registry = cpu_operation_registry();
    auto operation_factory = registry.find(operation_name);
    if (operation_factory == registry.end()) {
        PyErr_Format(PyExc_NotImplementedError,
                     "CPU carrier does not support operation '%s'",
                     operation_name.c_str());
        throw py::error_already_set();
    }
    return operation_factory->second();
}

void bind_cpu(py::module_& module) {
    py::class_<CPU, Carrier>(module, "CPU")
        // Closed through `__init_subclass__` rather than `py::is_final()` on
        // purpose. `is_final()` refuses at the C level, which is the stronger
        // guarantee, but it reports CPython's generic "not an acceptable base
        // type" instead of the wording every closed carrier shares. That shared
        // wording is the contract
        // `tests/test_carrier.py::test_a_closed_carrier_states_one_refusal`
        // pins, so switching to `is_final()` would break it.
        .def_static("__init_subclass__",
                    [](const py::args&, const py::kwargs&) { reject_cpu_subclass(); })
        .def(py::init<Index, py::object, bool, py::object, bool>(), py::arg("size"),
             py::arg("pointer") = py::none(), py::kw_only(), py::arg("mutable") = true,
             py::arg("dtype") = py::none(), py::arg("empty") = false)
        .def("new_like", &CPU::new_like_with_dtype, py::arg("values"), py::kw_only(),
             py::arg("mutable") = true, py::arg("dtype") = py::none())
        .def("allocate_like", &CPU::allocate_like, py::arg("size"), py::kw_only(),
             py::arg("mutable") = true, py::arg("dtype") = py::none(),
             py::arg("empty") = false)
        .def("_dispatch_op", &CPU::dispatch_registered_op, py::arg("operation_name"))
        .def("pointer", &CPU::pointer)
        .def("set_value", &CPU::set_value_public, py::arg("index"), py::arg("value"));

    bind_cpu_operation<CpuAddOperation>(module, "_CPUAddOperation");
    bind_cpu_operation<CpuSubOperation>(module, "_CPUSubOperation");
    bind_cpu_operation<CpuScalarMulOperation>(module, "_CPUScalarMulOperation");
    bind_cpu_operation<CpuElementwiseMulOperation>(module,
                                                   "_CPUElementwiseMulOperation");
    bind_cpu_operation<CpuDivOperation>(module, "_CPUDivOperation");
    bind_cpu_operation<CpuExpOperation>(module, "_CPUExpOperation");
    bind_cpu_operation<CpuReLUOperation>(module, "_CPUReLUOperation");
    bind_cpu_operation<CpuSigmoidOperation>(module, "_CPUSigmoidOperation");
    bind_cpu_operation<CpuTanhOperation>(module, "_CPUTanhOperation");
    bind_cpu_operation<CpuGELUOperation>(module, "_CPUGELUOperation");
    bind_cpu_operation<CpuSiLUOperation>(module, "_CPUSiLUOperation");
    bind_cpu_operation<CpuSoftplusOperation>(module, "_CPUSoftplusOperation");
    bind_cpu_operation<CpuELUOperation>(module, "_CPUELUOperation");
    bind_cpu_operation<CpuLeakyReLUOperation>(module, "_CPULeakyReLUOperation");
    bind_cpu_operation<CpuNegOperation>(module, "_CPUNegOperation");
    bind_cpu_operation<CpuAbsOperation>(module, "_CPUAbsOperation");
    bind_cpu_operation<CpuSignOperation>(module, "_CPUSignOperation");
    bind_cpu_operation<CpuRecipOperation>(module, "_CPURecipOperation");
    bind_cpu_operation<CpuSqrtOperation>(module, "_CPUSqrtOperation");
    bind_cpu_operation<CpuRsqrtOperation>(module, "_CPURsqrtOperation");
    bind_cpu_operation<CpuExp2Operation>(module, "_CPUExp2Operation");
    bind_cpu_operation<CpuLogOperation>(module, "_CPULogOperation");
    bind_cpu_operation<CpuLog2Operation>(module, "_CPULog2Operation");
    bind_cpu_operation<CpuSinOperation>(module, "_CPUSinOperation");
    bind_cpu_operation<CpuCosOperation>(module, "_CPUCosOperation");
    bind_cpu_operation<CpuErfOperation>(module, "_CPUErfOperation");
    bind_cpu_operation<CpuFloorOperation>(module, "_CPUFloorOperation");
    bind_cpu_operation<CpuCeilOperation>(module, "_CPUCeilOperation");
    bind_cpu_operation<CpuRoundOperation>(module, "_CPURoundOperation");
    bind_cpu_operation<CpuPowOperation>(module, "_CPUPowOperation");
    bind_cpu_operation<CpuReduceSumOperation>(module, "_CPUReduceSumOperation");
    bind_cpu_operation<CpuReduceProdOperation>(module, "_CPUReduceProdOperation");
    bind_cpu_operation<CpuReduceMaxOperation>(module, "_CPUReduceMaxOperation");
    bind_cpu_operation<CpuReduceMinOperation>(module, "_CPUReduceMinOperation");
    bind_cpu_operation<CpuArgmaxOperation>(module, "_CPUArgmaxOperation");
    bind_cpu_operation<CpuArgminOperation>(module, "_CPUArgminOperation");
    bind_cpu_operation<CpuCumsumOperation>(module, "_CPUCumsumOperation");
    bind_cpu_operation<CpuRemOperation>(module, "_CPURemOperation");
    bind_cpu_operation<CpuEqOperation>(module, "_CPUEqOperation");
    bind_cpu_operation<CpuNeOperation>(module, "_CPUNeOperation");
    bind_cpu_operation<CpuLtOperation>(module, "_CPULtOperation");
    bind_cpu_operation<CpuLeOperation>(module, "_CPULeOperation");
    bind_cpu_operation<CpuLogicalNotOperation>(module, "_CPULogicalNotOperation");
    bind_cpu_operation<CpuExtremaOperation<true>>(module, "_CPUMaximumOperation");
    bind_cpu_operation<CpuExtremaOperation<false>>(module, "_CPUMinimumOperation");
    bind_cpu_operation<CpuSelectOperation>(module, "_CPUSelectOperation");
    bind_cpu_operation<CpuClampOperation>(module, "_CPUClampOperation");
    py::class_<CpuGatherOperation, strideweave::operation::Operation>(
        module, "_CPUGatherOperation")
        .def(py::init<>());
    py::class_<CpuScatterOperation, strideweave::operation::Operation>(
        module, "_CPUScatterOperation")
        .def(py::init<bool>());
    py::class_<CpuSortOperation, strideweave::operation::Operation>(module,
                                                                    "_CPUSortOperation")
        .def(py::init<bool, bool>());
    py::class_<CpuConvGeneralOperation, strideweave::operation::Operation>(
        module, "_CPUConvGeneralOperation")
        .def(py::init<>());
    bind_cpu_operation<CpuMatmulOperation>(module, "_CPUMatmulOperation");

    register_native_cpu_operation<CpuAddOperation>("add");
    register_native_cpu_operation<CpuDivOperation>("div");
    register_native_cpu_operation<CpuELUOperation>("elu");
    register_native_cpu_operation<CpuElementwiseMulOperation>("elementwise_mul");
    register_native_cpu_operation<CpuExpOperation>("exp");
    register_native_cpu_operation<CpuGELUOperation>("gelu");
    register_native_cpu_operation<CpuLeakyReLUOperation>("leaky_relu");
    register_native_cpu_operation<CpuMatmulOperation>("matmul");
    register_native_cpu_operation<CpuScalarMulOperation>("mul");
    register_native_cpu_operation<CpuPowOperation>("pow");
    register_native_cpu_operation<CpuReduceSumOperation>("reduce_sum");
    register_native_cpu_operation<CpuReLUOperation>("relu");
    register_native_cpu_operation<CpuSigmoidOperation>("sigmoid");
    register_native_cpu_operation<CpuSiLUOperation>("silu");
    register_native_cpu_operation<CpuSoftplusOperation>("softplus");
    register_native_cpu_operation<CpuSubOperation>("sub");
    register_native_cpu_operation<CpuRemOperation>("rem");
    register_native_cpu_operation<CpuExtremaOperation<true>>("maximum");
    register_native_cpu_operation<CpuExtremaOperation<false>>("minimum");
    register_native_cpu_operation<CpuSelectOperation>("select");
    register_native_cpu_operation<CpuClampOperation>("clamp");
    register_native_cpu_operation<CpuEqOperation>("eq");
    register_native_cpu_operation<CpuNeOperation>("ne");
    register_native_cpu_operation<CpuLtOperation>("lt");
    register_native_cpu_operation<CpuLeOperation>("le");
    register_native_cpu_operation<CpuLogicalNotOperation>("logical_not");
    register_native_cpu_operation<CpuNegOperation>("neg");
    register_native_cpu_operation<CpuAbsOperation>("abs");
    register_native_cpu_operation<CpuSignOperation>("sign");
    register_native_cpu_operation<CpuRecipOperation>("recip");
    register_native_cpu_operation<CpuSqrtOperation>("sqrt");
    register_native_cpu_operation<CpuRsqrtOperation>("rsqrt");
    register_native_cpu_operation<CpuExp2Operation>("exp2");
    register_native_cpu_operation<CpuLogOperation>("log");
    register_native_cpu_operation<CpuLog2Operation>("log2");
    register_native_cpu_operation<CpuSinOperation>("sin");
    register_native_cpu_operation<CpuCosOperation>("cos");
    register_native_cpu_operation<CpuErfOperation>("erf");
    register_native_cpu_operation<CpuFloorOperation>("floor");
    register_native_cpu_operation<CpuCeilOperation>("ceil");
    register_native_cpu_operation<CpuRoundOperation>("round");
    register_native_cpu_operation<CpuReduceProdOperation>("reduce_prod");
    register_native_cpu_operation<CpuReduceMaxOperation>("reduce_max");
    register_native_cpu_operation<CpuReduceMinOperation>("reduce_min");
    register_native_cpu_operation<CpuArgmaxOperation>("argmax");
    register_native_cpu_operation<CpuArgminOperation>("argmin");
    register_native_cpu_operation<CpuCumsumOperation>("cumsum");
    register_native_cpu_operation<CpuConvGeneralOperation>("conv_general");
    register_native_cpu_operation<CpuGatherOperation>("gather");
    register_cpu_operation("scatter", [] {
        return py::cast(new CpuScatterOperation(false),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_operation("scatter_add", [] {
        return py::cast(new CpuScatterOperation(true),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_operation("_sort_values", [] {
        return py::cast(new CpuSortOperation(true, false),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_operation("_sort_indices", [] {
        return py::cast(new CpuSortOperation(false, false),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_operation("_topk_values", [] {
        return py::cast(new CpuSortOperation(true, true),
                        py::return_value_policy::take_ownership);
    });
    register_cpu_operation("_topk_indices", [] {
        return py::cast(new CpuSortOperation(false, true),
                        py::return_value_policy::take_ownership);
    });
    register_native_cpu_operation<CpuTanhOperation>("tanh");
    register_python_cpu_operation("broadcast_to", "strideweave.carriers.shared_ops",
                                  "BroadcastOperation");
    register_python_cpu_operation("permute", "strideweave.carriers.shared_ops",
                                  "PermuteOperation");
    register_python_cpu_operation("rearrange", "strideweave.carriers.shared_ops",
                                  "RearrangeOperation");
    register_python_cpu_operation("reshape", "strideweave.carriers.shared_ops",
                                  "ReshapeOperation");
    register_python_cpu_operation("as_strided",
                                  "strideweave.carriers.generic.as_strided_ops",
                                  "GenericAsStridedOperation");
    register_python_cpu_operation("squeeze", "strideweave.carriers.shared_ops",
                                  "SqueezeOperation");
    register_python_cpu_operation("unsqueeze", "strideweave.carriers.shared_ops",
                                  "UnsqueezeOperation");
    register_python_cpu_operation("view", "strideweave.carriers.shared_ops",
                                  "GenericViewOperation");
}

}  // namespace strideweave::carrier
