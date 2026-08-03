#pragma once

#include <pybind11/pybind11.h>

#include <utility>
#include <vector>

#include "_cpu_carrier.hpp"
#include "_operation.hpp"

namespace py = pybind11;

namespace strideweave::carrier {

inline CpuPlan resolve_binary_plan(py::handle lhs, const char* operation,
                                   const CpuTensorView& lhs_view,
                                   const CpuTensorView& rhs_view) {
    return resolve_cpu_plan(executing_carrier_class(lhs), operation,
                            cpu_dtype_object(lhs_view.carrier->cpu_dtype()),
                            cpu_dtype_object(rhs_view.carrier->cpu_dtype()));
}

inline CpuPlan resolve_binary_plan_with_options(py::handle lhs, const char* operation,
                                                py::handle options,
                                                const CpuTensorView& lhs_view,
                                                const CpuTensorView& rhs_view) {
    return resolve_cpu_plan_with_options(
        executing_carrier_class(lhs), operation, options,
        cpu_dtype_object(lhs_view.carrier->cpu_dtype()),
        cpu_dtype_object(rhs_view.carrier->cpu_dtype()));
}

struct AlignedBinaryOperands {
    py::object lhs;
    py::object rhs;
    py::object result_layout;
};

inline AlignedBinaryOperands align_binary_operands(py::object lhs, py::object rhs) {
    py::tuple aligned = py::cast<py::tuple>(
        py::module_::import("strideweave.carriers.operation_helpers")
            .attr("_align_binary_operands")(lhs, rhs));
    return {
        py::reinterpret_borrow<py::object>(aligned[0]),
        py::reinterpret_borrow<py::object>(aligned[1]),
        py::reinterpret_borrow<py::object>(aligned[2]),
    };
}

struct AlignedTernaryOperands {
    py::object first;
    py::object second;
    py::object third;
    py::object result_layout;
};

inline void require_same_carrier_class(py::handle first, py::handle second,
                                       const char* message) {
    if (!executing_carrier_class(first).is(executing_carrier_class(second))) {
        throw py::type_error(message);
    }
}

inline AlignedTernaryOperands
align_ternary_operands(py::object first, py::object second, py::object third) {
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

}  // namespace strideweave::carrier
