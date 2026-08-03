#pragma once

#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "_cpu_layout.hpp"
#include "_cpu_operation.hpp"

namespace py = pybind11;

namespace strideweave::carrier {

// The Float32 reductions with a pinned per-term rule — product, extrema, and
// the arg reductions — share one traversal, one axis lowering, and one VJP
// shape, and differ only in the Kind policy they are instantiated with.

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

}  // namespace strideweave::carrier
