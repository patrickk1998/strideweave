#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "_cpu_accumulation.hpp"
#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

template <typename Accumulator>
float floating_dot(const CpuTensorView& lhs_view, std::vector<Index>& lhs_key,
                   Index lhs_mode, const CpuTensorView& rhs_view,
                   std::vector<Index>& rhs_key, Index rhs_mode, Index length) {
    Accumulator sum = Accumulator{0};
    for (Index index = 0; index < length; ++index) {
        sum = accumulate_binary32_product(sum, lhs_view.read_float_expanded(lhs_key),
                                          rhs_view.read_float_expanded(rhs_key));
        lhs_view.cache->increment_mode(lhs_key.data(), lhs_key.size(), lhs_mode);
        rhs_view.cache->increment_mode(rhs_key.data(), rhs_key.size(), rhs_mode);
    }
    return store_float_accumulator(sum);
}

float planned_floating_dot(CpuAccumulatorKernel accumulator_kernel,
                           const CpuTensorView& lhs_view, std::vector<Index>& lhs_key,
                           Index lhs_mode, const CpuTensorView& rhs_view,
                           std::vector<Index>& rhs_key, Index rhs_mode, Index length) {
    if (accumulator_kernel == CpuAccumulatorKernel::Float32) {
        return floating_dot<float>(lhs_view, lhs_key, lhs_mode, rhs_view, rhs_key,
                                   rhs_mode, length);
    }
    if (accumulator_kernel == CpuAccumulatorKernel::Float64) {
        return floating_dot<double>(lhs_view, lhs_key, lhs_mode, rhs_view, rhs_key,
                                    rhs_mode, length);
    }
    throw std::logic_error("floating matmul plan has no accumulator dtype");
}

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

        const CpuPlan plan = resolve_binary_plan_with_options(
            lhs, "matmul", execution_options(), lhs_view, rhs_view);
        accumulator_kernel_ = accumulator_kernel_for(plan);
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
                    if (accumulator_kernel_ == CpuAccumulatorKernel::ExactInteger) {
                        ExactIntegerSum sum;
                        for (Index k = 0; k < lhs_k_size; ++k) {
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
                        result.view.write_float_expanded(
                            output_key,
                            planned_floating_dot(accumulator_kernel_, lhs_view, lhs_key,
                                                 1, rhs_view, rhs_key, 1, lhs_k_size));
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
                    gradient_key = gradient_i_base;
                    rhs_key = rhs_k_base;
                    lhs_result.view.write_float_expanded(
                        lhs_output_key,
                        planned_floating_dot(accumulator_kernel_, gradient_view,
                                             gradient_key, 1, rhs_view, rhs_key, 0,
                                             m_size));
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
                    gradient_key = gradient_j_base;
                    lhs_key = lhs_k_base_for_rhs;
                    rhs_result.view.write_float_expanded(
                        rhs_output_key,
                        planned_floating_dot(accumulator_kernel_, gradient_view,
                                             gradient_key, 0, lhs_view, lhs_key, 0,
                                             n_size));
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
    CpuAccumulatorKernel accumulator_kernel_ = CpuAccumulatorKernel::Float32;
};

constexpr CpuKernelMetadata kMetadata{"matmul", "cpu.matmul", "default",
                                      "_CPUMatmulOperation"};

}  // namespace

void register_cpu_matmul(py::module_& module) {
    bind_and_register_cpu_operation<CpuMatmulOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
