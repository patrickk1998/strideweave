#include <stdexcept>
#include <vector>

#include "_cpu_accumulation.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

template <typename Accumulator>
void reduce_floating(const CpuTensorView& tensor_view, CpuTensorView& result_view,
                     Index n_size, Index m_size) {
    std::vector<Index> row_key(tensor_view.leaf_rank(), 0);
    std::vector<Index> input_key(tensor_view.leaf_rank(), 0);
    std::vector<Index> output_key(result_view.leaf_rank(), 0);
    for (Index i = 0; i < n_size; ++i) {
        input_key = row_key;
        Accumulator sum = Accumulator{0};
        for (Index j = 0; j < m_size; ++j) {
            sum = accumulate_float(sum, tensor_view.read_float_expanded(input_key));
            tensor_view.cache->increment_mode(input_key.data(), input_key.size(), 1);
        }
        result_view.write_float_expanded(output_key, store_float_accumulator(sum));
        tensor_view.cache->increment_mode(row_key.data(), row_key.size(), 0);
        result_view.cache->increment_key(output_key.data(), output_key.size());
    }
}

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
        // or a floating accumulator in the dtype the plan declares.
        const CpuPlan plan = resolve_cpu_plan_with_options(
            executing_carrier_class(tensor), "reduce_sum", execution_options(),
            cpu_dtype_object(tensor_view.carrier->cpu_dtype()));
        const CpuAccumulatorKernel accumulator_kernel = accumulator_kernel_for(plan);
        CpuTensorAllocation result = allocate_cpu_tensor(output_layout_, plan.output);
        {
            py::gil_scoped_release release;
            if (accumulator_kernel == CpuAccumulatorKernel::ExactInteger) {
                std::vector<Index> row_key(tensor_view.leaf_rank(), 0);
                std::vector<Index> input_key(tensor_view.leaf_rank(), 0);
                std::vector<Index> output_key(result.view.leaf_rank(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    input_key = row_key;
                    ExactIntegerSum sum;
                    for (Index j = 0; j < m_size; ++j) {
                        sum.add(tensor_view.read_int_expanded(input_key));
                        tensor_view.cache->increment_mode(input_key.data(),
                                                          input_key.size(), 1);
                    }
                    write_accumulated_int(result.view, output_key, sum);
                    tensor_view.cache->increment_mode(row_key.data(), row_key.size(),
                                                      0);
                    result.view.cache->increment_key(output_key.data(),
                                                     output_key.size());
                }
            } else if (accumulator_kernel == CpuAccumulatorKernel::Float32) {
                reduce_floating<float>(tensor_view, result.view, n_size, m_size);
            } else {
                reduce_floating<double>(tensor_view, result.view, n_size, m_size);
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

constexpr CpuKernelMetadata kMetadata{"reduce_sum", "cpu.reduce_sum", "default",
                                      "_CPUReduceSumOperation"};

}  // namespace

void register_cpu_reduce_sum(py::module_& module) {
    bind_and_register_cpu_operation<CpuReduceSumOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
