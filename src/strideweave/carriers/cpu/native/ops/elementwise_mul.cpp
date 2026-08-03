#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

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

constexpr CpuKernelMetadata kMetadata{"elementwise_mul", "cpu.elementwise_mul",
                                      "default", "_CPUElementwiseMulOperation"};

}  // namespace

void register_cpu_elementwise_mul(py::module_& module) {
    bind_and_register_cpu_operation<CpuElementwiseMulOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
