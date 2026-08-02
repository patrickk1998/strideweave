#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

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

constexpr CpuKernelMetadata kMetadata{"add", "cpu.add", "default", "_CPUAddOperation"};

}  // namespace

void register_cpu_add(py::module_& module) {
    bind_and_register_cpu_operation<CpuAddOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
