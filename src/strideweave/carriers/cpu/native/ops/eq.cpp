#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

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

constexpr CpuKernelMetadata kMetadata{"eq", "cpu.eq", "default", "_CPUEqOperation"};

}  // namespace

void register_cpu_eq(py::module_& module) {
    bind_and_register_cpu_operation<CpuEqOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
