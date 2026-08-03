#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

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

constexpr CpuKernelMetadata kMetadata{"le", "cpu.le", "default", "_CPULeOperation"};

}  // namespace

void register_cpu_le(py::module_& module) {
    bind_and_register_cpu_operation<CpuLeOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
