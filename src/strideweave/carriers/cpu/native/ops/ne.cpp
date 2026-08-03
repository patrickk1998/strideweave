#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

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

constexpr CpuKernelMetadata kMetadata{"ne", "cpu.ne", "default", "_CPUNeOperation"};

}  // namespace

void register_cpu_ne(py::module_& module) {
    bind_and_register_cpu_operation<CpuNeOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
