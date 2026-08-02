#include <cmath>

#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuExpScalar {
    static constexpr const char* kOperation = "exp";
    static constexpr const char* kForwardError = "CPU exp requires a tensor";
    static float value(float input) { return std::exp(input); }
    static float gradient_multiplier(float input) { return std::exp(input); }
};

using CpuExpOperation = CpuUnaryElementwiseOperation<CpuExpScalar>;
constexpr CpuKernelMetadata kMetadata{"exp", "cpu.exp", "default", "_CPUExpOperation"};

}  // namespace

void register_cpu_exp(py::module_& module) {
    bind_and_register_cpu_operation<CpuExpOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
