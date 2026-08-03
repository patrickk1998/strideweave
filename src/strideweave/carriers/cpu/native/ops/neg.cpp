#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuNegScalar {
    static constexpr const char* kOperation = "neg";
    static constexpr const char* kForwardError = "CPU neg requires a tensor";
    static float value(float x) { return -x; }
    static float gradient_multiplier(float) { return -1.0f; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuNegScalar>;

constexpr CpuKernelMetadata kMetadata{"neg", "cpu.neg", "default", "_CPUNegOperation"};

}  // namespace

void register_cpu_neg(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
