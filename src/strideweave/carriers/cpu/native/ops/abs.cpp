#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuAbsScalar {
    static constexpr const char* kOperation = "abs";
    static constexpr const char* kForwardError = "CPU abs requires a tensor";
    static float value(float x) { return std::fabs(x); }
    static float gradient_multiplier(float x) {
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuAbsScalar>;

constexpr CpuKernelMetadata kMetadata{"abs", "cpu.abs", "default", "_CPUAbsOperation"};

}  // namespace

void register_cpu_abs(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
