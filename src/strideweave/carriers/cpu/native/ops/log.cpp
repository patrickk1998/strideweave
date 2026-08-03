#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuLogScalar {
    static constexpr const char* kOperation = "log";
    static constexpr const char* kForwardError = "CPU log requires a tensor";
    static float value(float x) { return std::log(x); }
    static float gradient_multiplier(float x) { return 1.0f / x; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuLogScalar>;

constexpr CpuKernelMetadata kMetadata{"log", "cpu.log", "default", "_CPULogOperation"};

}  // namespace

void register_cpu_log(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
