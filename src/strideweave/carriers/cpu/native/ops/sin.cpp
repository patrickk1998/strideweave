#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSinScalar {
    static constexpr const char* kOperation = "sin";
    static constexpr const char* kForwardError = "CPU sin requires a tensor";
    static float value(float x) { return std::sin(x); }
    static float gradient_multiplier(float x) { return std::cos(x); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuSinScalar>;

constexpr CpuKernelMetadata kMetadata{"sin", "cpu.sin", "default", "_CPUSinOperation"};

}  // namespace

void register_cpu_sin(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
