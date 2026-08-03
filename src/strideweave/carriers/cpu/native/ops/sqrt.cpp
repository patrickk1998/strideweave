#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSqrtScalar {
    static constexpr const char* kOperation = "sqrt";
    static constexpr const char* kForwardError = "CPU sqrt requires a tensor";
    static float value(float x) { return std::sqrt(x); }
    static float gradient_multiplier(float x) { return 1.0f / (2.0f * std::sqrt(x)); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuSqrtScalar>;

constexpr CpuKernelMetadata kMetadata{"sqrt", "cpu.sqrt", "default",
                                      "_CPUSqrtOperation"};

}  // namespace

void register_cpu_sqrt(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
