#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuCeilScalar {
    static constexpr const char* kOperation = "ceil";
    static constexpr const char* kForwardError = "CPU ceil requires a tensor";
    static float value(float x) { return std::ceil(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuCeilScalar>;

constexpr CpuKernelMetadata kMetadata{"ceil", "cpu.ceil", "default",
                                      "_CPUCeilOperation"};

}  // namespace

void register_cpu_ceil(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
