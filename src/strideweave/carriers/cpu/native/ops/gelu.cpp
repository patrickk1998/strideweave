#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuGELUScalar {
    static constexpr const char* kOperation = "gelu";
    static constexpr const char* kForwardError = "CPU GELU requires a tensor";
    static float value(float input) { return gelu_value(input); }
    static float gradient_multiplier(float input) {
        return gelu_gradient_multiplier(input);
    }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuGELUScalar>;

constexpr CpuKernelMetadata kMetadata{"gelu", "cpu.gelu", "default",
                                      "_CPUGELUOperation"};

}  // namespace

void register_cpu_gelu(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
