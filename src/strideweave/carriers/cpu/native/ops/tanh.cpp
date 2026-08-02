#include <cmath>

#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuTanhScalar {
    static constexpr const char* kOperation = "tanh";
    static constexpr const char* kForwardError = "CPU tanh requires a tensor";
    static float value(float input) { return std::tanh(input); }
    static float gradient_multiplier(float input) {
        const float output = std::tanh(input);
        return 1.0f - output * output;
    }
};

using CpuTanhOperation = CpuUnaryElementwiseOperation<CpuTanhScalar>;
constexpr CpuKernelMetadata kMetadata{"tanh", "cpu.tanh", "default",
                                      "_CPUTanhOperation"};

}  // namespace

void register_cpu_tanh(py::module_& module) {
    bind_and_register_cpu_operation<CpuTanhOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
