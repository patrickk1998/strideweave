#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuLeakyReLUScalar {
    static constexpr const char* kOperation = "leaky_relu";
    static constexpr const char* kForwardError = "CPU leaky ReLU requires a tensor";
    static float value(float input) {
        return input >= 0.0f ? input : kLeakyReluNegativeSlope * input;
    }
    static float gradient_multiplier(float input) {
        return input >= 0.0f ? 1.0f : kLeakyReluNegativeSlope;
    }
};

using CpuLeakyReLUOperation = CpuUnaryElementwiseOperation<CpuLeakyReLUScalar>;
constexpr CpuKernelMetadata kMetadata{"leaky_relu", "cpu.leaky_relu", "default",
                                      "_CPULeakyReLUOperation"};

}  // namespace

void register_cpu_leaky_relu(py::module_& module) {
    bind_and_register_cpu_operation<CpuLeakyReLUOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
