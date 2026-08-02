#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSiLUScalar {
    static constexpr const char* kOperation = "silu";
    static constexpr const char* kForwardError = "CPU SiLU requires a tensor";
    static float value(float input) { return input * sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid + input * sigmoid * (1.0f - sigmoid);
    }
};

using CpuSiLUOperation = CpuUnaryElementwiseOperation<CpuSiLUScalar>;
constexpr CpuKernelMetadata kMetadata{"silu", "cpu.silu", "default",
                                      "_CPUSiLUOperation"};

}  // namespace

void register_cpu_silu(py::module_& module) {
    bind_and_register_cpu_operation<CpuSiLUOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
