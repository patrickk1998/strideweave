#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSigmoidScalar {
    static constexpr const char* kOperation = "sigmoid";
    static constexpr const char* kForwardError = "CPU sigmoid requires a tensor";
    static float value(float input) { return sigmoid_value(input); }
    static float gradient_multiplier(float input) {
        const float sigmoid = sigmoid_value(input);
        return sigmoid * (1.0f - sigmoid);
    }
};

using CpuSigmoidOperation = CpuUnaryElementwiseOperation<CpuSigmoidScalar>;
constexpr CpuKernelMetadata kMetadata{"sigmoid", "cpu.sigmoid", "default",
                                      "_CPUSigmoidOperation"};

}  // namespace

void register_cpu_sigmoid(py::module_& module) {
    bind_and_register_cpu_operation<CpuSigmoidOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
