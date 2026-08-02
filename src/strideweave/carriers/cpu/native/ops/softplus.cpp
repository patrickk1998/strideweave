#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSoftplusScalar {
    static constexpr const char* kOperation = "softplus";
    static constexpr const char* kForwardError = "CPU softplus requires a tensor";
    static float value(float input) { return softplus_value(input); }
    static float gradient_multiplier(float input) { return sigmoid_value(input); }
};

using CpuSoftplusOperation = CpuUnaryElementwiseOperation<CpuSoftplusScalar>;
constexpr CpuKernelMetadata kMetadata{"softplus", "cpu.softplus", "default",
                                      "_CPUSoftplusOperation"};

}  // namespace

void register_cpu_softplus(py::module_& module) {
    bind_and_register_cpu_operation<CpuSoftplusOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
