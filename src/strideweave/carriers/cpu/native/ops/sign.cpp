#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuSignScalar {
    static constexpr const char* kOperation = "sign";
    static constexpr const char* kForwardError = "CPU sign requires a tensor";
    static float value(float x) {
        return std::isnan(x) ? std::numeric_limits<float>::quiet_NaN()
                             : (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f));
    }
    static float gradient_multiplier(float) { return 0.0f; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuSignScalar>;

constexpr CpuKernelMetadata kMetadata{"sign", "cpu.sign", "default",
                                      "_CPUSignOperation"};

}  // namespace

void register_cpu_sign(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
