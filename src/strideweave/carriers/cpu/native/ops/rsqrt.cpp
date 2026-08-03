#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuRsqrtScalar {
    static constexpr const char* kOperation = "rsqrt";
    static constexpr const char* kForwardError = "CPU rsqrt requires a tensor";
    static float value(float x) { return 1.0f / std::sqrt(x); }
    static float gradient_multiplier(float x) {
        const float y = value(x);
        return -0.5f * y * y * y;
    }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuRsqrtScalar>;

constexpr CpuKernelMetadata kMetadata{"rsqrt", "cpu.rsqrt", "default",
                                      "_CPURsqrtOperation"};

}  // namespace

void register_cpu_rsqrt(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
