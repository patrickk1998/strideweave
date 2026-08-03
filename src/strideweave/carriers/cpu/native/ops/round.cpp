#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuRoundScalar {
    static constexpr const char* kOperation = "round";
    static constexpr const char* kForwardError = "CPU round requires a tensor";
    static float value(float x) { return std::nearbyint(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuRoundScalar>;

constexpr CpuKernelMetadata kMetadata{"round", "cpu.round", "default",
                                      "_CPURoundOperation"};

}  // namespace

void register_cpu_round(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
