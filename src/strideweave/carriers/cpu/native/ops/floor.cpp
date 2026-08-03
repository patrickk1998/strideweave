#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuFloorScalar {
    static constexpr const char* kOperation = "floor";
    static constexpr const char* kForwardError = "CPU floor requires a tensor";
    static float value(float x) { return std::floor(x); }
    static float gradient_multiplier(float) { return 0.0f; }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuFloorScalar>;

constexpr CpuKernelMetadata kMetadata{"floor", "cpu.floor", "default",
                                      "_CPUFloorOperation"};

}  // namespace

void register_cpu_floor(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
