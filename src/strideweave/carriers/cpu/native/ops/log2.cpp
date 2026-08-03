#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuLog2Scalar {
    static constexpr const char* kOperation = "log2";
    static constexpr const char* kForwardError = "CPU log2 requires a tensor";
    static float value(float x) { return std::log2(x); }
    static float gradient_multiplier(float x) { return 1.0f / (x * std::log(2.0f)); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuLog2Scalar>;

constexpr CpuKernelMetadata kMetadata{"log2", "cpu.log2", "default",
                                      "_CPULog2Operation"};

}  // namespace

void register_cpu_log2(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
