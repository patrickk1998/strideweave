#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuExp2Scalar {
    static constexpr const char* kOperation = "exp2";
    static constexpr const char* kForwardError = "CPU exp2 requires a tensor";
    static float value(float x) { return std::exp2(x); }
    static float gradient_multiplier(float x) { return std::log(2.0f) * std::exp2(x); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuExp2Scalar>;

constexpr CpuKernelMetadata kMetadata{"exp2", "cpu.exp2", "default",
                                      "_CPUExp2Operation"};

}  // namespace

void register_cpu_exp2(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
