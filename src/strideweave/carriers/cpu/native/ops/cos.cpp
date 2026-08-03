#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuCosScalar {
    static constexpr const char* kOperation = "cos";
    static constexpr const char* kForwardError = "CPU cos requires a tensor";
    static float value(float x) { return std::cos(x); }
    static float gradient_multiplier(float x) { return -std::sin(x); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuCosScalar>;

constexpr CpuKernelMetadata kMetadata{"cos", "cpu.cos", "default", "_CPUCosOperation"};

}  // namespace

void register_cpu_cos(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
