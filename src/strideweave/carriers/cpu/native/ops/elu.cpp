#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuELUScalar {
    static constexpr const char* kOperation = "elu";
    static constexpr const char* kForwardError = "CPU ELU requires a tensor";
    static float value(float input) { return input > 0.0f ? input : std::expm1(input); }
    static float gradient_multiplier(float input) {
        return input > 0.0f ? 1.0f : std::exp(input);
    }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuELUScalar>;

constexpr CpuKernelMetadata kMetadata{"elu", "cpu.elu", "default", "_CPUELUOperation"};

}  // namespace

void register_cpu_elu(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
