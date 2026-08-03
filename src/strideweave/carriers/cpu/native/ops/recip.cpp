#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuRecipScalar {
    static constexpr const char* kOperation = "recip";
    static constexpr const char* kForwardError = "CPU recip requires a tensor";
    static float value(float x) { return 1.0f / x; }
    static float gradient_multiplier(float x) { return -1.0f / (x * x); }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuRecipScalar>;

constexpr CpuKernelMetadata kMetadata{"recip", "cpu.recip", "default",
                                      "_CPURecipOperation"};

}  // namespace

void register_cpu_recip(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
