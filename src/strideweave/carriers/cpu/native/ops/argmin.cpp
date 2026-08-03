#include "_cpu_reduction.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuArgminKind {
    static constexpr bool kProd = false, kMax = false, kArg = true;
    static constexpr const char* operation() { return "argmin"; }
    static constexpr const char* error_message() {
        return "CPU argmin requires a tensor";
    }
};

using CpuOperation = CpuFloatReductionOperation<CpuArgminKind>;

constexpr CpuKernelMetadata kMetadata{"argmin", "cpu.argmin", "default",
                                      "_CPUArgminOperation"};

}  // namespace

void register_cpu_argmin(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
