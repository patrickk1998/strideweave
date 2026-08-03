#include "_cpu_reduction.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuArgmaxKind {
    static constexpr bool kProd = false, kMax = true, kArg = true;
    static constexpr const char* operation() { return "argmax"; }
    static constexpr const char* error_message() {
        return "CPU argmax requires a tensor";
    }
};

using CpuOperation = CpuFloatReductionOperation<CpuArgmaxKind>;

constexpr CpuKernelMetadata kMetadata{"argmax", "cpu.argmax", "default",
                                      "_CPUArgmaxOperation"};

}  // namespace

void register_cpu_argmax(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
