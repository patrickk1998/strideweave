#include "_cpu_reduction.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuReduceMaxKind {
    static constexpr bool kProd = false, kMax = true, kArg = false;
    static constexpr const char* operation() { return "reduce_max"; }
    static constexpr const char* error_message() {
        return "CPU reduce_max requires a tensor";
    }
};

using CpuOperation = CpuFloatReductionOperation<CpuReduceMaxKind>;

constexpr CpuKernelMetadata kMetadata{"reduce_max", "cpu.reduce_max", "default",
                                      "_CPUReduceMaxOperation"};

}  // namespace

void register_cpu_reduce_max(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
