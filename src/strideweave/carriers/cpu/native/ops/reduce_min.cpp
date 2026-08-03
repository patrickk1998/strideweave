#include "_cpu_reduction.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuReduceMinKind {
    static constexpr bool kProd = false, kMax = false, kArg = false;
    static constexpr const char* operation() { return "reduce_min"; }
    static constexpr const char* error_message() {
        return "CPU reduce_min requires a tensor";
    }
};

using CpuOperation = CpuFloatReductionOperation<CpuReduceMinKind>;

constexpr CpuKernelMetadata kMetadata{"reduce_min", "cpu.reduce_min", "default",
                                      "_CPUReduceMinOperation"};

}  // namespace

void register_cpu_reduce_min(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
