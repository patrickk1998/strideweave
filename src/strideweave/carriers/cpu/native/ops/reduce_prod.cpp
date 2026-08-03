#include "_cpu_reduction.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuReduceProdKind {
    static constexpr bool kProd = true;
    [[maybe_unused]] static constexpr bool kMax = false;
    static constexpr bool kArg = false;
    static constexpr const char* operation() { return "reduce_prod"; }
    static constexpr const char* error_message() {
        return "CPU reduce_prod requires a tensor";
    }
};

using CpuOperation = CpuFloatReductionOperation<CpuReduceProdKind>;

constexpr CpuKernelMetadata kMetadata{"reduce_prod", "cpu.reduce_prod", "default",
                                      "_CPUReduceProdOperation"};

}  // namespace

void register_cpu_reduce_prod(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
