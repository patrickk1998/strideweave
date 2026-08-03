#include "_cpu_extrema.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

using CpuOperation = CpuExtremaOperation<true>;

constexpr CpuKernelMetadata kMetadata{"maximum", "cpu.maximum", "default",
                                      "_CPUMaximumOperation"};

}  // namespace

void register_cpu_maximum(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
