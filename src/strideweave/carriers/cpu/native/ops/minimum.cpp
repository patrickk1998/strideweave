#include "_cpu_extrema.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

using CpuOperation = CpuExtremaOperation<false>;

constexpr CpuKernelMetadata kMetadata{"minimum", "cpu.minimum", "default",
                                      "_CPUMinimumOperation"};

}  // namespace

void register_cpu_minimum(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
