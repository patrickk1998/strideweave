#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

struct CpuErfScalar {
    static constexpr const char* kOperation = "erf";
    static constexpr const char* kForwardError = "CPU erf requires a tensor";
    static float value(float x) { return std::erf(x); }
    static float gradient_multiplier(float x) {
        // Generic's NumPy binary32 scalar forces the 2/sqrt(pi) constant to
        // Float32 before multiplying the binary32 exponential.
        const float coefficient =
            static_cast<float>(2.0 / std::sqrt(3.14159265358979323846));
        return coefficient * std::exp(-(x * x));
    }
};

using CpuOperation = CpuUnaryElementwiseOperation<CpuErfScalar>;

constexpr CpuKernelMetadata kMetadata{"erf", "cpu.erf", "default", "_CPUErfOperation"};

}  // namespace

void register_cpu_erf(py::module_& module) {
    bind_and_register_cpu_operation<CpuOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
