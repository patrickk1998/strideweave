#pragma once

#include <stdexcept>

#include "_cpu_policy.hpp"

namespace strideweave::carrier {

enum class CpuAccumulatorKernel { ExactInteger, Float32, Float64 };

inline CpuAccumulatorKernel accumulator_kernel_for(const CpuPlan& plan) {
    if (plan.compute != CpuArithmetic::Binary32 &&
        plan.accumulation == CpuAccumulation::ExactInteger &&
        plan.accumulator_dtype == CpuAccumulatorDType::None &&
        plan.output == CpuDType::Int32) {
        return CpuAccumulatorKernel::ExactInteger;
    }
    if (plan.compute == CpuArithmetic::Binary32 &&
        plan.accumulation == CpuAccumulation::Floating &&
        plan.output == CpuDType::Float32) {
        if (plan.accumulator_dtype == CpuAccumulatorDType::Float32) {
            return CpuAccumulatorKernel::Float32;
        }
        if (plan.accumulator_dtype == CpuAccumulatorDType::Float64) {
            return CpuAccumulatorKernel::Float64;
        }
    }
    throw std::logic_error(
        "CPU capability lowered to an incoherent accumulator kernel");
}

template <typename Accumulator>
inline Accumulator accumulate_float(Accumulator sum, float term) {
    return sum + static_cast<Accumulator>(term);
}

template <typename Accumulator>
inline Accumulator accumulate_binary32_product(Accumulator sum, float lhs, float rhs) {
    const float product = lhs * rhs;
    return sum + static_cast<Accumulator>(product);
}

template <typename Accumulator>
inline float store_float_accumulator(Accumulator value) {
    return static_cast<float>(value);
}

}  // namespace strideweave::carrier
