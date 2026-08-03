#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace strideweave::carrier {

// One registration entry point per operation-owned translation unit. Each
// implementation binds its operation type and registers exactly one kernel
// metadata record per dispatch name it owns.
void register_cpu_abs(py::module_& module);
void register_cpu_add(py::module_& module);
void register_cpu_argmax(py::module_& module);
void register_cpu_argmin(py::module_& module);
void register_cpu_ceil(py::module_& module);
void register_cpu_clamp(py::module_& module);
void register_cpu_conv_general(py::module_& module);
void register_cpu_cos(py::module_& module);
void register_cpu_cumsum(py::module_& module);
void register_cpu_div(py::module_& module);
void register_cpu_elementwise_mul(py::module_& module);
void register_cpu_elu(py::module_& module);
void register_cpu_eq(py::module_& module);
void register_cpu_erf(py::module_& module);
void register_cpu_exp(py::module_& module);
void register_cpu_exp2(py::module_& module);
void register_cpu_floor(py::module_& module);
void register_cpu_gather(py::module_& module);
void register_cpu_gelu(py::module_& module);
void register_cpu_le(py::module_& module);
void register_cpu_leaky_relu(py::module_& module);
void register_cpu_log(py::module_& module);
void register_cpu_log2(py::module_& module);
void register_cpu_logical_not(py::module_& module);
void register_cpu_lt(py::module_& module);
void register_cpu_matmul(py::module_& module);
void register_cpu_maximum(py::module_& module);
void register_cpu_minimum(py::module_& module);
void register_cpu_ne(py::module_& module);
void register_cpu_neg(py::module_& module);
void register_cpu_pow(py::module_& module);
void register_cpu_recip(py::module_& module);
void register_cpu_reduce_max(py::module_& module);
void register_cpu_reduce_min(py::module_& module);
void register_cpu_reduce_prod(py::module_& module);
void register_cpu_reduce_sum(py::module_& module);
void register_cpu_relu(py::module_& module);
void register_cpu_rem(py::module_& module);
void register_cpu_round(py::module_& module);
void register_cpu_rsqrt(py::module_& module);
void register_cpu_scalar_mul(py::module_& module);
void register_cpu_scatter(py::module_& module);
void register_cpu_select(py::module_& module);
void register_cpu_sigmoid(py::module_& module);
void register_cpu_sign(py::module_& module);
void register_cpu_silu(py::module_& module);
void register_cpu_sin(py::module_& module);
void register_cpu_softplus(py::module_& module);
void register_cpu_sort(py::module_& module);
void register_cpu_sqrt(py::module_& module);
void register_cpu_sub(py::module_& module);
void register_cpu_tanh(py::module_& module);

}  // namespace strideweave::carrier
