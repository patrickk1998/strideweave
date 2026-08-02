#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace strideweave::carrier {

void register_cpu_add(py::module_& module);
void register_cpu_div(py::module_& module);
void register_cpu_elu(py::module_& module);
void register_cpu_elementwise_mul(py::module_& module);
void register_cpu_exp(py::module_& module);
void register_cpu_gelu(py::module_& module);
void register_cpu_leaky_relu(py::module_& module);
void register_cpu_matmul(py::module_& module);
void register_cpu_scalar_mul(py::module_& module);
void register_cpu_pow(py::module_& module);
void register_cpu_reduce(py::module_& module);
void register_cpu_relu(py::module_& module);
void register_cpu_sigmoid(py::module_& module);
void register_cpu_silu(py::module_& module);
void register_cpu_softplus(py::module_& module);
void register_cpu_sub(py::module_& module);
void register_cpu_tanh(py::module_& module);

}  // namespace strideweave::carrier
