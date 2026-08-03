#include <pybind11/pybind11.h>

#include <string>

#include "_carrier.hpp"
#include "_cpu_carrier.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace py = pybind11;

namespace strideweave::carrier {
namespace {

// CPU is a closed carrier implementation, so there is no Python subclass for a
// trampoline to dispatch back into. The wording matches
// `strideweave.carriers.base.CLOSED_CARRIER_MESSAGE`, which
// `tests/test_carrier.py::test_a_closed_carrier_states_one_refusal` pins.
[[noreturn]] void reject_cpu_subclass() {
    throw py::type_error(
        "CPU is a closed carrier implementation and cannot be subclassed; "
        "implement a sibling Carrier instead, normally by composing existing "
        "carriers and lowering operations onto them the way Evictable does");
}

// Every numerical kernel lives in its own translation unit under `ops/`, which
// owns that operation's formula, stable kernel ID, binding, and registration.
// This file owns only carrier storage and module glue, so a kernel is never
// reachable without a registry record naming it.
void register_native_operations(py::module_& module) {
    register_cpu_abs(module);
    register_cpu_add(module);
    register_cpu_argmax(module);
    register_cpu_argmin(module);
    register_cpu_ceil(module);
    register_cpu_clamp(module);
    register_cpu_conv_general(module);
    register_cpu_cos(module);
    register_cpu_cumsum(module);
    register_cpu_div(module);
    register_cpu_elementwise_mul(module);
    register_cpu_elu(module);
    register_cpu_eq(module);
    register_cpu_erf(module);
    register_cpu_exp(module);
    register_cpu_exp2(module);
    register_cpu_floor(module);
    register_cpu_gather(module);
    register_cpu_gelu(module);
    register_cpu_le(module);
    register_cpu_leaky_relu(module);
    register_cpu_log(module);
    register_cpu_log2(module);
    register_cpu_logical_not(module);
    register_cpu_lt(module);
    register_cpu_matmul(module);
    register_cpu_maximum(module);
    register_cpu_minimum(module);
    register_cpu_ne(module);
    register_cpu_neg(module);
    register_cpu_pow(module);
    register_cpu_recip(module);
    register_cpu_reduce_max(module);
    register_cpu_reduce_min(module);
    register_cpu_reduce_prod(module);
    register_cpu_reduce_sum(module);
    register_cpu_relu(module);
    register_cpu_rem(module);
    register_cpu_round(module);
    register_cpu_rsqrt(module);
    register_cpu_scalar_mul(module);
    register_cpu_scatter(module);
    register_cpu_select(module);
    register_cpu_sigmoid(module);
    register_cpu_sign(module);
    register_cpu_silu(module);
    register_cpu_sin(module);
    register_cpu_softplus(module);
    register_cpu_sort(module);
    register_cpu_sqrt(module);
    register_cpu_sub(module);
    register_cpu_tanh(module);
}

// Structural operations preserve dtype and layout rather than computing, so
// they are implemented once in Python and carry no native kernel metadata.
void register_python_operations() {
    register_python_cpu_operation("broadcast_to", "strideweave.carriers.shared_ops",
                                  "BroadcastOperation");
    register_python_cpu_operation("permute", "strideweave.carriers.shared_ops",
                                  "PermuteOperation");
    register_python_cpu_operation("rearrange", "strideweave.carriers.shared_ops",
                                  "RearrangeOperation");
    register_python_cpu_operation("reshape", "strideweave.carriers.shared_ops",
                                  "ReshapeOperation");
    register_python_cpu_operation("as_strided",
                                  "strideweave.carriers.generic.as_strided_ops",
                                  "GenericAsStridedOperation");
    register_python_cpu_operation("squeeze", "strideweave.carriers.shared_ops",
                                  "SqueezeOperation");
    register_python_cpu_operation("unsqueeze", "strideweave.carriers.shared_ops",
                                  "UnsqueezeOperation");
    register_python_cpu_operation("view", "strideweave.carriers.shared_ops",
                                  "GenericViewOperation");
}

}  // namespace

py::object CPU::_dispatch_op(const std::string& operation_name) const {
    return make_registered_cpu_operation(operation_name);
}

void bind_cpu(py::module_& module) {
    py::class_<CPU, Carrier>(module, "CPU")
        // Closed through `__init_subclass__` rather than `py::is_final()` on
        // purpose. `is_final()` refuses at the C level, which is the stronger
        // guarantee, but it reports CPython's generic "not an acceptable base
        // type" instead of the wording every closed carrier shares. That shared
        // wording is the contract
        // `tests/test_carrier.py::test_a_closed_carrier_states_one_refusal`
        // pins, so switching to `is_final()` would break it.
        .def_static("__init_subclass__",
                    [](const py::args&, const py::kwargs&) { reject_cpu_subclass(); })
        .def(py::init<Index, py::object, bool, py::object, bool>(), py::arg("size"),
             py::arg("pointer") = py::none(), py::kw_only(), py::arg("mutable") = true,
             py::arg("dtype") = py::none(), py::arg("empty") = false)
        .def("new_like", &CPU::new_like_with_dtype, py::arg("values"), py::kw_only(),
             py::arg("mutable") = true, py::arg("dtype") = py::none())
        .def("allocate_like", &CPU::allocate_like, py::arg("size"), py::kw_only(),
             py::arg("mutable") = true, py::arg("dtype") = py::none(),
             py::arg("empty") = false)
        .def("_dispatch_op", &CPU::dispatch_registered_op, py::arg("operation_name"))
        .def("pointer", &CPU::pointer)
        .def("set_value", &CPU::set_value_public, py::arg("index"), py::arg("value"));

    register_native_operations(module);
    register_python_operations();
    bind_cpu_registry_introspection(module);
}

}  // namespace strideweave::carrier
