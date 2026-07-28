#pragma once

// The CPU bridge to the shared dtype policy and to this backend's declared
// capabilities.
//
// Every promotion, compute, accumulation and result-dtype decision a CPU
// operation makes comes from strideweave.carriers.operation_policy through
// resolve_cpu_plan; this backend holds no policy table of its own (RT013).
// Whether CPU may then execute the resolved plan is decided by the capabilities
// its carrier class declared in strideweave.carriers.cpu.capabilities, consulted
// through the shared registry: execution and introspection read the same
// entries, and there is no native mirror of them to drift.
//
// Both steps happen while the GIL is still held, before the result is allocated
// and before the kernel releases it (CPP001), and neither inspects an element,
// so an unsupported plan fails before any storage exists and never falls through
// to a branch written for a different shape.
//
// The Python objects the bridge consults repeatedly — the resolver, the
// capability gate, the dtype singletons, the policy enum members and the Tensor
// type — are imported once into cpu_bindings() rather than re-looked-up per
// operation.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <utility>

namespace py = pybind11;

namespace strideweave::carrier {

enum class CpuDType { Float32, Int32 };

// Mirrors strideweave.carriers.operation_policy.Arithmetic. Binary32 is
// IEEE-754 binary32; Int32ExactChecked evaluates exactly over the integers and
// raises when the result leaves Int32; Int32Exact is exact integer arithmetic
// whose result provably fits, so it needs no per-element check.
enum class CpuArithmetic { Binary32, Int32ExactChecked, Int32Exact };

// Mirrors strideweave.carriers.operation_policy.Accumulation, plus None for an
// operation that combines no terms.
enum class CpuAccumulation { None, SequentialBinary32, ExactInteger };

struct CpuBindings {
    py::object resolve_operation_plan;
    py::object require_capability;
    py::object tensor_type;
    py::object compound_dtype;
    py::object float32;
    py::object int32;
    py::object binary32;
    py::object int32_exact_checked;
    py::object int32_exact;
    py::object sequential_binary32;
    py::object exact_integer;
};

// Imported once on first use and deliberately never freed: these are
// module-level singletons that outlive every operation, and a static
// py::object destructor would run after the interpreter has finalized.
inline const CpuBindings& cpu_bindings() {
    static const CpuBindings* bindings = [] {
        py::object carriers = py::module_::import("strideweave.carriers");
        py::object dtype = carriers.attr("DType");
        py::object policy =
            py::module_::import("strideweave.carriers.operation_policy");
        py::object arithmetic = policy.attr("Arithmetic");
        py::object accumulation = policy.attr("Accumulation");
        py::object capability =
            py::module_::import("strideweave.carriers.operation_capability");
        return new CpuBindings{
            policy.attr("resolve_operation_plan"),
            capability.attr("require_capability"),
            py::module_::import("strideweave.tensor").attr("Tensor"),
            carriers.attr("CompoundDType"),
            dtype.attr("Float32"),
            dtype.attr("Int32"),
            arithmetic.attr("BINARY32"),
            arithmetic.attr("INT32_EXACT_CHECKED"),
            arithmetic.attr("INT32_EXACT"),
            accumulation.attr("SEQUENTIAL_BINARY32"),
            accumulation.attr("EXACT_INTEGER"),
        };
    }();
    return *bindings;
}

inline py::object tensor_type() {
    return cpu_bindings().tensor_type;
}

// Mirrors the diagnostic that Python carriers raise through
// strideweave.carriers.dtype.validate_storage_dtype, so every carrier explains
// deferred compound storage the same way.
[[noreturn]] inline void throw_compound_dtype_error(py::handle dtype) {
    const std::string name = py::cast<std::string>(dtype.attr("name"));
    throw py::value_error("CPU cannot store compound dtype '" + name +
                          "': a carrier holds one simple dtype, and a compound "
                          "representation needs one carrier per simple_types "
                          "plane, which is not implemented");
}

// Dtype tags are identity values (SW002), and RT012 makes each carrier's
// accepted set exact, so recognition here is pointer identity against the
// registered singletons. Equality is deliberately never consulted: an object
// with a spoofed, raising, or side-effecting __eq__ must be rejected like any
// other unsupported dtype rather than deciding what this carrier stores.
inline CpuDType parse_cpu_dtype(py::handle dtype) {
    const CpuBindings& bindings = cpu_bindings();
    if (dtype.is_none() || dtype.is(bindings.float32)) {
        return CpuDType::Float32;
    }
    if (dtype.is(bindings.int32)) {
        return CpuDType::Int32;
    }
    if (py::isinstance(dtype, bindings.compound_dtype)) {
        throw_compound_dtype_error(dtype);
    }
    throw py::value_error("CPU dtype must be DType.Float32 or DType.Int32");
}

inline py::object cpu_dtype_object(CpuDType dtype) {
    const CpuBindings& bindings = cpu_bindings();
    return dtype == CpuDType::Float32 ? bindings.float32 : bindings.int32;
}

// One operation's resolved policy, in the form the kernels consume.
struct CpuPlan {
    CpuArithmetic compute;
    CpuAccumulation accumulation;
    CpuDType output;

    // Whether this operation's elements are computed as integers. The plan's
    // per-operand conversions agree with this by construction; see
    // require_uniform_conversion.
    bool is_integer() const { return compute != CpuArithmetic::Binary32; }
};

inline CpuArithmetic parse_cpu_arithmetic(py::handle value) {
    const CpuBindings& bindings = cpu_bindings();
    if (value.is(bindings.binary32)) {
        return CpuArithmetic::Binary32;
    }
    if (value.is(bindings.int32_exact_checked)) {
        return CpuArithmetic::Int32ExactChecked;
    }
    if (value.is(bindings.int32_exact)) {
        return CpuArithmetic::Int32Exact;
    }
    throw py::value_error("CPU does not implement this plan's compute arithmetic");
}

inline CpuAccumulation parse_cpu_accumulation(py::handle value) {
    const CpuBindings& bindings = cpu_bindings();
    if (value.is_none()) {
        return CpuAccumulation::None;
    }
    if (value.is(bindings.sequential_binary32)) {
        return CpuAccumulation::SequentialBinary32;
    }
    if (value.is(bindings.exact_integer)) {
        return CpuAccumulation::ExactInteger;
    }
    throw py::value_error("CPU does not implement this plan's accumulation");
}

// The exact carrier class whose declared capabilities decide whether this
// operation may run. Dispatch reaches an operation through one carrier, so a CPU
// subclass is asked about its own declarations rather than its base's.
inline py::object executing_carrier_class(py::handle tensor) {
    py::object carrier = tensor.attr("carrier");
    return py::type::of(carrier);
}

// Lower an accepted capability to the enums the kernels switch on. The
// capability is the backend's own declaration, so a member it names that this
// bridge cannot lower is a declaration bug rather than an unsupported plan, and
// it is reported here instead of being executed as something else.
inline CpuPlan cpu_plan_from_capability(py::handle capability) {
    const CpuArithmetic compute = parse_cpu_arithmetic(capability.attr("compute"));
    const CpuAccumulation accumulation =
        parse_cpu_accumulation(capability.attr("accumulation"));
    const CpuDType output = parse_cpu_dtype(capability.attr("output"));
    return CpuPlan{compute, accumulation, output};
}

// Resolve one operation's plan and require a capability for it. Tensor operands
// are passed as their storage dtype object and weak scalar operands as the
// Python value itself, in the operation's own argument order. Callers must still
// hold the GIL, and must not have allocated a result yet.
template <typename... Operands>
inline CpuPlan resolve_cpu_plan(py::handle carrier_class, const char* operation,
                                Operands&&... operands) {
    const CpuBindings& bindings = cpu_bindings();
    py::object plan =
        bindings.resolve_operation_plan(operation, std::forward<Operands>(operands)...);
    return cpu_plan_from_capability(bindings.require_capability(carrier_class, plan));
}

}  // namespace strideweave::carrier
