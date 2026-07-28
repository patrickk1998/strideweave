#include <pybind11/pybind11.h>

#include <cstddef>
#include <limits>

#include "_carrier.hpp"
#include "_operation.hpp"

namespace py = pybind11;

namespace {

using strideweave::carrier::Carrier;
using strideweave::carrier::Index;

std::size_t as_size(Index value) {
    if (value < 0) {
        throw py::value_error("Carrier index must be non-negative");
    }
    if constexpr (sizeof(Index) > sizeof(std::size_t)) {
        if (value > static_cast<Index>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("Carrier index does not fit in size_t");
        }
    }
    return static_cast<std::size_t>(value);
}

class PyCarrier : public Carrier {
public:
    using Carrier::Carrier;

    Index size() const override { PYBIND11_OVERRIDE_PURE(Index, Carrier, size); }

    py::object dtype() const override {
        PYBIND11_OVERRIDE_PURE(py::object, Carrier, dtype);
    }

    py::object get_value(Index index) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Carrier, get_value, index);
    }

    py::object new_like(py::iterable values, bool is_mutable) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Carrier, new_like, values, is_mutable);
    }

    py::object allocate_like(Index size, bool is_mutable, py::object dtype,
                             bool empty) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Carrier, allocate_like, size, is_mutable,
                               dtype, empty);
    }

    void scatter(py::object to_scatter, py::object scatter_onto, py::object mapping,
                 Index mapping_offset) override {
        PYBIND11_OVERRIDE_PURE(void, Carrier, scatter, to_scatter, scatter_onto,
                               mapping, mapping_offset);
    }

    py::dict dlpack_info() const override {
        PYBIND11_OVERRIDE(py::dict, Carrier, dlpack_info);
    }

protected:
    py::object _dispatch_op(const std::string& operation_name) const override {
        PYBIND11_OVERRIDE_NAME(py::object, Carrier, "_dispatch_op", _dispatch_op,
                               operation_name);
    }

    bool _is_mutable() const override {
        PYBIND11_OVERRIDE_NAME(bool, Carrier, "_is_mutable", _is_mutable);
    }

    bool _supports_storage_dtype(py::object dtype) const override {
        PYBIND11_OVERRIDE_NAME(bool, Carrier, "_supports_storage_dtype",
                               _supports_storage_dtype, dtype);
    }

    void set_value(Index index, py::object value) override {
        PYBIND11_OVERRIDE(void, Carrier, set_value, index, value);
    }

    void _release() override { PYBIND11_OVERRIDE(void, Carrier, _release); }
};

class VectorCarrierForTest : public Carrier {
public:
    explicit VectorCarrierForTest(py::iterable values) : values_(py::list(values)) {}

    Index size() const override { return static_cast<Index>(py::len(values_)); }

    py::object dtype() const override {
        return py::module_::import("strideweave.carriers").attr("DType").attr("Any");
    }

    py::object get_value(Index index) const override {
        return py::reinterpret_borrow<py::object>(values_[as_size(index)]);
    }

    py::object new_like(py::iterable values, bool) const override {
        return py::cast(VectorCarrierForTest(values));
    }

    py::object allocate_like(Index size, bool, py::object, bool) const override {
        if (size < 0) {
            throw py::value_error("Carrier allocation size must be non-negative");
        }
        py::list values(as_size(size));
        for (std::size_t i = 0; i < as_size(size); ++i) {
            values[i] = py::none();
        }
        return py::cast(VectorCarrierForTest(values));
    }

    void scatter(py::object, py::object, py::object, Index) override {
        throw py::type_error("_VectorCarrierForTest does not implement scatter");
    }

private:
    py::list values_;
};

// The read-only view of a backend's declared operation-plan capabilities. The
// module is imported once and deliberately never freed: it is a process-lifetime
// singleton, and a static py::object destructor would run after the interpreter
// has finalized.
const py::object& capability_api() {
    static const py::object* api = new py::object(
        py::module_::import("strideweave.carriers.operation_capability"));
    return *api;
}

// The dtype model, imported once for the same reason the capability module is.
const py::object& dtype_api() {
    static const py::object* api =
        new py::object(py::module_::import("strideweave.carriers.dtype"));
    return *api;
}

// Capabilities are owned either by an exact carrier class or by one constructed
// dependent instance, so a query is asked of the carrier itself and the
// capability module selects the owner. Casting back reaches the carrier's own
// Python object, so a Python subclass instance keeps its exact identity.
py::object carrier_object(const Carrier& carrier) {
    return py::cast(&carrier, py::return_value_policy::reference);
}

}  // namespace

bool strideweave::carrier::Carrier::supports_storage_dtype(py::object dtype) const {
    if (!py::isinstance(dtype, dtype_api().attr("DType"))) {
        throw py::type_error("storage dtype must be a DType");
    }
    return _supports_storage_dtype(std::move(dtype));
}

py::object
strideweave::carrier::Carrier::dispatch_op(const std::string& operation_name) const {
    py::object operation = _dispatch_op(operation_name);
    if (!py::isinstance<strideweave::operation::Operation>(operation)) {
        throw py::type_error("Carrier._dispatch_op must return an Operation");
    }

    auto& typed_operation = operation.cast<strideweave::operation::Operation&>();
    if (typed_operation.is_dispatched()) {
        throw py::type_error(
            "Carrier._dispatch_op must return a fresh Operation instance");
    }
    py::object carrier = py::cast(this, py::return_value_policy::reference);
    typed_operation.set_dispatch_metadata(operation_name, py::type::of(carrier));
    return operation;
}

PYBIND11_MODULE(_carrier, module) {
    module.doc() = "Native carrier base classes for StrideWeave";

    py::class_<Carrier, PyCarrier>(module, "Carrier")
        .def(py::init<>())
        .def("size", &Carrier::size)
        .def("dtype", &Carrier::dtype)
        .def("get_value", &Carrier::get_value, py::arg("index"))
        .def("new_like", &Carrier::new_like, py::arg("values"), py::kw_only(),
             py::arg("mutable") = true)
        .def("allocate_like", &Carrier::allocate_like, py::arg("size"), py::kw_only(),
             py::arg("mutable") = true, py::arg("dtype") = py::none(),
             py::arg("empty") = false)
        .def("scatter", &Carrier::scatter, py::arg("to_scatter"),
             py::arg("scatter_onto"), py::arg("mapping"), py::arg("mapping_offset") = 0)
        .def(
            "is_mutable", &Carrier::is_mutable,
            "Return whether public carrier interfaces may currently modify storage.\n\n"
            "This combines the carrier's intrinsic mutability with ownership. "
            "A carrier constructed as immutable always returns false. A mutable "
            "carrier "
            "also returns false while it is exclusively owned by another carrier, "
            "except during the owner's private access scope.\n\n"
            "Returns:\n"
            "    True when public mutation is currently permitted; otherwise "
            "False.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> carrier = sw.Generic([1.0], mutable=False)\n"
            "    >>> carrier.is_mutable()\n"
            "    False")
        .def("is_owned", &Carrier::is_owned,
             "Return whether another carrier exclusively owns this storage.\n\n"
             "An owned carrier remains readable while live, but public mutation, "
             "scatter, release, version increments, and direct moves are "
             "rejected. The owner retains private access for those operations.\n\n"
             "Returns:\n"
             "    True when this carrier has an exclusive owner; otherwise False.\n\n"
             "Examples:\n"
             "    >>> import strideweave as sw\n"
             "    >>> primary = sw.Generic([1.0])\n"
             "    >>> hierarchy = sw.Evictable(\n"
             "    ...     primary, sw.Generic([0.0])\n"
             "    ... )\n"
             "    >>> primary.is_owned()\n"
             "    True")
        .def("supports_storage_dtype", &Carrier::supports_storage_dtype,
             py::arg("dtype"),
             "Report whether this carrier implementation can store a dtype.\n\n"
             "This is a structural question about the implementation, not about "
             "this instance's current state: it allocates nothing, changes "
             "nothing, and is unaffected by size, mutability, ownership, "
             "eviction residency, release, or which dtype the carrier currently "
             "holds. A carrier composing others reports what every "
             "representation it must build can store, which is what lets it "
             "decide whether it could hold an operation's result before any "
             "work begins.\n\n"
             "Descriptors are registry singletons, so a dtype is recognized by "
             "identity; an object that merely compares equal to one is not that "
             "dtype. Support is narrower than being a valid descriptor: "
             "abstract categories no carrier stores, simple encodings no backend "
             "implements, and compound descriptors, whose per-plane storage is "
             "deferred, are all unsupported rather than errors.\n\n"
             "Args:\n"
             "    dtype: The DType to ask about.\n\n"
             "Returns:\n"
             "    True when this carrier implementation can allocate that "
             "dtype.\n\n"
             "Raises:\n"
             "    TypeError: If dtype is not a DType.\n\n"
             "Examples:\n"
             "    >>> import strideweave as sw\n"
             "    >>> carrier = sw.CPU(1, dtype=sw.DType.Float32)\n"
             "    >>> carrier.supports_storage_dtype(sw.DType.Int32)\n"
             "    True\n"
             "    >>> carrier.supports_storage_dtype(sw.DType.Integer)\n"
             "    False")
        .def("_has_owner_access", &Carrier::has_owner_access)
        .def("_claim_ownership", &Carrier::claim_ownership)
        .def("_relinquish_ownership", &Carrier::relinquish_ownership, py::arg("token"))
        .def("_begin_owner_access", &Carrier::begin_owner_access, py::arg("token"))
        .def("_end_owner_access", &Carrier::end_owner_access, py::arg("token"))
        .def("dlpack_info", &Carrier::dlpack_info)
        .def_property_readonly("version", &Carrier::version)
        .def("_increment_version", &Carrier::increment_version)
        .def("is_released", &Carrier::is_released)
        .def("release", &Carrier::release,
             "Release the carrier's storage; further element access raises.\n\n"
             "Contract: ``new_like`` must remain usable after ``release()`` --\n"
             "it constructs fresh storage and reads nothing from the released\n"
             "instance. Move's backward pass relies on this to materialize\n"
             "gradients in a released source carrier.")
        .def("dispatch_op", &Carrier::dispatch_op, py::arg("operation_name"))
        .def(
            "operation_capabilities",
            [](const Carrier& self, py::object operation_name) {
                return capability_api().attr("carrier_operation_capabilities")(
                    carrier_object(self), operation_name);
            },
            py::arg("operation_name") = py::none(),
            "Return the operation plans this carrier executes.\n\n"
            "Support means faithful execution of an already-resolved plan from "
            "strideweave.carriers.operation_policy. It is not a claim about "
            "promotion: which plan is correct for a set of operand dtypes is "
            "central policy that no backend owns. Both the policy and the "
            "capability surface are evolvable rather than a compatibility "
            "promise.\n\n"
            "The answer comes from whichever set owns this carrier: the sealed "
            "declarations of its exact class for an independent carrier, or "
            "the frozen snapshot a DependentCarrier generated for this "
            "instance. A caller asks the carrier and does not need to know "
            "which.\n\n"
            "Args:\n"
            "    operation_name: Restrict the result to one operation, or None "
            "for every operation this backend executes.\n\n"
            "Returns:\n"
            "    Immutable OperationCapability descriptors in a deterministic "
            "order, the same entries execution is accepted against.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> [entry.output.name for entry in "
            "sw.CPU(1).operation_capabilities('relu')]\n"
            "    ['Float32', 'Int32']\n")
        .def(
            "supports_operation_plan",
            [](const Carrier& self, py::object plan) {
                return capability_api().attr("carrier_supports_operation_plan")(
                    carrier_object(self), plan);
            },
            py::arg("plan"),
            "Report whether this carrier executes a resolved plan.\n\n"
            "The decision is the one enforcement makes, read from this "
            "carrier's own capabilities.\n\n"
            "Args:\n"
            "    plan: An OperationPlan resolved by "
            "strideweave.carriers.operation_policy.\n\n"
            "Returns:\n"
            "    True when a declared capability matches the plan exactly, "
            "including its operand conversions and whether it accumulates.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> from strideweave.carriers.operation_policy import (\n"
            "    ...     resolve_operation_plan,\n"
            "    ... )\n"
            "    >>> plan = resolve_operation_plan('relu', sw.DType.Int32)\n"
            "    >>> sw.CPU(1).supports_operation_plan(plan)\n"
            "    True\n")
        .def(
            "unsupported_plan_reason",
            [](const Carrier& self, py::object plan) {
                return capability_api().attr("carrier_unsupported_plan_reason")(
                    carrier_object(self), plan);
            },
            py::arg("plan"),
            "Explain why this carrier cannot execute a resolved plan.\n\n"
            "The reason distinguishes an operation this carrier has nothing "
            "for from an operation it has other shapes of.\n\n"
            "Args:\n"
            "    plan: An OperationPlan resolved by "
            "strideweave.carriers.operation_policy.\n\n"
            "Returns:\n"
            "    A stable explanatory sentence, or None when the plan is "
            "supported.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> from strideweave.carriers.operation_policy import (\n"
            "    ...     resolve_operation_plan,\n"
            "    ... )\n"
            "    >>> plan = resolve_operation_plan('relu', sw.DType.Int32)\n"
            "    >>> sw.CPU(1).unsupported_plan_reason(plan) is None\n"
            "    True\n")
        .def(
            "require_operation_plan",
            [](const Carrier& self, py::object plan) {
                return capability_api().attr("require_carrier_capability")(
                    carrier_object(self), plan);
            },
            py::arg("plan"),
            "Return the capability a resolved plan executes under, or refuse "
            "it.\n\n"
            "This is the acceptance gate operations use, exposed so a caller can "
            "ask for the same decision without running a kernel.\n\n"
            "Args:\n"
            "    plan: An OperationPlan resolved by "
            "strideweave.carriers.operation_policy.\n\n"
            "Returns:\n"
            "    The matching immutable OperationCapability.\n\n"
            "Raises:\n"
            "    UnsupportedOperationPlan: If this carrier has no capability "
            "for that exact shape.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> from strideweave.carriers.operation_policy import (\n"
            "    ...     resolve_operation_plan,\n"
            "    ... )\n"
            "    >>> plan = resolve_operation_plan('relu', sw.DType.Int32)\n"
            "    >>> sw.CPU(1).require_operation_plan(plan).compute.value\n"
            "    'int32_exact'\n")
        .def("__getitem__", &Carrier::get_item, py::arg("index"))
        .def("__setitem__", &Carrier::set_item, py::arg("index"), py::arg("value"));

    py::class_<VectorCarrierForTest, Carrier>(module, "_VectorCarrierForTest")
        .def(py::init<py::iterable>(), py::arg("values"));

    py::module_::import("strideweave._operation");
    strideweave::carrier::bind_cpu(module);
}
