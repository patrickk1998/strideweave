#include <pybind11/pybind11.h>

#include "_carrier.hpp"

namespace py = pybind11;

namespace {

using strideweave::carrier::Carrier;
using strideweave::carrier::Index;

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

    py::object empty_like(
        Index size, bool is_mutable, py::object dtype
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            py::object, Carrier, empty_like, size, is_mutable, dtype
        );
    }

    void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) override {
        PYBIND11_OVERRIDE_PURE(
            void, Carrier, scatter, to_scatter, scatter_onto, mapping, mapping_offset
        );
    }

    py::dict dlpack_info() const override {
        PYBIND11_OVERRIDE(py::dict, Carrier, dlpack_info);
    }

    py::object dispatch_op(const std::string& operation_name) const override {
        PYBIND11_OVERRIDE(py::object, Carrier, dispatch_op, operation_name);
    }

protected:
    bool _is_mutable() const override {
        PYBIND11_OVERRIDE_NAME(bool, Carrier, "_is_mutable", _is_mutable);
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
        return py::reinterpret_borrow<py::object>(values_[index]);
    }

    py::object new_like(py::iterable values, bool) const override {
        return py::cast(VectorCarrierForTest(values));
    }

    py::object empty_like(Index size, bool, py::object) const override {
        if (size < 0) {
            throw py::value_error("Carrier allocation size must be non-negative");
        }
        py::list values(size);
        for (Index i = 0; i < size; ++i) {
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

}  // namespace

PYBIND11_MODULE(_carrier, module) {
    module.doc() = "Native carrier base classes for StrideWeave";

    py::class_<Carrier, PyCarrier>(module, "Carrier")
        .def(py::init<>())
        .def("size", &Carrier::size)
        .def("dtype", &Carrier::dtype)
        .def("get_value", &Carrier::get_value, py::arg("index"))
        .def(
            "new_like",
            &Carrier::new_like,
            py::arg("values"),
            py::kw_only(),
            py::arg("mutable") = true
        )
        .def(
            "empty_like",
            &Carrier::empty_like,
            py::arg("size"),
            py::kw_only(),
            py::arg("mutable") = true,
            py::arg("dtype") = py::none()
        )
        .def(
            "scatter",
            &Carrier::scatter,
            py::arg("to_scatter"),
            py::arg("scatter_onto"),
            py::arg("mapping"),
            py::arg("mapping_offset") = 0
        )
        .def(
            "is_mutable",
            &Carrier::is_mutable,
            "Return whether public carrier interfaces may currently modify storage.\n\n"
            "This combines the carrier's intrinsic mutability with ownership. "
            "A carrier constructed as immutable always returns false. A mutable carrier "
            "also returns false while it is exclusively owned by another carrier, "
            "except during the owner's private access scope.\n\n"
            "Returns:\n"
            "    True when public mutation is currently permitted; otherwise "
            "False.\n\n"
            "Examples:\n"
            "    >>> import strideweave as sw\n"
            "    >>> carrier = sw.Generic([1.0], mutable=False)\n"
            "    >>> carrier.is_mutable()\n"
            "    False"
        )
        .def(
            "is_owned",
            &Carrier::is_owned,
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
            "    True"
        )
        .def("_has_owner_access", &Carrier::has_owner_access)
        .def("_claim_ownership", &Carrier::claim_ownership)
        .def(
            "_relinquish_ownership",
            &Carrier::relinquish_ownership,
            py::arg("token")
        )
        .def(
            "_begin_owner_access", &Carrier::begin_owner_access, py::arg("token")
        )
        .def("_end_owner_access", &Carrier::end_owner_access, py::arg("token"))
        .def("dlpack_info", &Carrier::dlpack_info)
        .def_property_readonly("version", &Carrier::version)
        .def("_increment_version", &Carrier::increment_version)
        .def("is_released", &Carrier::is_released)
        .def(
            "release",
            &Carrier::release,
            "Release the carrier's storage; further element access raises.\n\n"
            "Contract: ``new_like`` must remain usable after ``release()`` --\n"
            "it constructs fresh storage and reads nothing from the released\n"
            "instance. Move's backward pass relies on this to materialize\n"
            "gradients in a released source carrier."
        )
        .def("dispatch_op", &Carrier::dispatch_op, py::arg("operation_name"))
        .def("__getitem__", &Carrier::get_item, py::arg("index"))
        .def("__setitem__", &Carrier::set_item, py::arg("index"), py::arg("value"));

    py::class_<VectorCarrierForTest, Carrier>(module, "_VectorCarrierForTest")
        .def(py::init<py::iterable>(), py::arg("values"));

    py::module_::import("strideweave._operation");
    strideweave::carrier::bind_cpu(module);
}
