#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;

namespace {

using Index = long long;

class Data {
public:
    virtual ~Data() = default;

    virtual Index size() const = 0;
    virtual py::object type() const = 0;
    virtual py::object get_value(Index index) const = 0;
    virtual bool is_evictable() const { return false; }

    py::object get_item(Index index) const {
        const Index data_size = size();
        if (index < 0 || index >= data_size) {
            throw py::index_error("Data index out of range");
        }
        return get_value(index);
    }

    bool is_evicted() const { return is_evicted_; }

    void evict() {
        if (!is_evictable()) {
            throw std::runtime_error("Data is not evictable");
        }
        if (is_evicted_) {
            return;
        }
        _evict();
        is_evicted_ = true;
    }

    void promote() {
        if (!is_evictable()) {
            throw std::runtime_error("Data is not evictable");
        }
        if (!is_evicted_) {
            return;
        }
        _promote();
        is_evicted_ = false;
    }

protected:
    virtual void _evict() { throw std::runtime_error("Data is not evictable"); }

    virtual void _promote() { throw std::runtime_error("Data is not evictable"); }

private:
    bool is_evicted_ = false;
};

class PyData : public Data {
public:
    using Data::Data;

    Index size() const override { PYBIND11_OVERRIDE_PURE(Index, Data, size); }

    py::object type() const override {
        PYBIND11_OVERRIDE_PURE(py::object, Data, type);
    }

    py::object get_value(Index index) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Data, get_value, index);
    }

    bool is_evictable() const override {
        PYBIND11_OVERRIDE(bool, Data, is_evictable);
    }

protected:
    void _evict() override { PYBIND11_OVERRIDE(void, Data, _evict); }

    void _promote() override { PYBIND11_OVERRIDE(void, Data, _promote); }
};

class VectorDataForTest : public Data {
public:
    explicit VectorDataForTest(py::iterable values) : values_(py::list(values)) {}

    Index size() const override { return static_cast<Index>(py::len(values_)); }

    py::object type() const override {
        return py::module_::import("neotorch.data").attr("DataType").attr("Any");
    }

    py::object get_value(Index index) const override {
        return py::reinterpret_borrow<py::object>(values_[index]);
    }

private:
    py::list values_;
};

}  // namespace

PYBIND11_MODULE(_data, module) {
    module.doc() = "Native data base classes for neotorch";

    py::class_<Data, PyData>(module, "Data")
        .def(py::init<>())
        .def("size", &Data::size)
        .def("type", &Data::type)
        .def("get_value", &Data::get_value, py::arg("index"))
        .def("is_evictable", &Data::is_evictable)
        .def("is_evicted", &Data::is_evicted)
        .def("evict", &Data::evict)
        .def("promote", &Data::promote)
        .def("__getitem__", &Data::get_item, py::arg("index"));

    py::class_<VectorDataForTest, Data>(module, "_VectorDataForTest")
        .def(py::init<py::iterable>(), py::arg("values"));
}
