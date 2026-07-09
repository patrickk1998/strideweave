#pragma once

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace neotorch::data {

using Index = long long;

class Data {
public:
    virtual ~Data() = default;

    virtual Index size() const = 0;
    virtual py::object type() const = 0;
    virtual py::object get_value(Index index) const = 0;
    virtual py::object new_like(py::iterable values, bool is_mutable) const = 0;
    virtual void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) = 0;
    virtual bool is_evictable() const { return false; }
    virtual bool is_mutable() const { return false; }

    virtual py::dict dlpack_info() const {
        PyErr_SetString(
            PyExc_BufferError, "DLPack is not supported for this data class"
        );
        throw py::error_already_set();
    }

    static py::object dispatch_op(const std::string& operation_name) {
        PyErr_Format(
            PyExc_NotImplementedError,
            "Data class does not support operation '%s'",
            operation_name.c_str()
        );
        throw py::error_already_set();
    }

    py::object get_item(Index index) const {
        require_not_released();
        const Index data_size = size();
        if (index < 0 || index >= data_size) {
            throw py::index_error("Data index out of range");
        }
        return get_value(index);
    }

    void set_item(Index index, py::object value) {
        require_not_released();
        if (!is_mutable()) {
            throw std::runtime_error("Data is not mutable");
        }
        const Index data_size = size();
        if (index < 0 || index >= data_size) {
            throw py::index_error("Data index out of range");
        }
        set_value(index, value);
    }

    bool is_evicted() const { return is_evicted_; }

    void evict() {
        require_not_released();
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
        require_not_released();
        if (!is_evictable()) {
            throw std::runtime_error("Data is not evictable");
        }
        if (!is_evicted_) {
            return;
        }
        _promote();
        is_evicted_ = false;
    }

    bool is_released() const { return is_released_; }

    void release() {
        if (is_released_) {
            return;
        }
        _release();
        is_released_ = true;
    }

protected:
    virtual void set_value(Index, py::object) {
        throw std::runtime_error("Data is not mutable");
    }

    virtual void _evict() { throw std::runtime_error("Data is not evictable"); }

    virtual void _promote() { throw std::runtime_error("Data is not evictable"); }

    virtual void _release() {}

private:
    void require_not_released() const {
        if (is_released_) {
            throw std::runtime_error("Data is released");
        }
    }

    bool is_evicted_ = false;
    bool is_released_ = false;
};

void bind_cpu(py::module_& module);

}  // namespace neotorch::data
