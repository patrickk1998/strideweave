#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

namespace py = pybind11;

namespace strideweave::carrier {

using Index = long long;
using Version = std::uint64_t;
using OwnershipToken = std::uint64_t;

class Carrier {
public:
    virtual ~Carrier() = default;

    virtual Index size() const = 0;
    virtual py::object dtype() const = 0;
    virtual py::object get_value(Index index) const = 0;
    virtual py::object new_like(py::iterable values, bool is_mutable) const = 0;
    virtual py::object empty_like(
        Index size, bool is_mutable, py::object dtype
    ) const = 0;
    virtual void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) = 0;
    bool is_mutable() const {
        return _is_mutable() && (!is_owned() || has_owner_access());
    }

    bool is_owned() const { return ownership_token_ != 0; }

    bool has_owner_access() const {
        return owner_access_depth_ != 0 &&
               owner_access_thread_ == std::this_thread::get_id();
    }

    OwnershipToken claim_ownership() {
        require_not_released();
        if (is_owned()) {
            throw std::runtime_error("Carrier is already owned by another carrier");
        }
        ownership_token_ = next_ownership_token();
        return ownership_token_;
    }

    void relinquish_ownership(OwnershipToken token) {
        require_owner(token);
        if (owner_access_depth_ != 0) {
            throw std::runtime_error(
                "Carrier ownership cannot be relinquished during owner access"
            );
        }
        ownership_token_ = 0;
    }

    void begin_owner_access(OwnershipToken token) {
        require_owner(token);
        const std::thread::id current_thread = std::this_thread::get_id();
        if (owner_access_depth_ != 0 && owner_access_thread_ != current_thread) {
            throw std::runtime_error(
                "Carrier is already being accessed by its owner on another thread"
            );
        }
        owner_access_thread_ = current_thread;
        ++owner_access_depth_;
    }

    void end_owner_access(OwnershipToken token) {
        require_owner(token);
        if (!has_owner_access()) {
            throw std::runtime_error(
                "Carrier owner access must end on the thread where it began"
            );
        }
        --owner_access_depth_;
        if (owner_access_depth_ == 0) {
            owner_access_thread_ = std::thread::id();
        }
    }

    virtual py::dict dlpack_info() const {
        PyErr_SetString(
            PyExc_BufferError, "DLPack is not supported for this carrier"
        );
        throw py::error_already_set();
    }

    virtual py::object dispatch_op(const std::string& operation_name) const {
        PyErr_Format(
            PyExc_NotImplementedError,
            "Carrier class does not support operation '%s'",
            operation_name.c_str()
        );
        throw py::error_already_set();
    }

    py::object get_item(Index index) const {
        require_not_released();
        const Index data_size = size();
        if (index < 0 || index >= data_size) {
            throw py::index_error("Carrier index out of range");
        }
        return get_value(index);
    }

    void set_item(Index index, py::object value) {
        require_not_released();
        if (!is_mutable()) {
            throw std::runtime_error("Carrier is not mutable");
        }
        const Index data_size = size();
        if (index < 0 || index >= data_size) {
            throw py::index_error("Carrier index out of range");
        }
        const Version previous_version = version_;
        set_value(index, value);
        if (version_ == previous_version) {
            increment_version();
        }
    }

    Version version() const { return version_; }

    void increment_version() {
        require_external_or_owner_access();
        ++version_;
    }

    bool is_released() const { return is_released_; }

    void release() {
        require_external_or_owner_access();
        if (is_released_) {
            return;
        }
        _release();
        is_released_ = true;
    }

protected:
    // Backends implement storage capability here. Public is_mutable() also
    // applies the ownership guard and must not be overridden by carriers.
    virtual bool _is_mutable() const { return false; }

    virtual void set_value(Index, py::object) {
        throw std::runtime_error("Carrier is not mutable");
    }

    virtual void _release() {}

private:
    static OwnershipToken next_ownership_token() {
        static std::atomic<OwnershipToken> next_token{1};
        return next_token.fetch_add(1, std::memory_order_relaxed);
    }

    void require_owner(OwnershipToken token) const {
        if (!is_owned() || ownership_token_ != token) {
            throw std::runtime_error("Invalid carrier ownership token");
        }
    }

    void require_external_or_owner_access() const {
        if (is_owned() && !has_owner_access()) {
            throw std::runtime_error(
                "Carrier is owned by another carrier and cannot be modified directly"
            );
        }
    }

    void require_not_released() const {
        if (is_released_) {
            throw std::runtime_error("Carrier is released");
        }
    }

    bool is_released_ = false;
    Version version_ = 0;
    OwnershipToken ownership_token_ = 0;
    std::thread::id owner_access_thread_;
    std::size_t owner_access_depth_ = 0;
};

void bind_cpu(py::module_& module);

}  // namespace strideweave::carrier
