#include "_block_device.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace py = pybind11;

namespace {

using strideweave::block_device::BlockDeviceHandle;
using strideweave::block_device::CloseResult;
using strideweave::block_device::Geometry;
using strideweave::block_device::Index;
using strideweave::block_device::index_as_size;
using strideweave::block_device::InvalidDeviceError;
using strideweave::block_device::IoBackend;
using strideweave::block_device::open_block_device;
using strideweave::block_device::open_block_device_for_test;
using strideweave::block_device::SystemCallError;
using strideweave::block_device::TransferResult;
using strideweave::block_device::TransferStatus;
using strideweave::block_device::UnsupportedPlatformError;
using strideweave::block_device::validate_byte_range;

enum class ScriptStepKind { progress, interrupt, zero, error };

struct ScriptStep {
    ScriptStepKind kind;
    std::size_t value;
};

std::deque<ScriptStep> parse_script(py::iterable values, const char* name) {
    std::deque<ScriptStep> result;
    for (py::handle item : values) {
        if (!py::isinstance<py::tuple>(item)) {
            throw py::type_error(std::string(name) +
                                 " entries must be (kind, value) tuples");
        }
        const py::tuple entry = py::reinterpret_borrow<py::tuple>(item);
        if (py::len(entry) != 2) {
            throw py::value_error(std::string(name) +
                                  " entries must contain two items");
        }
        const std::string kind = entry[0].cast<std::string>();
        const Index raw_value = entry[1].cast<Index>();
        if (raw_value < 0) {
            throw py::value_error(std::string(name) +
                                  " entry values must be non-negative");
        }
        const std::size_t value = index_as_size(raw_value, "script value");

        if (kind == "progress") {
            if (value == 0) {
                throw py::value_error("progress script values must be positive");
            }
            result.push_back({ScriptStepKind::progress, value});
        } else if (kind == "interrupt") {
            result.push_back({ScriptStepKind::interrupt, 0});
        } else if (kind == "zero") {
            result.push_back({ScriptStepKind::zero, 0});
        } else if (kind == "error") {
            if (value == 0 ||
                value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw py::value_error(
                    "error script values must be positive errno values");
            }
            result.push_back({ScriptStepKind::error, value});
        } else {
            throw py::value_error(std::string(name) + " has unknown step kind " + kind);
        }
    }
    return result;
}

class TestIoState final : public IoBackend {
public:
    TestIoState(py::iterable read_script, py::iterable write_script, int close_error)
        : read_script_(parse_script(read_script, "read_script")),
          write_script_(parse_script(write_script, "write_script")),
          close_error_(close_error) {
        if (close_error_ < 0) {
            throw py::value_error("close_error must be a non-negative errno");
        }
    }

    ssize_t read_at(int descriptor, void* destination, std::size_t byte_count,
                    off_t byte_offset) noexcept override {
        ++read_attempts_;
        if (read_script_.empty()) {
            return ::pread(descriptor, destination, byte_count, byte_offset);
        }
        const ScriptStep step = read_script_.front();
        read_script_.pop_front();
        return run_read_step(step, descriptor, destination, byte_count, byte_offset);
    }

    ssize_t write_at(int descriptor, const void* source, std::size_t byte_count,
                     off_t byte_offset) noexcept override {
        ++write_attempts_;
        if (write_script_.empty()) {
            return ::pwrite(descriptor, source, byte_count, byte_offset);
        }
        const ScriptStep step = write_script_.front();
        write_script_.pop_front();
        return run_write_step(step, descriptor, source, byte_count, byte_offset);
    }

    int close_descriptor(int descriptor) noexcept override {
        ++close_attempts_;
        last_closed_descriptor_ = descriptor;
        const int close_result = ::close(descriptor);
        const int close_errno = errno;
        descriptor_was_closed_ = close_result == 0;
        if (close_error_ != 0) {
            errno = close_error_;
            return -1;
        }
        if (close_result != 0) {
            errno = close_errno;
        }
        return close_result;
    }

    void descriptor_opened(int descriptor) noexcept override {
        opened_descriptor_ = descriptor;
        descriptor_was_closed_ = false;
    }

    Index read_attempts() const noexcept { return read_attempts_; }
    Index write_attempts() const noexcept { return write_attempts_; }
    Index close_attempts() const noexcept { return close_attempts_; }
    int opened_descriptor() const noexcept { return opened_descriptor_; }
    int last_closed_descriptor() const noexcept { return last_closed_descriptor_; }
    bool descriptor_was_closed() const noexcept { return descriptor_was_closed_; }
    Index read_steps_remaining() const noexcept {
        return static_cast<Index>(read_script_.size());
    }
    Index write_steps_remaining() const noexcept {
        return static_cast<Index>(write_script_.size());
    }

private:
    static ssize_t scripted_terminal(const ScriptStep& step) noexcept {
        if (step.kind == ScriptStepKind::interrupt) {
            errno = EINTR;
            return -1;
        }
        if (step.kind == ScriptStepKind::zero) {
            return 0;
        }
        errno = static_cast<int>(step.value);
        return -1;
    }

    static ssize_t run_read_step(const ScriptStep& step, int descriptor,
                                 void* destination, std::size_t byte_count,
                                 off_t byte_offset) noexcept {
        if (step.kind != ScriptStepKind::progress) {
            return scripted_terminal(step);
        }
        return ::pread(descriptor, destination, std::min(step.value, byte_count),
                       byte_offset);
    }

    static ssize_t run_write_step(const ScriptStep& step, int descriptor,
                                  const void* source, std::size_t byte_count,
                                  off_t byte_offset) noexcept {
        if (step.kind != ScriptStepKind::progress) {
            return scripted_terminal(step);
        }
        return ::pwrite(descriptor, source, std::min(step.value, byte_count),
                        byte_offset);
    }

    std::deque<ScriptStep> read_script_;
    std::deque<ScriptStep> write_script_;
    int close_error_;
    Index read_attempts_ = 0;
    Index write_attempts_ = 0;
    Index close_attempts_ = 0;
    int opened_descriptor_ = -1;
    int last_closed_descriptor_ = -1;
    bool descriptor_was_closed_ = false;
};

template <typename Open>
std::shared_ptr<BlockDeviceHandle> translate_open_errors(const std::string& path,
                                                         Open&& open) {
    try {
        return open();
    } catch (const UnsupportedPlatformError& error) {
        PyErr_SetString(PyExc_NotImplementedError, error.what());
        throw py::error_already_set();
    } catch (const InvalidDeviceError& error) {
        throw py::value_error(error.what());
    } catch (const SystemCallError& error) {
        errno = error.error_number();
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path.c_str());
        throw py::error_already_set();
    }
}

std::string transfer_status_name(TransferStatus status) {
    switch (status) {
    case TransferStatus::complete:
        return "complete";
    case TransferStatus::zero_completion:
        return "zero_completion";
    case TransferStatus::os_error:
        return "os_error";
    }
    throw std::logic_error("unknown transfer status");
}

py::tuple read_bytes(BlockDeviceHandle& handle, Index byte_offset, Index byte_count) {
    validate_byte_range(handle.capacity_bytes(), byte_offset, byte_count);
    const std::size_t size = index_as_size(byte_count, "byte_count");
    if (size > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
        throw std::overflow_error("byte_count does not fit in Python bytes");
    }
    std::vector<char> payload(size);
    const std::uintptr_t pointer =
        size == 0 ? 0 : reinterpret_cast<std::uintptr_t>(payload.data());
    const TransferResult result = handle.read_into(pointer, byte_offset, byte_count);
    const char* data = size == 0 ? "" : payload.data();
    return py::make_tuple(result, py::bytes(data, static_cast<py::ssize_t>(size)));
}

TransferResult write_bytes(BlockDeviceHandle& handle, Index byte_offset,
                           const py::bytes& values) {
    const std::string payload = values;
    if (payload.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("byte payload length does not fit in Index");
    }
    const Index byte_count = static_cast<Index>(payload.size());
    const std::uintptr_t pointer =
        payload.empty() ? 0 : reinterpret_cast<std::uintptr_t>(payload.data());
    return handle.write_from(pointer, byte_offset, byte_count);
}

}  // namespace

PYBIND11_MODULE(_block_device, module) {
    module.doc() = "Private synchronous block-device I/O boundary for StrideWeave";

    py::class_<TransferResult>(module, "_TransferResult")
        .def_property_readonly("status",
                               [](const TransferResult& result) {
                                   return transfer_status_name(result.status);
                               })
        .def_readonly("completed_bytes", &TransferResult::completed_bytes)
        .def_readonly("error_number", &TransferResult::error_number)
        .def_property_readonly("is_complete", [](const TransferResult& result) {
            return result.status == TransferStatus::complete;
        });

    py::class_<CloseResult>(module, "_CloseResult")
        .def_readonly("error_number", &CloseResult::error_number)
        .def_property_readonly("succeeded", &CloseResult::succeeded);

    py::class_<BlockDeviceHandle, std::shared_ptr<BlockDeviceHandle>>(
        module, "_BlockDeviceHandle")
        .def_property_readonly("capacity_bytes", &BlockDeviceHandle::capacity_bytes)
        .def_property_readonly("logical_block_size",
                               &BlockDeviceHandle::logical_block_size)
        .def_property_readonly("physical_block_size",
                               &BlockDeviceHandle::physical_block_size)
        .def("is_open", &BlockDeviceHandle::is_open)
        .def("read_into", &BlockDeviceHandle::read_into, py::arg("destination_pointer"),
             py::arg("byte_offset"), py::arg("byte_count"))
        .def("write_from", &BlockDeviceHandle::write_from, py::arg("source_pointer"),
             py::arg("byte_offset"), py::arg("byte_count"))
        .def("read_bytes", &read_bytes, py::arg("byte_offset"), py::arg("byte_count"))
        .def("write_bytes", &write_bytes, py::arg("byte_offset"), py::arg("values"))
        .def("zero_fill", &BlockDeviceHandle::zero_fill, py::arg("byte_offset"),
             py::arg("byte_count"))
        .def("close", &BlockDeviceHandle::close);

    py::class_<TestIoState, std::shared_ptr<TestIoState>>(module, "_TestIoState")
        .def(py::init<py::iterable, py::iterable, int>(), py::kw_only(),
             py::arg("read_script") = py::tuple(),
             py::arg("write_script") = py::tuple(), py::arg("close_error") = 0)
        .def_property_readonly("read_attempts", &TestIoState::read_attempts)
        .def_property_readonly("write_attempts", &TestIoState::write_attempts)
        .def_property_readonly("close_attempts", &TestIoState::close_attempts)
        .def_property_readonly("opened_descriptor", &TestIoState::opened_descriptor)
        .def_property_readonly("last_closed_descriptor",
                               &TestIoState::last_closed_descriptor)
        .def_property_readonly("descriptor_was_closed",
                               &TestIoState::descriptor_was_closed)
        .def_property_readonly("read_steps_remaining",
                               &TestIoState::read_steps_remaining)
        .def_property_readonly("write_steps_remaining",
                               &TestIoState::write_steps_remaining);

    module.def(
        "_open_block_device",
        [](const std::string& path) {
            return translate_open_errors(path, [&] { return open_block_device(path); });
        },
        py::arg("path"));
    module.def(
        "_open_block_device_for_test",
        [](const std::string& path, Index capacity_bytes, Index logical_block_size,
           Index physical_block_size, const std::shared_ptr<TestIoState>& io) {
            const Geometry geometry{capacity_bytes, logical_block_size,
                                    physical_block_size};
            return translate_open_errors(
                path, [&] { return open_block_device_for_test(path, geometry, io); });
        },
        py::arg("path"), py::arg("capacity_bytes"), py::arg("logical_block_size"),
        py::arg("physical_block_size"), py::arg("io"));
}
