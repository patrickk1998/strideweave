#include <pybind11/pybind11.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using Index = long long;

void validate_copy_arguments(std::uintptr_t pointer, Index byte_count) {
    if (byte_count < 0) {
        throw py::value_error("byte_count must be non-negative");
    }
    if (byte_count > 0 && pointer == 0) {
        throw py::value_error("pointer must be non-null");
    }
}

void copy_memory_to_file(
    const std::string& path, std::uintptr_t pointer, Index byte_count
) {
    validate_copy_arguments(pointer, byte_count);
    if (byte_count == 0) {
        return;
    }

    py::gil_scoped_release release;
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Move could not open the destination file");
    }
    file.write(reinterpret_cast<const char*>(pointer), byte_count);
    if (!file) {
        throw std::runtime_error("Move could not write the destination file");
    }
}

void copy_file_to_memory(
    const std::string& path,
    Index byte_offset,
    std::uintptr_t pointer,
    Index byte_count
) {
    validate_copy_arguments(pointer, byte_count);
    if (byte_offset < 0) {
        throw py::value_error("byte_offset must be non-negative");
    }
    if (byte_count == 0) {
        return;
    }

    py::gil_scoped_release release;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Move could not open the source file");
    }
    file.seekg(byte_offset);
    file.read(reinterpret_cast<char*>(pointer), byte_count);
    if (file.gcount() != byte_count) {
        throw std::runtime_error("Move could not read the source file");
    }
}

}  // namespace

PYBIND11_MODULE(_move, module) {
    module.doc() = "Native bulk copy helpers for the neotorch move operation";

    module.def(
        "copy_memory_to_file",
        &copy_memory_to_file,
        py::arg("path"),
        py::arg("pointer"),
        py::arg("byte_count")
    );
    module.def(
        "copy_file_to_memory",
        &copy_file_to_memory,
        py::arg("path"),
        py::arg("byte_offset"),
        py::arg("pointer"),
        py::arg("byte_count")
    );
}
