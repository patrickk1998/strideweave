#pragma once

#include <pybind11/pybind11.h>

#include <functional>
#include <string>

#include "_cpu_operation.hpp"

namespace py = pybind11;

namespace strideweave::carrier {

using CpuOperationFactory = std::function<py::object()>;

struct CpuKernelMetadata {
    const char* dispatch_name;
    const char* kernel_id;
    const char* variant;
    const char* pybind_name;
};

void register_cpu_native_operation(const CpuKernelMetadata& metadata,
                                   CpuOperationFactory operation_factory);
void register_python_cpu_operation(const char* operation_name,
                                   const char* operation_module_name,
                                   const char* operation_type_name);
py::object make_registered_cpu_operation(const std::string& operation_name);
void bind_cpu_registry_introspection(py::module_& module);

template <typename Operation>
void bind_and_register_cpu_operation(py::module_& module,
                                     const CpuKernelMetadata& metadata) {
    bind_cpu_operation<Operation>(module, metadata.pybind_name);
    register_cpu_native_operation(metadata, [] {
        return py::cast(new Operation(), py::return_value_policy::take_ownership);
    });
}

}  // namespace strideweave::carrier
