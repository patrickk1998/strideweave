#include "_cpu_registry.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace strideweave::carrier {
namespace {

struct CpuRegistryEntry {
    CpuOperationFactory factory;
    bool is_native;
    std::string kernel_id;
    std::string variant;
    std::string pybind_name;
};

class CpuOperationRegistry {
public:
    void register_native(const CpuKernelMetadata& metadata,
                         CpuOperationFactory operation_factory) {
        validate_metadata(metadata);
        const std::string dispatch_name(metadata.dispatch_name);
        const std::string kernel_id(metadata.kernel_id);
        reject_duplicate_dispatch(dispatch_name);
        if (kernel_ids_.find(kernel_id) != kernel_ids_.end()) {
            throw std::runtime_error("duplicate CPU kernel ID '" + kernel_id + "'");
        }
        entries_.emplace(dispatch_name,
                         CpuRegistryEntry{std::move(operation_factory), true, kernel_id,
                                          metadata.variant, metadata.pybind_name});
        kernel_ids_.insert(kernel_id);
    }

    void register_python(std::string operation_name,
                         CpuOperationFactory operation_factory) {
        reject_duplicate_dispatch(operation_name);
        entries_.emplace(
            std::move(operation_name),
            CpuRegistryEntry{std::move(operation_factory), false, "", "", ""});
    }

    py::object make(const std::string& operation_name) const {
        const auto operation = entries_.find(operation_name);
        if (operation == entries_.end()) {
            PyErr_Format(PyExc_NotImplementedError,
                         "CPU carrier does not support operation '%s'",
                         operation_name.c_str());
            throw py::error_already_set();
        }
        return operation->second.factory();
    }

    py::tuple native_metadata() const {
        std::vector<std::pair<std::string, const CpuRegistryEntry*>> ordered;
        ordered.reserve(entries_.size());
        for (const auto& [dispatch_name, entry] : entries_) {
            if (entry.is_native) {
                ordered.emplace_back(dispatch_name, &entry);
            }
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        py::tuple result(ordered.size());
        for (std::size_t index = 0; index < ordered.size(); ++index) {
            const auto& [dispatch_name, entry] = ordered[index];
            result[index] = py::make_tuple(dispatch_name, entry->kernel_id,
                                           entry->variant, entry->pybind_name);
        }
        return result;
    }

private:
    static void validate_metadata(const CpuKernelMetadata& metadata) {
        if (metadata.dispatch_name == nullptr || metadata.dispatch_name[0] == '\0' ||
            metadata.kernel_id == nullptr || metadata.kernel_id[0] == '\0' ||
            metadata.variant == nullptr || metadata.variant[0] == '\0' ||
            metadata.pybind_name == nullptr || metadata.pybind_name[0] == '\0') {
            throw std::invalid_argument("CPU kernel metadata fields must be non-empty");
        }
        if (std::string(metadata.variant) != "default") {
            throw std::invalid_argument(
                "CPU kernel variant must be 'default' in this registry version");
        }
    }

    void reject_duplicate_dispatch(const std::string& operation_name) const {
        if (entries_.find(operation_name) != entries_.end()) {
            throw std::runtime_error("duplicate CPU dispatch name '" + operation_name +
                                     "'");
        }
    }

    std::unordered_map<std::string, CpuRegistryEntry> entries_;
    std::unordered_set<std::string> kernel_ids_;
};

CpuOperationRegistry& cpu_operation_registry() {
    static CpuOperationRegistry registry;
    return registry;
}

CpuKernelMetadata metadata_from_tuple(py::handle value) {
    const py::tuple entry = py::cast<py::tuple>(value);
    if (py::len(entry) != 4) {
        throw py::value_error("CPU kernel metadata test entries require four fields");
    }
    // The strings remain alive for the duration of the immediate registration.
    static thread_local std::string dispatch_name;
    static thread_local std::string kernel_id;
    static thread_local std::string variant;
    static thread_local std::string pybind_name;
    dispatch_name = py::cast<std::string>(entry[0]);
    kernel_id = py::cast<std::string>(entry[1]);
    variant = py::cast<std::string>(entry[2]);
    pybind_name = py::cast<std::string>(entry[3]);
    return {dispatch_name.c_str(), kernel_id.c_str(), variant.c_str(),
            pybind_name.c_str()};
}

void validate_cpu_native_registry_for_test(py::iterable entries) {
    CpuOperationRegistry registry;
    for (py::handle entry : entries) {
        registry.register_native(metadata_from_tuple(entry), [] { return py::none(); });
    }
}

}  // namespace

void register_cpu_native_operation(const CpuKernelMetadata& metadata,
                                   CpuOperationFactory operation_factory) {
    cpu_operation_registry().register_native(metadata, std::move(operation_factory));
}

void register_python_cpu_operation(const char* operation_name,
                                   const char* operation_module_name,
                                   const char* operation_type_name) {
    cpu_operation_registry().register_python(operation_name, [operation_module_name,
                                                              operation_type_name] {
        return py::module_::import(operation_module_name).attr(operation_type_name)();
    });
}

py::object make_registered_cpu_operation(const std::string& operation_name) {
    return cpu_operation_registry().make(operation_name);
}

void bind_cpu_registry_introspection(py::module_& module) {
    module.def("_cpu_native_kernel_metadata",
               [] { return cpu_operation_registry().native_metadata(); });
    module.def("_validate_cpu_native_registry_for_test",
               &validate_cpu_native_registry_for_test, py::arg("entries"));
}

}  // namespace strideweave::carrier
