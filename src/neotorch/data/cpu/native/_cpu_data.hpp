#pragma once

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_data.hpp"
#include "_layout_index.hpp"

namespace py = pybind11;

namespace neotorch::data {

inline py::object tensor_type() {
    return py::module_::import("neotorch.tensor").attr("Tensor");
}

inline bool layouts_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

inline bool objects_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

inline py::object data_type(const char* name) {
    return py::module_::import("neotorch.data").attr("DataType").attr(name);
}

enum class CpuDType { Float32, Int32 };

inline CpuDType parse_cpu_dtype(py::handle dtype) {
    if (dtype.is_none() || objects_equal(dtype, data_type("Float32"))) {
        return CpuDType::Float32;
    }
    if (objects_equal(dtype, data_type("Int32"))) {
        return CpuDType::Int32;
    }
    throw py::value_error("CPU dtype must be DataType.Float32 or DataType.Int32");
}

inline py::object cpu_dtype_object(CpuDType dtype) {
    if (dtype == CpuDType::Float32) {
        return data_type("Float32");
    }
    return data_type("Int32");
}

inline std::size_t cpu_dtype_size(CpuDType dtype) {
    if (dtype == CpuDType::Float32) {
        return sizeof(float);
    }
    return sizeof(std::int32_t);
}

inline void throw_overflow_error(const std::string& message) {
    throw std::overflow_error(message);
}

class CPU : public Data {
public:
    CPU(Index size, py::object pointer, bool is_mutable, py::object dtype)
        : size_(size), dtype_(parse_cpu_dtype(dtype)), is_mutable_(is_mutable) {
        if (size_ < 0) {
            throw py::value_error("CPU size must be non-negative");
        }

        if (pointer.is_none()) {
            if (dtype_ == CpuDType::Float32) {
                owned_float_data_ =
                    std::make_unique<float[]>(static_cast<std::size_t>(size_));
                data_ = reinterpret_cast<std::byte*>(owned_float_data_.get());
                std::fill_n(data_as<float>(), static_cast<std::size_t>(size_), 0.0f);
            } else {
                owned_int_data_ =
                    std::make_unique<std::int32_t[]>(static_cast<std::size_t>(size_));
                data_ = reinterpret_cast<std::byte*>(owned_int_data_.get());
                std::fill_n(
                    data_as<std::int32_t>(), static_cast<std::size_t>(size_), 0
                );
            }
            return;
        }

        if (!py::isinstance<py::int_>(pointer)) {
            throw py::type_error("CPU pointer must be an integer address or None");
        }

        const auto pointer_value = py::cast<long long>(pointer);
        if (pointer_value <= 0) {
            throw py::value_error("CPU pointer must be a positive integer address");
        }
        data_ = reinterpret_cast<std::byte*>(
            static_cast<std::uintptr_t>(pointer_value)
        );
    }

    Index size() const override { return size_; }

    py::object type() const override { return cpu_dtype_object(dtype_); }

    CpuDType dtype() const { return dtype_; }

    py::object get_value(Index index) const override {
        if (dtype_ == CpuDType::Float32) {
            return py::float_(element<float>(index));
        }
        return py::int_(element<std::int32_t>(index));
    }

    py::object new_like(py::iterable values, bool is_mutable) const override {
        return new_like_with_dtype(values, is_mutable, py::none());
    }

    py::object new_like_with_dtype(
        py::iterable values, bool is_mutable, py::object dtype
    ) const {
        py::list materialized_values(values);
        py::object cpu_type = py::module_::import("neotorch._data").attr("CPU");
        CpuDType result_dtype = dtype.is_none() ? dtype_ : parse_cpu_dtype(dtype);
        py::object result = cpu_type(
            py::int_(py::len(materialized_values)),
            py::none(),
            py::arg("mutable") = is_mutable,
            py::arg("dtype") = cpu_dtype_object(result_dtype)
        );
        CPU& result_data = py::cast<CPU&>(result);
        for (py::ssize_t i = 0; i < py::len(materialized_values); ++i) {
            if (materialized_values[i].is_none()) {
                result_data.write_zero(static_cast<Index>(i));
                continue;
            }
            result_data.set_value_at(static_cast<Index>(i), materialized_values[i]);
        }
        return result;
    }

    py::object empty_like(
        Index size, bool is_mutable, py::object dtype
    ) const override {
        if (size < 0) {
            throw py::value_error("CPU allocation size must be non-negative");
        }
        py::object cpu_type = py::module_::import("neotorch._data").attr("CPU");
        CpuDType result_dtype = dtype.is_none() ? dtype_ : parse_cpu_dtype(dtype);
        return cpu_type(
            py::int_(size),
            py::none(),
            py::arg("mutable") = is_mutable,
            py::arg("dtype") = cpu_dtype_object(result_dtype)
        );
    }

    void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) override;

    std::uintptr_t pointer() const {
        return reinterpret_cast<std::uintptr_t>(data_);
    }

    py::dict dlpack_info() const override {
        py::dict info;
        info["pointer"] = py::int_(pointer());
        info["device_type"] = py::int_(1);
        info["device_id"] = py::int_(0);
        return info;
    }

    void set_value_public(Index index, py::object value) {
        if (!is_mutable()) {
            throw std::runtime_error("Data is not mutable");
        }
        set_value_at(index, value);
        increment_version();
    }

    template <typename T>
    T* data_as() {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(data_);
    }

    template <typename T>
    T& element(Index index) {
        validate_index(index);
        return unchecked_element<T>(index);
    }

    template <typename T>
    const T& element(Index index) const {
        validate_index(index);
        return unchecked_element<T>(index);
    }

    template <typename T>
    T& unchecked_element(Index index) {
        return *reinterpret_cast<T*>(data_ + index * static_cast<Index>(sizeof(T)));
    }

    template <typename T>
    const T& unchecked_element(Index index) const {
        return *reinterpret_cast<const T*>(
            data_ + index * static_cast<Index>(sizeof(T))
        );
    }

    py::object dispatch_op(const std::string& operation_name) const override;

protected:
    bool _is_mutable() const override { return is_mutable_; }

    void set_value(Index index, py::object value) override {
        set_value_public(index, value);
    }

    void _release() override {
        owned_float_data_.reset();
        owned_int_data_.reset();
        data_ = nullptr;
        size_ = 0;
    }

private:
    void write_zero(Index index) {
        if (dtype_ == CpuDType::Float32) {
            element<float>(index) = 0.0f;
        } else {
            element<std::int32_t>(index) = 0;
        }
    }

    void set_value_at(Index index, py::handle value) {
        if (dtype_ == CpuDType::Float32) {
            element<float>(index) = py::cast<float>(value);
        } else {
            element<std::int32_t>(index) = require_int32(value, "CPU value");
        }
    }

    static std::int32_t require_int32(py::handle value, const char* name) {
        PyObject* index_object = PyNumber_Index(value.ptr());
        if (index_object == nullptr) {
            PyErr_Clear();
            throw py::type_error(std::string(name) + " must be an integer");
        }
        py::object index = py::reinterpret_steal<py::object>(index_object);
        const long long result = PyLong_AsLongLong(index.ptr());
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (result < std::numeric_limits<std::int32_t>::min() ||
            result > std::numeric_limits<std::int32_t>::max()) {
            throw_overflow_error(std::string(name) + " is out of int32 range");
        }
        return static_cast<std::int32_t>(result);
    }

    void validate_index(Index index) const {
        if (index < 0 || index >= size_) {
            throw py::index_error("Data index out of range");
        }
    }

    Index size_;
    CpuDType dtype_;
    bool is_mutable_;
    std::unique_ptr<float[]> owned_float_data_;
    std::unique_ptr<std::int32_t[]> owned_int_data_;
    std::byte* data_ = nullptr;
};

inline float require_float(py::handle value, const char* name) {
    PyObject* value_ptr = value.ptr();
    const double result = PyFloat_AsDouble(value_ptr);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::type_error(std::string(name) + " must be a numerical scalar");
    }
    return static_cast<float>(result);
}

inline bool is_integral_scalar(py::handle value) {
    if (PyBool_Check(value.ptr())) {
        return false;
    }
    PyObject* index_object = PyNumber_Index(value.ptr());
    if (index_object == nullptr) {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(index_object);
    return true;
}

inline std::int32_t require_int32_scalar(py::handle value, const char* name) {
    PyObject* index_object = PyNumber_Index(value.ptr());
    if (index_object == nullptr) {
        PyErr_Clear();
        throw py::type_error(std::string(name) + " must be an integer");
    }
    py::object index = py::reinterpret_steal<py::object>(index_object);
    const long long result = PyLong_AsLongLong(index.ptr());
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (result < std::numeric_limits<std::int32_t>::min() ||
        result > std::numeric_limits<std::int32_t>::max()) {
        throw_overflow_error(std::string(name) + " is out of int32 range");
    }
    return static_cast<std::int32_t>(result);
}

inline std::int32_t checked_int32(long long value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw_overflow_error("CPU Int32 operation result is out of int32 range");
    }
    return static_cast<std::int32_t>(value);
}

inline long long checked_add(long long lhs, long long rhs) {
    if ((rhs > 0 && lhs > std::numeric_limits<long long>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<long long>::min() - rhs)) {
        throw_overflow_error("CPU integer accumulation overflowed");
    }
    return lhs + rhs;
}

inline std::int32_t checked_int32_pow(std::int32_t base, int exponent) {
    long long result = 1;
    long long factor = base;
    int remaining = exponent;
    while (remaining > 0) {
        if (remaining & 1) {
            result = checked_int32(result * factor);
        }
        remaining >>= 1;
        if (remaining > 0) {
            factor = checked_int32(factor * factor);
        }
    }
    return static_cast<std::int32_t>(result);
}

inline CPU& cpu_data_from_tensor(py::handle tensor, const char* name) {
    if (!py::isinstance(tensor, tensor_type())) {
        throw py::type_error(std::string(name) + " must be a Tensor");
    }
    py::object data = tensor.attr("data");
    if (!py::isinstance<CPU>(data)) {
        throw py::type_error(std::string(name) + " must be backed by CPU data");
    }
    CPU& cpu_data = py::cast<CPU&>(data);
    if (cpu_data.is_released()) {
        throw std::runtime_error(std::string(name) + " data is released");
    }
    return cpu_data;
}

inline Index tensor_offset(py::handle tensor) {
    return py::cast<Index>(tensor.attr("offset"));
}

inline py::object tensor_layout(py::handle tensor) { return tensor.attr("layout"); }

struct CpuTensorView {
    CPU* data;
    Index offset;
    py::object cache_owner;
    const neotorch::layout_index::LayoutCache* cache;
    Index logical_size;

    std::size_t leaf_rank() const {
        return static_cast<std::size_t>(cache->leaf_rank());
    }

    float read_float_logical(Index key) const {
        const Index layout_index = cache->index_logical(key);
        return read_float_storage(offset + layout_index);
    }

    void write_float_logical(Index key, float value) const {
        const Index layout_index = cache->index_logical(key);
        data->unchecked_element<float>(offset + layout_index) = value;
    }

    float read_float_coords(const Index* key, std::size_t size) const {
        const Index layout_index = cache->index(key, size);
        return read_float_storage(offset + layout_index);
    }

    void write_float_coords(const Index* key, std::size_t size, float value) const {
        const Index layout_index = cache->index(key, size);
        data->unchecked_element<float>(offset + layout_index) = value;
    }

    float read_float_expanded(const Index* key, std::size_t size) const {
        const Index layout_index = cache->index_expanded(key, size);
        return read_float_storage(offset + layout_index);
    }

    float read_float_expanded(const std::vector<Index>& key) const {
        return read_float_expanded(key.data(), key.size());
    }

    void write_float_expanded(const Index* key, std::size_t size, float value) const {
        const Index layout_index = cache->index_expanded(key, size);
        data->unchecked_element<float>(offset + layout_index) = value;
    }

    void write_float_expanded(const std::vector<Index>& key, float value) const {
        write_float_expanded(key.data(), key.size(), value);
    }

    std::int32_t read_int_expanded(const Index* key, std::size_t size) const {
        const Index layout_index = cache->index_expanded(key, size);
        return data->unchecked_element<std::int32_t>(offset + layout_index);
    }

    std::int32_t read_int_expanded(const std::vector<Index>& key) const {
        return read_int_expanded(key.data(), key.size());
    }

    void write_int_expanded(
        const Index* key, std::size_t size, std::int32_t value
    ) const {
        const Index layout_index = cache->index_expanded(key, size);
        data->unchecked_element<std::int32_t>(offset + layout_index) = value;
    }

    void write_int_expanded(const std::vector<Index>& key, std::int32_t value) const {
        write_int_expanded(key.data(), key.size(), value);
    }

    float read_float_storage(Index index) const {
        if (data->dtype() == CpuDType::Float32) {
            return data->unchecked_element<float>(index);
        }
        return static_cast<float>(data->unchecked_element<std::int32_t>(index));
    }
};

inline CpuTensorView cpu_tensor_view(py::handle tensor, const char* name) {
    CPU& data = cpu_data_from_tensor(tensor, name);
    py::object layout = tensor_layout(tensor);
    py::object cache_owner = layout.attr("_cache");
    auto& cache = py::cast<neotorch::layout_index::LayoutCache&>(cache_owner);
    return CpuTensorView{
        &data,
        tensor_offset(tensor),
        std::move(cache_owner),
        &cache,
        cache.logical_size(),
    };
}

inline void CPU::scatter(
    py::object to_scatter,
    py::object scatter_onto,
    py::object mapping,
    Index mapping_offset
) {
    if (mapping_offset < 0) {
        throw py::value_error("mapping_offset must be non-negative");
    }
    CPU& destination_data = cpu_data_from_tensor(scatter_onto, "scatter_onto");
    if (&destination_data != this) {
        throw py::value_error("scatter_onto must be backed by this data object");
    }
    if (!is_mutable()) {
        throw std::runtime_error("Data is not mutable");
    }

    CpuTensorView source = cpu_tensor_view(to_scatter, "to_scatter");
    py::object source_layout = tensor_layout(to_scatter);
    py::object source_shape = source_layout.attr("shape");
    py::object mapping_shape = mapping.attr("shape");
    if (!layouts_equal(mapping_shape, source_shape)) {
        throw py::value_error("mapping shape must match to_scatter layout shape");
    }
    if (dtype_ == CpuDType::Int32 && source.data->dtype() != CpuDType::Int32) {
        throw py::type_error("cannot scatter Float32 values into Int32 CPU data");
    }

    py::object mapping_cache_owner = mapping.attr("_cache");
    auto& mapping_cache =
        py::cast<neotorch::layout_index::LayoutCache&>(mapping_cache_owner);
    const Index destination_offset = tensor_offset(scatter_onto);
    const Index mapped_storage_size = mapping_cache.cosize();
    const bool storage_exceeds_data =
        destination_offset > size_ ||
        mapping_offset > size_ - destination_offset ||
        mapped_storage_size > size_ - destination_offset - mapping_offset;
    if (storage_exceeds_data) {
        throw py::value_error("scatter mapping exceeds destination data size");
    }

    {
        py::gil_scoped_release release;
        std::vector<Index> key(mapping_cache.leaf_rank(), 0);
        for (Index i = 0; i < mapping_cache.logical_size(); ++i) {
            const Index source_index = source.cache->index_expanded(
                key.data(), key.size()
            );
            const Index destination_index =
                destination_offset + mapping_offset +
                mapping_cache.index_expanded(key.data(), key.size());
            if (dtype_ == CpuDType::Float32) {
                unchecked_element<float>(destination_index) =
                    source.read_float_storage(source.offset + source_index);
            } else {
                unchecked_element<std::int32_t>(destination_index) =
                    source.data->unchecked_element<std::int32_t>(
                        source.offset + source_index
                    );
            }
            mapping_cache.increment_key(key.data(), key.size());
        }
    }
    increment_version();
}

inline void require_same_layout(py::handle lhs, py::handle rhs) {
    if (!layouts_equal(tensor_layout(lhs), tensor_layout(rhs))) {
        throw py::value_error("Tensor layouts must match");
    }
}

inline void require_layout(py::handle tensor, py::handle layout) {
    if (!layouts_equal(tensor_layout(tensor), layout)) {
        throw py::value_error("Tensor layouts must match");
    }
}

inline py::object mode_shape(py::handle layout, Index mode) {
    return layout.attr("shape").attr("top_level")[py::int_(mode)];
}

inline Index mode_logical_size(py::handle layout, Index mode) {
    py::object shape = mode_shape(layout, mode);
    if (neotorch::layout_index::is_int(shape)) {
        return py::cast<Index>(shape);
    }
    return py::cast<Index>(shape.attr("logical_size"));
}

inline void require_two_mode_tensor(py::handle tensor, const char* name) {
    cpu_data_from_tensor(tensor, name);
    if (py::len(tensor_layout(tensor)) != 2) {
        throw py::value_error(std::string(name) + " must have a two-mode layout");
    }
}

inline std::pair<py::object, Index> canonical_stride_level(
    py::handle shape_level, Index stride
) {
    if (neotorch::layout_index::is_int(shape_level)) {
        return {py::int_(stride), stride * py::cast<Index>(shape_level)};
    }

    py::list stride_level;
    Index next_stride = stride;
    for (py::handle shape : py::reinterpret_borrow<py::sequence>(shape_level)) {
        auto [child_stride, child_next_stride] =
            canonical_stride_level(shape, next_stride);
        stride_level.append(child_stride);
        next_stride = child_next_stride;
    }
    return {std::move(stride_level), next_stride};
}

inline py::object canonical_layout_from_modes(
    std::initializer_list<py::object> modes
) {
    py::object layout_module = py::module_::import("neotorch.layout");
    py::object shape_type = layout_module.attr("Shape");
    py::object stride_type = layout_module.attr("Stride");
    py::object layout_type = layout_module.attr("Layout");

    py::object shape;
    if (modes.size() == 1) {
        shape = shape_type(*modes.begin());
    } else {
        py::list mode_list;
        for (const py::object& mode : modes) {
            mode_list.append(mode);
        }
        shape = shape_type(mode_list);
    }

    auto [stride_level, _] = canonical_stride_level(shape.attr("top_level"), 1);
    return layout_type(shape, stride_type(stride_level));
}

inline py::object make_cpu_data(Index size, CpuDType dtype = CpuDType::Float32) {
    py::object cpu_type = py::module_::import("neotorch._data").attr("CPU");
    return cpu_type(
        py::int_(size), py::none(), py::arg("dtype") = cpu_dtype_object(dtype)
    );
}

inline py::object make_tensor(py::object data, py::object layout) {
    return tensor_type()(std::move(data), py::int_(0), std::move(layout));
}

struct CpuTensorAllocation {
    py::object data_object;
    py::object layout_object;
    CpuTensorView view;
};

inline CpuTensorAllocation allocate_cpu_tensor(
    py::object layout, CpuDType dtype = CpuDType::Float32
) {
    py::object cache_owner = layout.attr("_cache");
    auto& cache = py::cast<neotorch::layout_index::LayoutCache&>(cache_owner);
    py::object data = make_cpu_data(cache.cosize(), dtype);
    CPU& cpu = py::cast<CPU&>(data);
    return CpuTensorAllocation{
        std::move(data),
        std::move(layout),
        CpuTensorView{&cpu, 0, std::move(cache_owner), &cache, cache.logical_size()},
    };
}

inline py::object copy_gradient_for(py::handle target, py::handle gradient) {
    require_same_layout(target, gradient);
    CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
    py::object output_layout = tensor_layout(target);
    CpuTensorAllocation output = allocate_cpu_tensor(output_layout, CpuDType::Float32);
    {
        py::gil_scoped_release release;
        std::vector<Index> key(output.view.leaf_rank(), 0);
        for (Index i = 0; i < output.view.logical_size; ++i) {
            output.view.write_float_expanded(
                key, gradient_view.read_float_expanded(key)
            );
            output.view.cache->increment_key(key.data(), key.size());
        }
    }
    return make_tensor(std::move(output.data_object), std::move(output.layout_object));
}

inline py::object copy_negated_gradient_for(py::handle target, py::handle gradient) {
    require_same_layout(target, gradient);
    CpuTensorView gradient_view = cpu_tensor_view(gradient, "gradient");
    py::object output_layout = tensor_layout(target);
    CpuTensorAllocation output = allocate_cpu_tensor(output_layout, CpuDType::Float32);
    {
        py::gil_scoped_release release;
        std::vector<Index> key(output.view.leaf_rank(), 0);
        for (Index i = 0; i < output.view.logical_size; ++i) {
            output.view.write_float_expanded(
                key, -gradient_view.read_float_expanded(key)
            );
            output.view.cache->increment_key(key.data(), key.size());
        }
    }
    return make_tensor(std::move(output.data_object), std::move(output.layout_object));
}

inline CpuDType promote_cpu_binary_dtype(py::handle lhs, py::handle rhs) {
    CPU& lhs_data = cpu_data_from_tensor(lhs, "lhs");
    CPU& rhs_data = cpu_data_from_tensor(rhs, "rhs");
    if (lhs_data.dtype() == CpuDType::Float32 ||
        rhs_data.dtype() == CpuDType::Float32) {
        return CpuDType::Float32;
    }
    return CpuDType::Int32;
}

inline void write_int_result(
    CpuTensorView& view, const std::vector<Index>& key, long long value
) {
    view.write_int_expanded(key, checked_int32(value));
}

}  // namespace neotorch::data
