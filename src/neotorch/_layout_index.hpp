#pragma once

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace neotorch::layout_index {

using Index = long long;

enum class CacheEntryKind {
    Integer,
    Push,
    Pop,
};

struct CacheEntry {
    CacheEntryKind kind;
    Index value;
};

struct LayoutNode {
    bool leaf = false;
    Index shape = 1;
    Index stride = 0;
    Index logical_size = 1;
    Index max_index = 0;
    std::vector<LayoutNode> children;
};

struct ModeRange {
    std::size_t begin = 0;
    std::size_t end = 0;
};

inline bool is_int(py::handle value) { return py::isinstance<py::int_>(value); }

inline Index as_index(py::handle value) { return py::cast<Index>(value); }

inline Index logical_size(py::handle shape_level) {
    return as_index(shape_level.attr("logical_size"));
}

inline Index python_mod(Index value, Index divisor) {
    Index result = value % divisor;
    if (result < 0) {
        result += divisor;
    }
    return result;
}

class LayoutCache {
public:
    explicit LayoutCache(py::object layout) {
        py::object shape = layout.attr("shape").attr("top_level");
        py::object stride = layout.attr("stride").attr("top_level");
        construction_shape_entries_ = &shape_entries_;
        construction_stride_entries_ = &stride_entries_;
        try {
            root_ = parse_level(shape, stride, false);
        } catch (...) {
            construction_shape_entries_ = nullptr;
            construction_stride_entries_ = nullptr;
            throw;
        }
        construction_shape_entries_ = nullptr;
        construction_stride_entries_ = nullptr;
        logical_size_ = root_.logical_size;
        cosize_ = root_.max_index + 1;
        rank_ = static_cast<Index>(root_.children.size());
        collect_leaf_metadata();
    }

    Index logical_size() const { return logical_size_; }

    Index cosize() const { return cosize_; }

    Index rank() const { return rank_; }

    Index leaf_rank() const {
        return static_cast<Index>(leaf_shapes_.size());
    }

    const std::vector<CacheEntry>& shape_entries() const { return shape_entries_; }

    const std::vector<CacheEntry>& stride_entries() const { return stride_entries_; }

    const std::vector<Index>& leaf_shapes() const { return leaf_shapes_; }

    const std::vector<Index>& leaf_strides() const { return leaf_strides_; }

    std::vector<Index> expand_key(Index logical_key) const {
        std::vector<Index> expanded;
        expanded.reserve(leaf_shapes_.size());
        expand_logical_key(root_, logical_key, expanded);
        return expanded;
    }

    Index index_logical(Index logical_key) const {
        return index_logical_node(root_, logical_key);
    }

    Index index(const std::vector<Index>& key) const {
        return index(key.data(), key.size());
    }

    Index index(const Index* key, std::size_t size) const {
        std::size_t key_position = 0;
        return index_coordinate_node(root_, key, size, key_position);
    }

    Index index_expanded(const std::vector<Index>& key) const {
        return index_expanded(key.data(), key.size());
    }

    Index index_expanded(const Index* key, std::size_t size) const {
        if (size != leaf_shapes_.size()) {
            throw py::value_error("Expanded key rank does not match layout rank");
        }

        Index index = 0;
        for (std::size_t i = 0; i < size; ++i) {
            if (key[i] < 0 || key[i] >= leaf_shapes_[i]) {
                throw py::value_error("Key is not in domain of shape");
            }
            index += leaf_strides_[i] * key[i];
        }
        return index;
    }

    Index get_index(py::handle key) const {
        return index_python_key(root_, key);
    }

    void increment_key(Index* key, std::size_t size) const {
        if (size != leaf_shapes_.size()) {
            throw py::value_error("Expanded key rank does not match layout rank");
        }
        increment_range(key, 0, size);
    }

    void increment_mode(Index* key, std::size_t size, Index mode) const {
        if (size != leaf_shapes_.size()) {
            throw py::value_error("Expanded key rank does not match layout rank");
        }
        if (mode < 0 || mode >= rank_) {
            throw py::value_error("Mode is not in domain of layout");
        }
        const ModeRange& range = top_level_mode_ranges_[static_cast<std::size_t>(mode)];
        increment_range(key, range.begin, range.end);
    }

private:
    static LayoutNode parse_level(
        py::handle shape_object, py::handle stride_object, bool emit_markers
    ) {
        py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
        py::sequence stride = py::reinterpret_borrow<py::sequence>(stride_object);

        if (py::len(shape) != py::len(stride)) {
            throw py::value_error("Shape and Stride Lengths do not match");
        }

        LayoutNode node;
        node.leaf = false;
        node.logical_size = 1;
        node.max_index = 0;

        if (emit_markers) {
            current_shape_entries().push_back({CacheEntryKind::Push, 0});
            current_stride_entries().push_back({CacheEntryKind::Push, 0});
        }

        for (py::ssize_t i = 0; i < py::len(shape); ++i) {
            py::object shape_value = shape[i];
            py::object stride_value = stride[i];
            LayoutNode child;
            if (is_int(shape_value)) {
                if (!is_int(stride_value)) {
                    throw py::value_error("Shape and Stride do not match in Structure");
                }
                child.leaf = true;
                child.shape = as_index(shape_value);
                child.stride = as_index(stride_value);
                child.logical_size = child.shape;
                child.max_index = child.stride * (child.shape - 1);
                current_shape_entries().push_back(
                    {CacheEntryKind::Integer, child.shape}
                );
                current_stride_entries().push_back(
                    {CacheEntryKind::Integer, child.stride}
                );
            } else {
                if (is_int(stride_value)) {
                    throw py::value_error("Shape and Stride do not match in Structure");
                }
                child = parse_level(shape_value, stride_value, true);
            }
            node.logical_size *= child.logical_size;
            node.max_index += child.max_index;
            node.children.push_back(std::move(child));
        }

        if (emit_markers) {
            current_shape_entries().push_back({CacheEntryKind::Pop, 0});
            current_stride_entries().push_back({CacheEntryKind::Pop, 0});
        }

        return node;
    }

    static std::vector<CacheEntry>& current_shape_entries() {
        return *construction_shape_entries_;
    }

    static std::vector<CacheEntry>& current_stride_entries() {
        return *construction_stride_entries_;
    }

    void collect_leaf_metadata() {
        leaf_shapes_.clear();
        leaf_strides_.clear();
        top_level_mode_ranges_.clear();
        top_level_mode_ranges_.reserve(root_.children.size());

        for (const LayoutNode& child : root_.children) {
            const std::size_t begin = leaf_shapes_.size();
            collect_leaf_metadata(child);
            top_level_mode_ranges_.push_back({begin, leaf_shapes_.size()});
        }
    }

    void collect_leaf_metadata(const LayoutNode& node) {
        if (node.leaf) {
            leaf_shapes_.push_back(node.shape);
            leaf_strides_.push_back(node.stride);
            return;
        }

        for (const LayoutNode& child : node.children) {
            collect_leaf_metadata(child);
        }
    }

    void increment_range(Index* key, std::size_t begin, std::size_t end) const {
        for (std::size_t i = begin; i < end; ++i) {
            ++key[i];
            if (key[i] < leaf_shapes_[i]) {
                return;
            }
            key[i] = 0;
        }
    }

    void expand_logical_key(
        const LayoutNode& node, Index logical_key, std::vector<Index>& expanded
    ) const {
        if (logical_key < 0 || logical_key >= node.logical_size) {
            throw py::value_error("Key is not in domain of shape");
        }

        for (const LayoutNode& child : node.children) {
            Index coordinate_value = python_mod(logical_key, child.logical_size);
            logical_key -= coordinate_value;
            logical_key /= child.logical_size;

            if (child.leaf) {
                expanded.push_back(coordinate_value);
            } else {
                expand_logical_key(child, coordinate_value, expanded);
            }
        }
    }

    Index index_logical_node(const LayoutNode& node, Index logical_key) const {
        if (logical_key < 0 || logical_key >= node.logical_size) {
            throw py::value_error("Key is not in domain of shape");
        }

        Index index = 0;
        for (const LayoutNode& child : node.children) {
            Index coordinate_value = python_mod(logical_key, child.logical_size);
            logical_key -= coordinate_value;
            logical_key /= child.logical_size;

            if (child.leaf) {
                if (coordinate_value < 0 || coordinate_value >= child.shape) {
                    throw py::value_error("Key is not in domain of shape");
                }
                index += child.stride * coordinate_value;
            } else {
                index += index_logical_node(child, coordinate_value);
            }
        }
        return index;
    }

    Index index_coordinate_node(
        const LayoutNode& node,
        const Index* key,
        std::size_t size,
        std::size_t& key_position
    ) const {
        Index index = 0;
        for (const LayoutNode& child : node.children) {
            if (key_position >= size) {
                break;
            }
            const Index coordinate_value = key[key_position];
            ++key_position;

            if (child.leaf) {
                if (coordinate_value < 0 || coordinate_value >= child.shape) {
                    throw py::value_error("Key is not in domain of shape");
                }
                index += child.stride * coordinate_value;
            } else {
                index += index_logical_node(child, coordinate_value);
            }
        }
        return index;
    }

    Index index_python_key(const LayoutNode& node, py::handle key) const {
        if (is_int(key)) {
            return index_logical_node(node, as_index(key));
        }

        py::sequence sequence = py::reinterpret_borrow<py::sequence>(key);
        Index index = 0;
        const py::ssize_t sequence_size = static_cast<py::ssize_t>(py::len(sequence));
        const py::ssize_t children_size =
            static_cast<py::ssize_t>(node.children.size());
        const py::ssize_t limit =
            sequence_size < children_size ? sequence_size : children_size;
        for (py::ssize_t i = 0; i < limit; ++i) {
            const LayoutNode& child = node.children[static_cast<std::size_t>(i)];
            py::object coordinate = sequence[i];
            if (child.leaf) {
                const Index coordinate_value = as_index(coordinate);
                if (coordinate_value < 0 || coordinate_value >= child.shape) {
                    throw py::value_error("Key is not in domain of shape");
                }
                index += child.stride * coordinate_value;
            } else {
                index += index_python_key(child, coordinate);
            }
        }
        return index;
    }

    LayoutNode root_;
    Index logical_size_ = 1;
    Index cosize_ = 1;
    Index rank_ = 0;
    std::vector<CacheEntry> shape_entries_;
    std::vector<CacheEntry> stride_entries_;
    std::vector<Index> leaf_shapes_;
    std::vector<Index> leaf_strides_;
    std::vector<ModeRange> top_level_mode_ranges_;

    inline static thread_local std::vector<CacheEntry>* construction_shape_entries_ =
        nullptr;
    inline static thread_local std::vector<CacheEntry>* construction_stride_entries_ =
        nullptr;
};

inline std::vector<py::object> expand_int(
    py::handle key_object, py::handle shape_object
) {
    Index key = as_index(key_object);
    if (key < 0 || key >= logical_size(shape_object)) {
        throw py::value_error("Key is not in domain of shape");
    }

    py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    std::vector<py::object> coordinate;
    coordinate.reserve(static_cast<std::size_t>(py::len(shape)));

    for (py::handle element : shape) {
        Index level_size = is_int(element) ? as_index(element) : logical_size(element);
        Index coordinate_value = python_mod(key, level_size);
        coordinate.emplace_back(py::int_(coordinate_value));
        key -= coordinate_value;
        key /= level_size;
    }

    return coordinate;
}

inline Index get_index_levels(
    py::handle shape_object, py::handle stride_object, py::handle key
) {
    py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    py::sequence stride = py::reinterpret_borrow<py::sequence>(stride_object);

    if (py::len(shape) != py::len(stride)) {
        throw py::value_error("Shape and Stride Lengths do not match");
    }

    std::vector<py::object> expanded_key;
    py::iterator key_iterator;
    const bool key_is_int = is_int(key);
    if (key_is_int) {
        expanded_key = expand_int(key, shape);
    } else {
        key_iterator = py::iter(key);
    }

    Index index = 0;
    for (py::ssize_t i = 0; i < py::len(shape); ++i) {
        py::object coordinate;
        if (key_is_int) {
            coordinate = expanded_key[static_cast<std::size_t>(i)];
        } else {
            if (key_iterator == py::iterator::sentinel()) {
                break;
            }
            coordinate = py::reinterpret_borrow<py::object>(*key_iterator);
            ++key_iterator;
        }

        py::object shape_value = shape[i];
        py::object stride_value = stride[i];
        if (is_int(shape_value)) {
            Index coordinate_value = as_index(coordinate);
            if (coordinate_value < 0 || coordinate_value >= as_index(shape_value)) {
                throw py::value_error("Key is not in domain of shape");
            }
            index += as_index(stride_value) * coordinate_value;
        } else {
            index += get_index_levels(shape_value, stride_value, coordinate);
        }
    }

    return index;
}

inline const LayoutCache& cache_from_layout(py::handle layout) {
    py::object cache = layout.attr("_cache");
    return py::cast<const LayoutCache&>(cache);
}

inline Index get_index(py::object layout, py::object key) {
    return cache_from_layout(layout).get_index(key);
}

inline Index max_index_levels(py::handle shape_object, py::handle stride_object) {
    py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    py::sequence stride = py::reinterpret_borrow<py::sequence>(stride_object);

    if (py::len(shape) != py::len(stride)) {
        throw py::value_error("Shape and Stride Lengths do not match");
    }

    Index max_index = 0;
    for (py::ssize_t i = 0; i < py::len(shape); ++i) {
        py::object shape_value = shape[i];
        py::object stride_value = stride[i];
        if (is_int(shape_value)) {
            max_index += as_index(stride_value) * (as_index(shape_value) - 1);
        } else {
            max_index += max_index_levels(shape_value, stride_value);
        }
    }

    return max_index;
}

inline Index cosize(py::object layout) {
    return cache_from_layout(layout).cosize();
}

}  // namespace neotorch::layout_index
