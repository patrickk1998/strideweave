#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "_cpu_carrier.hpp"

namespace py = pybind11;

namespace strideweave::carrier {

// Layout, coordinate, and axis mechanics shared by every operation family
// that walks a hierarchical shape rather than one flat elementwise key.

inline Index product_extents(const std::vector<Index>& extents);

struct CpuModeInfo {
    py::object shape;
    std::vector<Index> leaf_extents;

    Index logical_size() const { return product_extents(leaf_extents); }
};

struct CpuLayoutModes {
    std::vector<CpuModeInfo> modes;
    std::vector<Index> leaf_counts;
};

inline void flatten_mode_shape(py::handle shape, std::vector<Index>& leaves) {
    if (strideweave::layout_index::is_int(shape)) {
        leaves.push_back(py::cast<Index>(shape));
        return;
    }
    for (py::handle child : py::reinterpret_borrow<py::sequence>(shape)) {
        flatten_mode_shape(child, leaves);
    }
}

inline CpuLayoutModes cpu_layout_modes(py::handle tensor, const char*) {
    py::object top_level = tensor_layout(tensor).attr("shape").attr("top_level");
    CpuLayoutModes result;
    for (py::handle shape : py::reinterpret_borrow<py::sequence>(top_level)) {
        CpuModeInfo mode;
        mode.shape = py::reinterpret_borrow<py::object>(shape);
        flatten_mode_shape(shape, mode.leaf_extents);
        result.leaf_counts.push_back(static_cast<Index>(mode.leaf_extents.size()));
        result.modes.push_back(std::move(mode));
    }
    return result;
}

inline Index normalize_axis(Index axis, Index rank) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw py::value_error("axis is out of range");
    }
    return axis;
}

inline Index product_extents(const std::vector<Index>& extents) {
    Index result = 1;
    for (Index extent : extents) {
        if (extent < 0 ||
            (extent != 0 && result > std::numeric_limits<Index>::max() / extent)) {
            throw py::value_error("layout extent is too large");
        }
        result *= extent;
    }
    return result;
}

inline Index floor_division(Index numerator, Index denominator) {
    Index quotient = numerator / denominator;
    const Index remainder = numerator % denominator;
    if (remainder < 0) {
        --quotient;
    }
    return quotient;
}

inline std::vector<Index> decode_ordinal(Index ordinal,
                                         const std::vector<Index>& leaves) {
    std::vector<Index> key(leaves.size(), 0);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const Index extent = leaves[i];
        key[i] = extent == 0 ? 0 : ordinal % extent;
        if (extent != 0) {
            ordinal /= extent;
        }
    }
    return key;
}

// Encode expanded (leaf-level) coordinates in StrideWeave's first-mode-fast
// order.  This is used by selection VJPs to retain a permutation keyed by the
// logical output ordinal, including nested modes.
inline Index expanded_key_ordinal(const std::vector<Index>& key,
                                  const std::vector<Index>& extents) {
    if (key.size() != extents.size()) {
        throw py::value_error("expanded coordinate rank does not match layout");
    }
    Index ordinal = 0;
    Index factor = 1;
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (key[i] < 0 || key[i] >= extents[i]) {
            throw py::value_error("expanded coordinate is outside layout");
        }
        ordinal += key[i] * factor;
        factor *= extents[i];
    }
    return ordinal;
}

inline std::vector<Index> mode_leaf_offsets(const CpuLayoutModes& modes) {
    std::vector<Index> offsets(modes.modes.size(), 0);
    Index offset = 0;
    for (std::size_t i = 0; i < modes.modes.size(); ++i) {
        offsets[i] = offset;
        offset += static_cast<Index>(modes.modes[i].leaf_extents.size());
    }
    return offsets;
}

inline py::object canonical_layout_from_top_level(py::object top_level) {
    py::object layout_module = py::module_::import("strideweave.layout");
    py::object shape_type = layout_module.attr("Shape");
    py::object stride_type = layout_module.attr("Stride");
    py::object layout_type = layout_module.attr("Layout");
    py::object shape = shape_type(std::move(top_level));
    auto [stride_level, _] = canonical_stride_level(shape.attr("top_level"), 1);
    return layout_type(std::move(shape), stride_type(std::move(stride_level)));
}

inline std::vector<Index>
outer_key_for_ordinal(Index ordinal, const CpuLayoutModes& modes, Index axis) {
    std::vector<Index> key;
    for (Index i = 0; i < static_cast<Index>(modes.modes.size()); ++i) {
        if (i == axis) {
            continue;
        }
        const Index size = modes.modes[static_cast<std::size_t>(i)].logical_size();
        const Index mode_ordinal = size == 0 ? 0 : ordinal % size;
        if (size != 0) {
            ordinal /= size;
        }
        std::vector<Index> mode_key = decode_ordinal(
            mode_ordinal, modes.modes[static_cast<std::size_t>(i)].leaf_extents);
        key.insert(key.end(), mode_key.begin(), mode_key.end());
    }
    return key;
}

}  // namespace strideweave::carrier
