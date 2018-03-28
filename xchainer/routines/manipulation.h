#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/shape.h"

namespace xchainer {

// Returns a transposed view of the array.
Array Transpose(const Array& a);

// Returns a reshaped array.
// TODO(niboshi): Support reshape that require a copy.
// TODO(niboshi): Support shape with dimension -1.
Array Reshape(const Array& a, const Shape& shape);

// Returns a squeezed array with unit-length axes removed.
//
// If no axes are specified, all axes of unit-lengths are removed.
// If no axes can be removed, an array with aliased data is returned.
Array Squeeze(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt);

// Broadcasts the array to the specified shape.
// Returned array is always a view to this array.
Array BroadcastTo(const Array& array, const Shape& shape);

}  // namespace xchainer
