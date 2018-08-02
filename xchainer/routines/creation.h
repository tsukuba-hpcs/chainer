#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace internal {

// Returns the minimum number of bytes required to pack the data with specified strides and shape.
size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t item_size);

// Creates an array with given data packed with specified strides
Array FromHostData(
        const Shape& shape,
        Dtype dtype,
        const std::shared_ptr<void>& data,
        const Strides& strides,
        int64_t offset,
        Device& device = GetDefaultDevice());

// Creates an array with given contiguous data
Array FromContiguousHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device = GetDefaultDevice());

// Creates an empty array with specified strides.
Array Empty(const Shape& shape, Dtype dtype, const Strides& strides, Device& device = GetDefaultDevice());

// Creates an empty array with reduced shape.
Array EmptyReduced(const Shape& shape, Dtype dtype, const Axes& axes, bool keepdims, Device& device = GetDefaultDevice());

}  // namespace internal

// Creates an array with given data without copying.
//
// The data must reside in the specified device.
// If strides are not given, the data is considered as a contiguous data.
Array FromData(
        const Shape& shape,
        Dtype dtype,
        const std::shared_ptr<void>& data,
        const nonstd::optional<Strides>& strides = nonstd::nullopt,
        int64_t offset = 0,
        Device& device = GetDefaultDevice());

Array Empty(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar fill_value, Dtype dtype, Device& device = GetDefaultDevice());
Array Full(const Shape& shape, Scalar fill_value, Device& device = GetDefaultDevice());
Array Zeros(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());
Array Ones(const Shape& shape, Dtype dtype, Device& device = GetDefaultDevice());

Array Arange(Scalar start, Scalar stop, Scalar step, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Scalar step, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar start, Scalar stop, Device& device = GetDefaultDevice());
Array Arange(Scalar stop, Dtype dtype, Device& device = GetDefaultDevice());
Array Arange(Scalar stop, Device& device = GetDefaultDevice());

// Creates an array which has the same shape and dtype as the other array.
// The new array is allocated in the default device. The device of the other array
// is ignored.
Array EmptyLike(const Array& a, Device& device = GetDefaultDevice());
Array FullLike(const Array& a, Scalar fill_value, Device& device = GetDefaultDevice());
Array ZerosLike(const Array& a, Device& device = GetDefaultDevice());
Array OnesLike(const Array& a, Device& device = GetDefaultDevice());

// Creates a copy.
// It will be connected to all the graphs.
// It will be always C-contiguous.
Array Copy(const Array& a);

// Creates the identity array.
Array Identity(int64_t n, Dtype dtype, Device& device = GetDefaultDevice());

// Creates a 2-dimensional array with ones along the k-th diagonal and zeros elsewhere.
Array Eye(
        int64_t n,
        nonstd::optional<int64_t> m,
        nonstd::optional<int64_t> k,
        nonstd::optional<Dtype> dtype,
        Device& device = GetDefaultDevice());

// Returns a C-contiguous array.
Array AsContiguousArray(const Array& a, const nonstd::optional<Dtype>& dtype = nonstd::nullopt);

Array Diag(const Array& v, int64_t k = 0, Device& device = GetDefaultDevice());

Array Diagflat(const Array& v, int64_t k = 0, Device& device = GetDefaultDevice());

// Creates a 1-d array with evenly spaced numbers.
Array Linspace(
        Scalar start,
        Scalar stop,
        const nonstd::optional<int64_t>& num = nonstd::nullopt,
        bool endpoint = true,
        const nonstd::optional<Dtype>& dtype = nonstd::nullopt,
        Device& device = GetDefaultDevice());

}  // namespace xchainer
