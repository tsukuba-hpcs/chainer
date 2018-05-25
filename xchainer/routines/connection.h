#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

namespace internal {

int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all);

}  // namespace internal

// Computes the n-dimensional convolution.
//
// x: (batch_size, in_channels, in_1, in_2, ..., in_n)
// w: (out_channels, in_channels, k_1, k_2, ..., k_n)
// b: (out_channels)
//
// Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all = false);

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size = nonstd::nullopt);

}  // namespace xchainer
