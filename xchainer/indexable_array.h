#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/indexer.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace indexable_array_detail {

// Adds `const` to To if From is const
template <typename To, typename From>
using WithConstnessOf = std::conditional_t<std::is_const<From>::value, std::add_const_t<To>, std::remove_const_t<To>>;

}  // namespace indexable_array_detail

// Statically typed array data structure that can be passed to device kernels.
//
// TypedArary only contains the pointer to the first element and the strides information. To access elements with
// linear indexes, Indexer is also needed.
template <typename T, int8_t kNdim = kDynamicNdim>
class IndexableArray {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data} {
        assert(strides.ndim() == kNdim);
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(const Array& array)
        : IndexableArray{reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(array.raw_data()) + array.offset()), array.strides()} {
        assert(TypeToDtype<T> == array.dtype());
#ifndef NDEBUG
        gsl::span<const uint8_t> data_range = array.GetDataRange();
        first_ = data_range.data();
        last_ = first_ + data_range.size_bytes();
#endif
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto data_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<uint8_t, T>*>(data_);
        for (int8_t dim = 0; dim < kNdim; ++dim) {
            data_ptr += strides_[dim] * index[dim];
        }
        assert(first_ == nullptr || first_ <= data_ptr);
        assert(last_ == nullptr || data_ptr <= last_ - sizeof(T));  // catch for less than sizeof(T) bytes overflow
        return *reinterpret_cast<T*>(data_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<kNdim>& indexer) const { return operator[](indexer.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const std::vector<int8_t>& axes) {
        assert(axes.size() <= static_cast<size_t>(kNdim));
        int64_t c[kNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < axes.size(); ++i) {
            strides_[i] = c[axes[i]];
        }
        return *this;
    }

private:
    T* data_;
#ifndef NDEBUG
    const uint8_t* first_ = nullptr;
    const uint8_t* last_ = nullptr;
#endif
    int64_t strides_[kNdim];
};

// IndexableArray with dynamic ndim.
template <typename T>
class IndexableArray<T, kDynamicNdim> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data}, ndim_{strides.ndim()} {
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(const Array& array)
        : IndexableArray{reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(array.raw_data()) + array.offset()), array.strides()} {
        assert(TypeToDtype<T> == array.dtype());

#ifndef NDEBUG
        gsl::span<const uint8_t> data_range = array.GetDataRange();
        first_ = data_range.data();
        last_ = first_ + data_range.size_bytes();
#endif
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto data_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<uint8_t, T>*>(data_);
        for (int8_t dim = 0; dim < ndim_; ++dim) {
            data_ptr += strides_[dim] * index[dim];
        }
        assert(first_ == nullptr || first_ <= data_ptr);
        assert(last_ == nullptr || data_ptr <= last_ - sizeof(T));  // catch less than sizeof(T) bytes overflow
        return *reinterpret_cast<T*>(data_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<kDynamicNdim>& indexer) const { return operator[](indexer.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const std::vector<int8_t>& axes) {
        assert(axes.size() <= static_cast<size_t>(ndim_));
        int64_t c[kMaxNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < axes.size(); ++i) {
            strides_[i] = c[axes[i]];
        }
        ndim_ = static_cast<int8_t>(axes.size());
        return *this;
    }

private:
    T* data_;
#ifndef NDEBUG
    const uint8_t* first_ = nullptr;
    const uint8_t* last_ = nullptr;
#endif
    int64_t strides_[kMaxNdim];
    int8_t ndim_;
};

}  // namespace xchainer
