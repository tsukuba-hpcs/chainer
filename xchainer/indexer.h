#pragma once

#include <algorithm>
#include <cstdint>

#include <gsl/gsl>

#include "xchainer/constant.h"
#include "xchainer/macro.h"

namespace xchainer {

// Kernel object to index typed arrays.
//
// Indexer holds the shape information. It can be used to access elements of TypedArray by linear indexes.
template <int8_t n_dim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotalSize()) {
        Expects(shape.size() == n_dim);
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return n_dim; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t index) {
        raw_index_ = index;
        for (int8_t i = n_dim; --i >= 0;) {
            index_[i] = index % shape_[i];
            index /= shape_[i];
        }
    }

private:
    int64_t shape_[n_dim];
    int64_t index_[n_dim];
    int64_t raw_index_;
    int64_t total_size_;
};

// Dynamic-length indexer.
template <>
class Indexer<kDynamicNdim> {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotaSize()), ndim_(shape.ndim()) {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

private:
    int64_t shape_[kMaxNdim];
    int64_t index_[kMaxNdim];
    int64_t raw_index_;
    int64_t total_size_;
    int8_t ndim_;
};

}  // namespace xchainer
