#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ostream>

#include "xchainer/constant.h"
#include "xchainer/macro.h"
#include "xchainer/ndim_vector.h"
#include "xchainer/shape.h"

namespace xchainer {

class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotalSize()), ndim_(shape.ndim()) {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        assert(0 <= i);
        assert(i < total_size_);
        raw_index_ = i;
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

    // Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
    template <typename... Args>
    XCHAINER_HOST_DEVICE void SetIndexers(Args&&... indexers) {
        int8_t processed_dims = SetIndexersImpl(0, std::forward<Args>(indexers)...);
        assert(processed_dims == ndim_);
#ifndef NDEBUG
        for (int8_t i = 0; i < ndim_; ++i) {
            assert(0 <= index_[i]);
            assert(index_[i] < shape_[i]);
        }
#endif
    }

private:
    // Implementation of SetIndexers.
    // Returns the number of written dimensions, which is equal to ndim_.
    // `processed_dims` is the number of written dimensions so far.
    template <typename... Args>
    XCHAINER_HOST_DEVICE int8_t SetIndexersImpl(int8_t processed_dims, const Indexer& first_indexer, Args&&... indexers) {
        processed_dims = SetIndexersImpl(processed_dims, first_indexer);
        int8_t dims = SetIndexersImpl(processed_dims, std::forward<Args>(indexers)...);
        assert(dims == ndim_);
        return dims;
    }

    XCHAINER_HOST_DEVICE int8_t SetIndexersImpl(int8_t processed_dims, const Indexer& indexer) {
        assert(processed_dims + indexer.ndim_ <= ndim_);
        for (int8_t i = 0; i < indexer.ndim_; ++i) {
            index_[processed_dims + i] = indexer.index_[i];
        }
        return processed_dims + indexer.ndim_;
    }

    int64_t shape_[kMaxNdim]{};
    int64_t index_[kMaxNdim]{};
    int64_t raw_index_{};
    int64_t total_size_{};
    int8_t ndim_{};
};

inline std::ostream& operator<<(std::ostream& os, const Indexer& indexer) {
    NdimVector<int64_t> index_vec{indexer.index(), indexer.index() + indexer.ndim()};
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    Shape index{indexer.index(), indexer.index() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << " index=" << index << ")";
}

}  // namespace xchainer
