#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ostream>

#include "xchainer/constant.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"

namespace xchainer {

template <int8_t kNdim>
class IndexIterator;

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()} { std::copy(shape.begin(), shape.end(), shape_); }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(int64_t start, int64_t step = 1) const;

    // Sets an index from multiple indexers each of which composes a portion of dimensions in order.
    template <int8_t kNdimFirst, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest);

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }
    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }
    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int64_t total_size_{};
    int64_t shape_[kNdim]{};
};

// Dynamic-length specialization.
template <>
class Indexer<kDynamicNdim> {
public:
    explicit Indexer(const Shape& shape) : ndim_{shape.ndim()}, total_size_{shape.GetTotalSize()} {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(int64_t start, int64_t step = 1) const;

    template <int8_t kNdimFirst, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest);

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }
    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }
    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int8_t ndim_{};
    int64_t total_size_{};
    int64_t shape_[kMaxNdim]{};
};

template <int8_t kNdim = kDynamicNdim>
class IndexIterator {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const Indexer<kNdim>& indexer, int64_t start, int64_t step)
        : indexer_{indexer}, index_{}, raw_index_{0}, step_{step} {
        // backward iteration is not supported in order to omit lower-bound check for performance.
        assert(start >= 0);
        assert(step > 0);
        Set(start);
    }

    XCHAINER_HOST_DEVICE const Indexer<kNdim>& indexer() const { return indexer_; }
    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }
    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }
    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }
    XCHAINER_HOST_DEVICE operator bool() const { return ok(); }
    XCHAINER_HOST_DEVICE IndexIterator<kNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

private:
    XCHAINER_HOST_DEVICE bool ok() const { return raw_index_ < indexer_.total_size(); }

    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        if (indexer_.total_size() == 0) {
            // In this case there are some j such that shape[j] == 0.
            return;
        }
        const int64_t* shape = indexer_.shape();
        for (int8_t j = indexer_.ndim(); --j >= 0;) {
            index_[j] = i % shape[j];
            i /= shape[j];
        }
    }

    const Indexer<kNdim>& indexer_;
    int64_t index_[kNdim];
    int64_t raw_index_;
    const int64_t step_;
};

// Dynamic-length specialization.
template <>
class IndexIterator<kDynamicNdim> {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const Indexer<kDynamicNdim>& indexer, int64_t start, int64_t step)
        : indexer_{indexer}, index_{}, raw_index_{0}, step_{step} {
        assert(start >= 0);
        assert(step > 0);
        Set(start);
    }

    XCHAINER_HOST_DEVICE const Indexer<kDynamicNdim>& indexer() const { return indexer_; }
    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }
    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }
    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }
    XCHAINER_HOST_DEVICE operator bool() const { return ok(); }
    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

private:
    XCHAINER_HOST_DEVICE bool ok() const { return raw_index_ < indexer_.total_size(); }

    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        if (indexer_.total_size() == 0) {
            // In this case there are some j such that shape[j] == 0.
            return;
        }
        const int64_t* shape = indexer_.shape();
        for (int8_t j = indexer_.ndim(); --j >= 0;) {
            index_[j] = i % shape[j];
            i /= shape[j];
        }
    }

    const Indexer<kDynamicNdim>& indexer_;
    int64_t index_[kMaxNdim];
    int64_t raw_index_;
    const int64_t step_;
};

template <int8_t kNdim>
XCHAINER_HOST_DEVICE inline IndexIterator<kNdim> Indexer<kNdim>::It(int64_t start, int64_t step) const {
    return IndexIterator<kNdim>{*this, start, step};
}

XCHAINER_HOST_DEVICE inline IndexIterator<kDynamicNdim> Indexer<kDynamicNdim>::It(int64_t start, int64_t step) const {
    return IndexIterator<kDynamicNdim>{*this, start, step};
}

namespace internal {

template <int8_t kNdim, int8_t kNdimIter>
XCHAINER_HOST_DEVICE inline int8_t CombineIterators(IndexIterator<kNdim>& it, int8_t processed_dims, const IndexIterator<kNdimIter>& iter) {
    assert(processed_dims + iter.indexer().ndim() <= it.indexer().ndim());
    for (int8_t i = 0; i < iter.indexer().ndim(); ++i) {
        it.index()[processed_dims + i] = iter.index()[i];
    }
    return processed_dims + iter.indexer().ndim();
}

// Combine multiple sub-iterators to make a combined iterator.
// Returns the number of written dimensions, which is equal to ndim_.
// `processed_dims` is the number of written dimensions so far.
template <int8_t kNdim, int8_t kNdimFirstIter, typename... IndexIterators>
XCHAINER_HOST_DEVICE int8_t
CombineIterators(IndexIterator<kNdim>& it, int8_t processed_dims, const IndexIterator<kNdimFirstIter>& first, IndexIterators&&... rest) {
    processed_dims = CombineIterators(it, processed_dims, first);
    int8_t dims = CombineIterators(it, processed_dims, std::forward<IndexIterators>(rest)...);
    assert(dims == it.indexer().ndim());
    return dims;
}

template <int8_t kNdim, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kNdim> CombinedIterator(IndexIterator<kNdim>& it, IndexIterators&&... rest) {
    int8_t processed_dims = internal::CombineIterators<kNdim>(it, 0, std::forward<IndexIterators>(rest)...);
    assert(processed_dims == it.indexer().ndim());
#ifndef NDEBUG
    for (int8_t i = 0; i < it.indexer().ndim(); ++i) {
        assert(0 <= it.index()[i]);
        assert(it.index()[i] < it.indexer().shape()[i]);
    }
#endif
    return it;
}

}  // namespace internal

// Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
template <int8_t kNdim>
template <int8_t kNdimFirst, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kNdim> Indexer<kNdim>::It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest) {
    auto it = It(0);
    return internal::CombinedIterator<kNdim>(it, first, std::forward<IndexIterators>(rest)...);
}

// Dynamic-length specialization.
template <int8_t kNdimFirst, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> Indexer<kDynamicNdim>::It(
        const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest) {
    auto it = It(0);
    return internal::CombinedIterator<kDynamicNdim>(it, first, std::forward<IndexIterators>(rest)...);
}

template <int8_t kNdim>
inline std::ostream& operator<<(std::ostream& os, const Indexer<kNdim>& indexer) {
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << ")";
}

}  // namespace xchainer
