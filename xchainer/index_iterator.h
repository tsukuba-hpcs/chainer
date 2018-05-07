#pragma once

#include <cassert>
#include <cstdint>

#include "xchainer/constant.h"
#include "xchainer/macro.h"

namespace xchainer {

template <int8_t kNdim = kDynamicNdim>
class IndexIterator {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, total_size_{total_size}, raw_index_{0}, step_{step}, index_{} {
        // backward iteration is not supported in order to omit lower-bound check for performance.
        assert(start >= 0);
        assert(step > 0);
        Set(start);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }

    XCHAINER_HOST_DEVICE constexpr int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        if (total_size_ == 0) {
            // In this case there are some j such that shape[j] == 0.
            return;
        }
        for (int8_t j = kNdim; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

    const int64_t* shape_;
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t step_{};
    int64_t index_[kNdim];
};

// Dynamic-length specialization.
template <>
class IndexIterator<kDynamicNdim> {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int8_t ndim, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, ndim_{ndim}, total_size_{total_size}, raw_index_{0}, step_{step}, index_{} {
        assert(start >= 0);
        assert(step > 0);
        Set(start);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        if (total_size_ == 0) {
            // In this case there are some j such that shape[j] == 0.
            return;
        }
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

    const int64_t* shape_;
    int8_t ndim_{};
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t step_{};
    int64_t index_[kMaxNdim];
};

}  // namespace xchainer
