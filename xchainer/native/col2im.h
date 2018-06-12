#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace internal {

Array Col2Im(
        const Array& col,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size);

}  // namespace internal
}  // namespace native
}  // namespace xchainer
