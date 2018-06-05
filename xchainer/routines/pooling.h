#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

Array MaxPooling(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all = true);

}  // namespace xchainer
