#pragma once

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/constant.h"
#include "xchainer/scalar.h"
#include "xchainer/stack_vector.h"

namespace xchainer {

// Computes the batch normalization along the given axis.
// If axis is omitted, the first axis is treated as the batch axis and will be reduced during normalization.
// Running mean and running variance that are passed as arguments will be updated in-place.
Array BatchNormalization(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps = 2e-5,
        Scalar decay = 0.9,
        const OptionalAxes& axis = nonstd::nullopt);

}  // namespace xchainer
