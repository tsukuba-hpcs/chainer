#pragma once

#include <functional>
#include <initializer_list>

#include "xchainer/array.h"
#include "xchainer/error.h"

namespace xchainer {
namespace internal {

void CheckNoInplaceWithRequiredGrad(const Array& out, std::initializer_list<std::reference_wrapper<const Array>> inputs) {
    if (internal::HasAnyArrayNode(out)) {
        throw XchainerError{"In-place assignment to output array requiring grad is not allowed."};
    }

    bool any_grad_required = false;
    bool any_inplace = false;
    for (const Array& input : inputs) {
        any_grad_required |= input.IsGradRequired(AnyGraph{});
        any_inplace |= (out.body() == input.body());
    }

    if (any_grad_required && any_inplace) {
        throw XchainerError{"In-place assignment that involves input arrays requiring grad is not allowed."};
    }
}

}  // namespace internal
}  // namespace xchainer
