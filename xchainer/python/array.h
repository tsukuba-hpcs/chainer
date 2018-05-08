#pragma once

#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "xchainer/array.h"

namespace xchainer {
namespace python {
namespace internal {

// TODO(beam2d): The current binding has an overhead on wrapping ArrayBodyPtr by Array, which copies shared_ptr. One
// simple way to avoid this overhead is to use reinterpret_cast<Array&>(ptr). This cast is valid if ArrayBodyPtr (i.e.,
// shared_ptr) satisfies "standard layout" conditions. We can test if ArrayBodyPtr satisfies these conditions by
// std::is_standard_layout (see http://en.cppreference.com/w/cpp/types/is_standard_layout#Notes).

using ArrayBody = xchainer::internal::ArrayBody;
using ArrayBodyPtr = std::shared_ptr<ArrayBody>;
using ConstArrayBodyPtr = std::shared_ptr<const ArrayBody>;

// If `reuse_numpy_strides` is true and `object` is a NumPy array, its strides is used as-is.
ArrayBodyPtr MakeArray(
        pybind11::handle object, pybind11::handle dtype, bool copy, pybind11::handle device, bool reuse_numpy_strides = false);

void InitXchainerArray(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
