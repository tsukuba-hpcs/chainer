#pragma once

#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "xchainer/array_body.h"

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

ArrayBodyPtr MakeArray(pybind11::handle object, pybind11::handle dtype, bool copy, pybind11::handle device);

// Makes an array from a NumPy array. Shape, dtype, strides will be kept.
ArrayBodyPtr MakeArrayFromNumpyArray(pybind11::array array, Device& device);

void InitXchainerArray(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
