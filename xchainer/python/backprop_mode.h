#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/backprop_mode.h"

namespace xchainer {
namespace python {
namespace python_internal {

void InitXchainerBackpropMode(pybind11::module&);

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
