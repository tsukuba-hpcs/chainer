#pragma once

#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "chainerx/dtype.h"

namespace chainerx {
namespace python {
namespace python_internal {

void InitXchainerDtype(pybind11::module&);

Dtype GetDtypeFromString(const std::string& name);

Dtype GetDtypeFromNumpyDtype(const pybind11::dtype& npdtype);

Dtype GetDtype(pybind11::handle handle);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
