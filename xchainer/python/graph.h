#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace python_internal {

void InitXchainerGraph(pybind11::module&);

void InitXchainerBackpropScope(pybind11::module&);

}  // namespace python_internal
}  // namespace python
}  // namespace xchainer
