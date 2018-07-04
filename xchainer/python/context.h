#pragma once

#include <pybind11/pybind11.h>

#include "xchainer/context.h"

namespace xchainer {
namespace python {
namespace internal {

Context& GetContext(pybind11::handle handle);

void InitXchainerContext(pybind11::module&);

void InitXchainerContextScope(pybind11::module&);

}  // namespace internal
}  // namespace python
}  // namespace xchainer
