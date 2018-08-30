#pragma once

#include <pybind11/pybind11.h>

namespace xchainer {
namespace python {
namespace testing {
namespace testing_internal {

void InitXchainerDeviceBuffer(pybind11::module&);

}  // namespace testing_internal
}  // namespace testing
}  // namespace python
}  // namespace xchainer
