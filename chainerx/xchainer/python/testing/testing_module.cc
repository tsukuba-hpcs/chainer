#include "chainerx/python/testing/testing_module.h"

#include "chainerx/python/testing/device_buffer.h"

#include <pybind11/pybind11.h>

#include "chainerx/python/array.h"
#include "chainerx/python/device.h"

namespace chainerx {
namespace python {
namespace testing {
namespace testing_internal {

namespace py = pybind11;

void InitXchainerTestingModule(pybind11::module& m) {
    InitXchainerDeviceBuffer(m);

    // Converts from NumPy array. It supports keepstrides option.
    m.def("_fromnumpy",
          [](py::array array, bool keepstrides, py::handle device) {
              if (keepstrides) {
                  return python_internal::MakeArrayFromNumpyArray(array, python_internal::GetDevice(device));
              }
              return python_internal::MakeArray(array, array.dtype(), true, device);
          },
          py::arg("array"),
          py::arg("keepstrides") = false,
          py::arg("device") = nullptr);
}

}  // namespace testing_internal
}  // namespace testing
}  // namespace python
}  // namespace chainerx
