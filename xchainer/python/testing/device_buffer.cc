#include "xchainer/python/testing/device_buffer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/shape.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace testing {
namespace internal {

namespace py = pybind11;  // standard convention

// A device buffer that upon construction allocates device memory and creates a py::buffer_info, sharing ownership of the managed data
// (py::buffer_info only holds a raw pointer and does not manage the lifetime of the pointed data). Memoryviews created from this buffer
// will also share ownership. Note that accessing the .obj attribute of a memoryview may increase the reference count and should thus be
// avoided.
class PyDeviceBuffer {
public:
    PyDeviceBuffer(std::shared_ptr<void> data, std::shared_ptr<py::buffer_info> info) : data_{std::move(data)}, info_{std::move(info)} {}

    PyDeviceBuffer(
            const std::shared_ptr<void>& data,
            int64_t item_size,
            std::string format,
            int8_t ndim,
            const Shape& shape,
            const Strides& strides)
        : PyDeviceBuffer{data, std::make_shared<py::buffer_info>(data.get(), item_size, std::move(format), ndim, shape, strides)} {}

    std::shared_ptr<py::buffer_info> info() const { return info_; }

private:
    std::shared_ptr<void> data_;
    std::shared_ptr<py::buffer_info> info_;
};

void InitXchainerDeviceBuffer(pybind11::module& m) {
    py::class_<PyDeviceBuffer> c{m, "_DeviceBuffer", py::buffer_protocol()};
    c.def(py::init([](const py::list& list, const py::tuple& shape_tup, const py::handle& dtype_handle, const py::handle& device) {
              Shape shape = python::internal::ToShape(shape_tup);
              int64_t total_size = shape.GetTotalSize();
              if (static_cast<size_t>(total_size) != list.size()) {
                  throw DimensionError{"Invalid data length"};
              }

              // Copy the Python list to a buffer on the host.
              Dtype dtype = python::internal::GetDtype(dtype_handle);
              int64_t item_size = GetItemSize(dtype);
              int64_t bytes = item_size * total_size;
              std::shared_ptr<void> host_data = std::make_unique<uint8_t[]>(bytes);
              std::string format = VisitDtype(dtype, [&host_data, &list](auto pt) {
                  using T = typename decltype(pt)::type;
                  std::transform(list.begin(), list.end(), static_cast<T*>(host_data.get()), [](auto& item) { return py::cast<T>(item); });
                  return py::format_descriptor<T>::format();  // Return the dtype format, e.g. "f" for xchainer.float32.
              });

              // Copy the data on the host buffer to the target device.
              std::shared_ptr<void> device_data = python::internal::GetDevice(device).FromHostMemory(host_data, bytes);
              return PyDeviceBuffer{device_data, item_size, format, shape.ndim(), shape, Strides{shape, dtype}};
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("data"),
          py::arg("device") = nullptr);
    c.def_buffer([](const PyDeviceBuffer& self) {
        // py::buffer_info cannot be copied.
        std::shared_ptr<py::buffer_info> info = self.info();
        return py::buffer_info{info->ptr, info->itemsize, info->format, info->ndim, info->shape, info->strides};
    });
}

}  // namespace internal
}  // namespace testing
}  // namespace python
}  // namespace xchainer
