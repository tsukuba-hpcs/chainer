#include "xchainer/python/slice.h"

#include <nonstd/optional.hpp>

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Slice MakeSlice(const py::slice& slice) {
    const auto* py_slice = reinterpret_cast<const PySliceObject*>(slice.ptr());  // NOLINT: reinterpret_cast
    auto to_optional = [](PyObject* var) -> nonstd::optional<int64_t> {
        if (var == Py_None) {
            return nonstd::nullopt;
        }
        return py::cast<int64_t>(var);
    };
    return Slice{to_optional(py_slice->start), to_optional(py_slice->stop), to_optional(py_slice->step)};
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
