#include "xchainer/python/array_index.h"

#include "xchainer/array_index.h"

#include "xchainer/python/slice.h"

namespace xchainer {
namespace python {
namespace internal {
namespace {

namespace py = pybind11;

ArrayIndex MakeArrayIndex(py::handle handle) {
    if (handle.is_none()) {
        return ArrayIndex{NewAxis{}};
    }
    if (py::int_::check_(handle)) {
        return ArrayIndex{py::cast<int64_t>(handle)};
    }
    if (py::slice::check_(handle)) {
        return ArrayIndex{internal::MakeSlice(py::cast<py::slice>(handle))};
    }
    throw py::index_error("only integers, slices (`:`), and xchainer.newaxis (`None`) are valid indices");
}

std::vector<ArrayIndex> MakeArrayIndicesFromTuple(py::tuple tup) {
    std::vector<ArrayIndex> indicies;
    for (auto& handle : tup) {
        indicies.emplace_back(MakeArrayIndex(handle));
    }
    return indicies;
}

}  // namespace

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle) {
    if (py::tuple::check_(handle)) {
        return MakeArrayIndicesFromTuple(py::cast<py::tuple>(handle));
    }
    return {MakeArrayIndex(handle)};
}

void InitXchainerArrayIndex(py::module& m) { m.attr("newaxis") = py::none(); }

}  // namespace internal
}  // namespace python
}  // namespace xchainer
