#include "xchainer/python/strides.h"

#include <algorithm>
#include <cstdint>

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Strides ToStrides(const py::tuple& tup) {
    Strides strides;
    std::transform(tup.begin(), tup.end(), std::back_inserter(strides), [](auto& item) { return py::cast<int64_t>(item); });
    return strides;
}

py::tuple ToTuple(const Strides& strides) {
    py::tuple ret{strides.size()};
    for (size_t i = 0; i < strides.size(); ++i) {
        ret[i] = strides[i];
    }
    return ret;
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
