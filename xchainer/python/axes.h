#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <nonstd/optional.hpp>

#include "xchainer/axes.h"

namespace xchainer {
namespace python {
namespace internal {

Axes ToAxes(const pybind11::tuple& tup);

inline OptionalAxes ToAxes(const nonstd::optional<std::vector<int8_t>>& vec) {
    if (vec.has_value()) {
        return Axes{vec->begin(), vec->end()};
    } else {
        return nonstd::nullopt;
    }
}

inline OptionalAxes ToAxes(const nonstd::optional<int8_t>& vec) {
    if (vec.has_value()) {
        return Axes{*vec};
    } else {
        return nonstd::nullopt;
    }
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
