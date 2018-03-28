#include "xchainer/routines/util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "xchainer/error.h"

namespace xchainer {
namespace internal {

std::vector<int8_t> GetSortedAxes(const std::vector<int8_t>& axis, int8_t ndim) {
    std::vector<int8_t> sorted_axis = axis;

    for (auto& a : sorted_axis) {
        if (a < -ndim || ndim <= a) {
            throw DimensionError("Axis " + std::to_string(a) + " is out of bounds for array of dimension " + std::to_string(ndim));
        }
        if (a < 0) {
            a += ndim;
        }
    }
    std::sort(sorted_axis.begin(), sorted_axis.end());
    if (std::unique(sorted_axis.begin(), sorted_axis.end()) != sorted_axis.end()) {
        throw XchainerError("Duplicate axis values.");
    }

    // sorted_axis is sorted, unique, and within bounds [0, ndim).
    assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));
    assert(std::set<int8_t>(sorted_axis.begin(), sorted_axis.end()).size() == sorted_axis.size());
    assert(std::all_of(sorted_axis.begin(), sorted_axis.end(), [ndim](int8_t x) -> bool { return 0 <= x && x < ndim; }));
    return sorted_axis;
}

}  // namespace internal
}  // namespace xchainer
