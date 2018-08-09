#pragma once

#include "xchainer/macro.h"

namespace xchainer {

template <typename T>
class ArithmeticOps {
public:
    XCHAINER_HOST_DEVICE static T Add(T lhs, T rhs) { return lhs + rhs; }
    XCHAINER_HOST_DEVICE static T Multiply(T lhs, T rhs) { return lhs * rhs; }
    XCHAINER_HOST_DEVICE static T Divide(T lhs, T rhs) { return lhs / rhs; }
};

template <>
class ArithmeticOps<bool> {
public:
    XCHAINER_HOST_DEVICE static bool Add(bool lhs, bool rhs) { return lhs || rhs; }
    XCHAINER_HOST_DEVICE static bool Multiply(bool lhs, bool rhs) { return lhs && rhs; }
    // TODO(beam2d): It's a tentative implementation. Make distinction between TrueDivide and FloorDivide for better NumPy compatibility.
    // The current implementation is of boolean FloorDivide except for warnings.
    XCHAINER_HOST_DEVICE static bool Divide(bool lhs, bool rhs) { return lhs && rhs; }
};

}  // namespace xchainer
