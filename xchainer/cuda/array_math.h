#pragma once

#include "xchainer/array.h"

namespace xchainer {
namespace cuda {

void Identity(const Array& rhs, Array& out);
void Add(const Array& lhs, const Array& rhs, Array& out);
void Mul(const Array& lhs, const Array& rhs, Array& out);

}  // namespace cuda
}  // namespace xchainer
