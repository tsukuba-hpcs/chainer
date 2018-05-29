#pragma once

#include <cstddef>

namespace xchainer {
namespace cuda {
namespace internal {

// Borrowed from boost::hash_combine
//
// See LICENSE.txt of xChainer.
void hash_combine(std::size_t& seed, std::size_t hash_value) { seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
