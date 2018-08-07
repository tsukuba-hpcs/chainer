#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

#include "xchainer/error.h"
#include "xchainer/hash_combine.h"

namespace xchainer {

using BackpropOrdinal = uint64_t;

class Context;

class BackpropId {
public:
    BackpropId(const BackpropId&) = default;
    BackpropId(BackpropId&&) = default;
    BackpropId& operator=(const BackpropId&) = default;
    BackpropId& operator=(BackpropId&&) = default;

    bool operator==(const BackpropId& other) const { return &context_.get() == &other.context_.get() && ordinal_ == other.ordinal_; }

    bool operator!=(const BackpropId& other) const { return !operator==(other); }

    bool operator<(const BackpropId& other) const { return CompareImpl<std::less<BackpropOrdinal>>(other); }

    bool operator<=(const BackpropId& other) const { return CompareImpl<std::less_equal<BackpropOrdinal>>(other); }

    bool operator>(const BackpropId& other) const { return CompareImpl<std::greater<BackpropOrdinal>>(other); }

    bool operator>=(const BackpropId& other) const { return CompareImpl<std::greater_equal<BackpropOrdinal>>(other); }

    Context& context() const { return context_; }

    BackpropOrdinal ordinal() const { return ordinal_; }

private:
    // A BackpropId is always constructed by a Context.
    friend class Context;

    BackpropId(Context& context, BackpropOrdinal ordinal) : context_{context}, ordinal_{ordinal} {}

    template <typename Compare>
    bool CompareImpl(const BackpropId& other) const {
        if (&context_.get() != &other.context_.get()) {
            throw ContextError{"Cannot compare backprop ids with different contexts."};
        }
        return Compare{}(ordinal_, other.ordinal_);
    }

    // Using reference_wrapper to make this class move assignable
    std::reference_wrapper<Context> context_;

    BackpropOrdinal ordinal_;
};

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id);

// Used to represent any graph (id).
class AnyGraph {};

}  // namespace xchainer

namespace std {

template <>
struct hash<xchainer::BackpropId> {
    size_t operator()(const xchainer::BackpropId& backprop_id) const {
        size_t seed = std::hash<xchainer::Context*>()(&backprop_id.context());
        xchainer::internal::HashCombine(seed, std::hash<xchainer::BackpropOrdinal>()(backprop_id.ordinal()));
        return seed;
    }
};

}  // namespace std
