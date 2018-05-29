#include "xchainer/routines/manipulation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

Scalar AsScalar(const Array& a) {
    if (a.GetTotalSize() != 1) {
        throw DimensionError{"Cannot convert an array of size ", a.GetTotalSize(), " to a scalar, size must be 1."};
    }

    // Copy to the native device
    Array native_copy = a.ToNative();

    // Retrieve the value
    return VisitDtype(a.dtype(), [&native_copy](auto pt) -> Scalar {
        using T = typename decltype(pt)::type;
        const uint8_t* ptr = static_cast<const uint8_t*>(native_copy.data().get()) + native_copy.offset();
        auto typed_ptr = reinterpret_cast<const T*>(ptr);  // NOLINT: reinterpret_cast
        return Scalar{*typed_ptr};
    });
}

Array RollAxis(const Array& a, int8_t axis, int8_t start) {
    // TODO(hvy): Optimize the implementation.
    axis = internal::NormalizeAxis(axis, a.ndim());

    // start can be a.ndim() so we cannot use NormalizeAxis here.
    if (start < -a.ndim() || start > a.ndim()) {
        throw DimensionError{"start arg out of bounds. start: ", start, ", ndim: ", a.ndim()};
    }
    if (start < 0) {
        start += a.ndim();
    }

    Axes axes;
    for (int8_t i = 0; i < a.ndim(); ++i) {
        if (i == start) {
            axes.emplace_back(axis);
        }
        if (i != axis) {
            axes.emplace_back(i);
        }
    }
    if (start == a.ndim()) {
        axes.emplace_back(axis);
    }
    return Transpose(a, axes);
}

Array Transpose(const Array& a, const OptionalAxes& axes) {
    Axes real_axes;
    if (axes.has_value()) {
        if (axes->ndim() != a.ndim()) {
            throw DimensionError{"Axes do not match, input array dimensions: ", a.ndim(), " but axes: ", axes->ndim()};
        }
        real_axes = *axes;
    } else {
        for (int8_t i = 0; i < a.ndim(); ++i) {
            real_axes.emplace_back(a.ndim() - i - 1);
        }
    }
    assert(real_axes.ndim() == a.ndim());

    Shape out_shape;
    Strides out_strides;
    for (int8_t axis : real_axes) {
        out_shape.emplace_back(a.shape()[axis]);
        out_strides.emplace_back(a.strides()[axis]);
    }

    Array out = internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), a.offset());

    auto backward_function = [real_axes](const Array& gout, const std::vector<GraphId>&) {
        Axes backward_axes;
        backward_axes.resize(real_axes.ndim());
        for (int8_t i = 0; i < real_axes.ndim(); ++i) {
            backward_axes[real_axes[i]] = i;
        }
        return Transpose(gout, backward_axes);
    };
    internal::SetUpOpNodes("transpose", {a}, out, {backward_function});

    return out;
}

Array Reshape(const Array& a, const Shape& newshape) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    // If the shape is unchanged, just return a view.
    if (in_shape == newshape) {
        return a.MakeView();
    }

    // Check for invalid shape.
    int64_t total_size = in_shape.GetTotalSize();
    if (total_size != newshape.GetTotalSize()) {
        throw DimensionError{"Cannot reshape array of size ", total_size, " into shape ", newshape};
    }

    int64_t item_size = GetItemSize(a.dtype());
    Strides strides{};
    if (total_size == 0) {
        // Calculate the strides for 0-sized array.
        strides.resize(newshape.ndim());
        strides.back() = item_size;
        for (int8_t i = newshape.ndim() - 1; i >= 1; --i) {
            strides[i - 1] = strides[i] * std::max(int64_t{1}, newshape[i]);
        }
    } else {
        // Calculate the strides for non-0-sized array.

        // reduced_shape and reduced_strides are the shortest shape and strides which can be convertible from input shape and strides
        // without copy.
        Shape reduced_shape{};
        Strides reduced_strides{};
        if (in_shape.ndim() == 0) {
            // Input shape is (). Treat as if it were (1).
            reduced_shape.push_back(int64_t{1});
            reduced_strides.push_back(item_size);
        } else {
            // Add the first pair
            reduced_shape.emplace_back(in_shape[0]);
            reduced_strides.emplace_back(in_strides[0]);
            // Reduce the remaining
            for (int8_t i = 1; i < in_shape.ndim(); ++i) {
                int64_t dim = in_shape[i];
                int64_t st = in_strides[i];
                assert(dim > 0);
                if (dim == 1 && st == 0) {
                    // If the axis has unit-length with no stride, skip this dimension.
                } else if (dim * st == reduced_strides.back()) {
                    // If the pair is compatible with the previous stride, reduce the pair to it.
                    reduced_shape.back() *= dim;
                    reduced_strides.back() = st;
                } else {
                    // Otherwise, add a new shape and stride.
                    reduced_shape.push_back(dim);
                    reduced_strides.push_back(st);
                }
            }
        }
        assert(reduced_shape.size() == reduced_strides.size());
        assert(!reduced_shape.empty());

        // Construct the strides for no-copy reshape.
        // If it's not possible, can_reshape_without_copy will be false.
        bool can_reshape_without_copy = true;
        if (newshape.ndim() > 0) {
            int64_t last_stride = reduced_shape[0] * reduced_strides[0];
            size_t i_dim = 0;
            for (int64_t dim : newshape) {
                if (dim <= 1) {
                    strides.push_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides.push_back(last_stride);
                if (reduced_shape[i_dim] == 1) {
                    ++i_dim;
                }
            }
        }

        if (!can_reshape_without_copy) {
            // Copy is required.
            return a.Copy().Reshape(newshape);
        }
        assert(strides.size() == newshape.size());
    }

    Array out = internal::MakeArray(newshape, strides, a.dtype(), a.device(), a.data(), a.offset());
    internal::SetUpOpNodes(
            "reshape", {a}, out, {[in_shape](const Array& gout, const std::vector<GraphId>&) { return gout.Reshape(in_shape); }}, {});

    assert(out.shape() == newshape);
    assert(out.strides().size() == newshape.size());
    return out;
}

Array Squeeze(const Array& a, const OptionalAxes& axis) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    Shape out_shape{};
    Strides out_strides{};

    if (axis.has_value()) {
        const Axes sorted_axis = internal::GetSortedAxes(*axis, in_shape.ndim());

        int64_t i_axis = 0;
        for (int64_t i = 0; i < in_shape.ndim(); ++i) {
            if (i_axis < static_cast<int64_t>(sorted_axis.size()) && sorted_axis[i_axis] == i) {
                ++i_axis;
                if (in_shape[i] != 1) {
                    std::ostringstream os;
                    os << "Cannot squeeze out non-unit-length axes, where shape was " << in_shape.ToString();
                    os << " and axes were (";
                    for (auto it = axis->begin(); it != axis->end(); ++it) {
                        if (it != axis->begin()) {
                            os << ", ";
                        }
                        os << *it;
                    }
                    os << (axis->size() == 1 ? ",)." : ").");
                    throw DimensionError{os.str()};
                }
            } else {
                out_shape.emplace_back(in_shape[i]);
                out_strides.emplace_back(in_strides[i]);
            }
        }
    } else {  // All axes are candidates for removal if none are given.
        for (int64_t i = 0; i < in_shape.ndim(); ++i) {
            if (in_shape[i] != 1) {
                out_shape.emplace_back(in_shape[i]);
                out_strides.emplace_back(in_strides[i]);
            }
        }
    }

    Array out = in_shape.size() == out_shape.size()
                        ? a
                        : internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), a.offset());
    internal::SetUpOpNodes(
            "squeeze", {a}, out, {[in_shape](const Array& gout, const std::vector<GraphId>&) { return gout.Reshape(in_shape); }});

    return out;
}

Array BroadcastTo(const Array& array, const Shape& shape) {
    const Shape& in_shape = array.shape();
    const Strides& in_strides = array.strides();

    if (in_shape.size() > shape.size()) {
        throw DimensionError{"Cannot broadcast to smaller dimensions"};
    }

    Array out = internal::MakeArray(
            shape, internal::BroadcastStrides(in_strides, in_shape, shape), array.dtype(), array.device(), array.data(), array.offset());

    auto backward_function = [in_shape](const Array& gout, const std::vector<GraphId>&) {
        if (gout.shape() == in_shape) {
            return gout;
        }

        int8_t lead = gout.ndim() - in_shape.ndim();
        Axes lead_axis{};
        lead_axis.resize(lead);
        std::iota(lead_axis.begin(), lead_axis.end(), int8_t{0});

        Axes axis{lead_axis};
        for (int8_t i = 0; i < in_shape.ndim(); ++i) {
            if (in_shape[i] == 1) {
                axis.emplace_back(i + lead);
            }
        }
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());  // Sum does not accept axis with duplicate elements

        Array gin = gout.Sum(axis, true);
        if (lead > 0) {
            return gin.Squeeze(lead_axis);
        }
        return gin;
    };
    internal::SetUpOpNodes("broadcast_to", {array}, out, {backward_function});

    return out;
}

}  // namespace xchainer
