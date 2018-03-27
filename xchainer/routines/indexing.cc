#include "xchainer/routines/indexing.h"

#include <string>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"

namespace xchainer {
namespace routines {
namespace {

// Returns an array where elements at indices are added by the addends `b`.
//
// The original values of this array are not altered.
Array AddAt(const Array& a, const std::vector<ArrayIndex>& indices, const Array& b) {
    // TODO(sonots): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array out = a.AsConstant(CopyKind::kCopy);
    Array out_view = out.At(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), b.shape());

    a.device().Add(b, out_view, out_view);

    auto this_backward_function = [](const Array& gout, const std::vector<GraphId>&) { return gout; };
    auto addend_backward_function = [indices](const Array& gout, const std::vector<GraphId>&) { return gout.At(indices); };
    xchainer::internal::SetUpOpNodes("add_at", {a, b}, out, {this_backward_function, addend_backward_function});

    return out;
}

}  // namespace

Array At(const Array& a, const std::vector<ArrayIndex>& indices) {
    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_strides;
    int64_t out_offset = a.offset();
    int64_t i_in = 0;
    for (const ArrayIndex& index : indices) {
        switch (index.tag()) {
            case ArrayIndexTag::kSingleElement: {
                int64_t dim = a.shape()[i_in];
                if (index.index() < -dim || dim <= index.index()) {
                    throw DimensionError(
                            "Index " + std::to_string(index.index()) + " is out of bounds for axis " + std::to_string(i_in) +
                            " with size " + std::to_string(dim));
                }
                out_offset += a.strides()[i_in] * ((index.index() + dim) % dim);
                ++i_in;
                break;
            }
            case ArrayIndexTag::kSlice: {
                const Slice& slice = index.slice();
                int64_t slice_length = slice.GetLength(a.shape()[i_in]);
                out_offset += a.strides()[i_in] * slice.GetStart(a.shape()[i_in]);
                out_shape.push_back(slice_length);
                out_strides.push_back(a.strides()[i_in] * slice.step());
                ++i_in;
                break;
            }
            case ArrayIndexTag::kNewAxis:
                out_shape.push_back(1);
                out_strides.push_back(0);
                break;
            default:
                assert(false);
        }
    }
    for (int64_t i = i_in; i < a.ndim(); ++i) {
        out_shape.push_back(a.shape()[i]);
        out_strides.push_back(a.strides()[i]);
    }

    Array out{{out_shape.begin(), out_shape.end()}, {out_strides.begin(), out_strides.end()}, a.dtype(), a.device(), a.data(), out_offset};

    auto backward_function = [ indices, other = a ](const Array& gout, const std::vector<GraphId>&) {
        Array gin = Array::ZerosLike(other, other.device());
        return AddAt(gin, indices, gout);
    };
    xchainer::internal::SetUpOpNodes("get_item", {a}, out, {backward_function});

    return out;
}
}  // namespace routines
}  // namespace xchainer
