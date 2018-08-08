#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {

// Argument to reduction kernels.
//
// It contains five data:
//
// - Input indexable array
// - Output indexable array
// - Input indexer (using the input shape)
// - Output indexer (using the output shape)
// - Reduction indexer (using the input shape only at reduction axes)
//
// Input and output arrays are transposed so that the reduction axes come last. Axes of length 1 are also removed.
//
// Any instance of this struct can be passed directly to a kernel function (including CUDA __global__ function).
template <typename In, typename Out, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim, int8_t ReduceNdim = kDynamicNdim>
struct ReductionKernelArg {
    IndexableArray<const In, InNdim> in;
    IndexableArray<Out, OutNdim> out;
    Indexer<InNdim> in_indexer;
    Indexer<OutNdim> out_indexer;
    Indexer<ReduceNdim> reduce_indexer;
};

// A structure to represent argument of Reduce function.
//
// This structure is used to make a reduction kernel argument having indexers with dynamic ndim or statically optmized ndim.
//
// Strides and shapes are permuted so that the reduction axes come last. Axes of length 1 are also removed.
// Contiguous dimensions of strides and shapes are squashed.
class ReductionArg {
public:
    ReductionArg(const Array& in, const Axes& axis, const Array& out);

    const Array& in() const { return in_; }
    const Array& out() const { return out_; }
    const Strides& in_strides() const { return in_strides_; }
    const Strides& out_strides() const { return out_strides_; }
    const Shape& in_shape() const { return in_shape_; }
    const Shape& out_shape() const { return out_shape_; }
    const Shape& reduce_shape() const { return reduce_shape_; }

private:
    void Permute(const Axes& axis);
    void Squash();

    const Array& in_;
    const Array& out_;
    Strides in_strides_;
    Strides out_strides_;
    Shape in_shape_;
    Shape out_shape_;
    Shape reduce_shape_;
};

// Creates ReductionKernelArg from ReductionArg
template <typename In, typename Out, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim, int8_t ReduceNdim = kDynamicNdim>
ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim> MakeReductionKernelArg(const ReductionArg& arg) {
    return ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim>{IndexableArray<const In, InNdim>{arg.in(), arg.in_strides()},
                                                                    IndexableArray<Out, OutNdim>{arg.out(), arg.out_strides()},
                                                                    Indexer<InNdim>{arg.in_shape()},
                                                                    Indexer<OutNdim>{arg.out_shape()},
                                                                    Indexer<ReduceNdim>{arg.reduce_shape()}};
}

}  // namespace xchainer
