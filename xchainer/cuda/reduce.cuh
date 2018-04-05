#pragma once

#include <cstdint>
#include <type_traits>

#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/reduction_kernel_arg.h"

namespace xchainer {
namespace cuda {
namespace reduce_detail {

static constexpr int kMaxReductionBlockSize = 512;

int64_t RoundUpToPowerOf2(int64_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

template <typename In, typename Out, typename ReductionImpl>
__global__ void ReductionKernel(ReductionKernelArg<In, Out> arg, int reduce_block_size, ReductionImpl impl) {
    using T = decltype(impl.Identity());

    extern __shared__ __align__(8) uint8_t work_bytes[];
    T* work = reinterpret_cast<T*>(work_bytes);
    int tid = threadIdx.x;
    int reduce_blocks_per_grid = (blockDim.x + reduce_block_size - 1) / reduce_block_size * gridDim.x;

    for (int64_t i_out = blockIdx.x; i_out < arg.out_indexer.total_size(); i_out += gridDim.x * reduce_blocks_per_grid) {
        arg.out_indexer.Set(i_out);

        T accum = impl.Identity();

        // Set output indices in the corresponding indices (out_axis) in src_index.
        for (int8_t i_out_dim = 0; i_out_dim < arg.out_indexer.ndim(); ++i_out_dim) {
            arg.in_indexer.index()[i_out_dim] = arg.out_indexer.index()[i_out_dim];
        }

        // Linearly compute the partial sum into at most kMaxReductionBlockSize values.
        for (int64_t i_reduce = tid; i_reduce < arg.reduce_indexer.total_size(); i_reduce += reduce_block_size) {
            arg.reduce_indexer.Set(i_reduce);

            // Set reduction indices in the corresponding indices (axis) in src_index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < arg.reduce_indexer.ndim(); ++i_reduce_dim) {
                arg.in_indexer.index()[arg.out_indexer.ndim() + i_reduce_dim] = arg.reduce_indexer.index()[i_reduce_dim];
            }

            impl.Reduce(impl.MapIn(arg.in[arg.in_indexer], i_reduce), accum);
        }

        if (reduce_block_size >= 2) {
            // Synchronize partial sums
            work[tid] = accum;
            __syncthreads();

            // Reduction
            if (reduce_block_size > 2) {
                if (reduce_block_size > 4) {
                    if (reduce_block_size > 8) {
                        if (reduce_block_size > 16) {
                            if (reduce_block_size > 32) {
                                if (reduce_block_size > 64) {
                                    if (reduce_block_size > 128) {
                                        if (reduce_block_size > 256) {
                                            static_assert(kMaxReductionBlockSize == 512, "");

                                            if (tid < 256) {
                                                impl.Reduce(work[tid + 256], work[tid]);
                                            }
                                            __syncthreads();
                                        }
                                        if (tid < 128) {
                                            impl.Reduce(work[tid + 128], work[tid]);
                                        }
                                        __syncthreads();
                                    }
                                    if (tid < 64) {
                                        impl.Reduce(work[tid + 64], work[tid]);
                                    }
                                    __syncthreads();
                                }
                                if (tid < 32) {
                                    impl.Reduce(work[tid + 32], work[tid]);
                                }
                                __syncthreads();
                            }
                            if (tid < 16) {
                                impl.Reduce(work[tid + 16], work[tid]);
                            }
                            __syncthreads();
                        }
                        if (tid < 8) {
                            impl.Reduce(work[tid + 8], work[tid]);
                        }
                        __syncthreads();
                    }
                    if (tid < 4) {
                        impl.Reduce(work[tid + 4], work[tid]);
                    }
                    __syncthreads();
                }
                if (tid < 2) {
                    impl.Reduce(work[tid + 2], work[tid]);
                }
                __syncthreads();
            }
            accum = work[1];
            impl.Reduce(work[0], accum);
        }
        // Store the output value
        if (tid == 0) {
            arg.out[arg.out_indexer] = impl.MapOut(accum);
        }
    }
}

}  // namespace reduce_detail

// Computes the reduction of the input and stores into the output array.
//
// `ReductionImpl` is required to provide the following device member function.
// T can be arbitrary but should be common between these functions.
//
// - T Identity();
//       Returns the initial value of reduction.
// - T MapIn(In in, int64_t index);
//       Applies pre-reduction mapping of the input and its index.
// - void Reduce(T next, T& accum);
//       Accumulates the iterated value to accum.
// - Out MapOut(T accum);
//       Applies post-reduction mapping of the output.
//
// Example:
//     Simple summation over a float array can be implemented as the following reduction impl.
//
//         struct SumImpl {
//             __device__ float Identity() { return 0; }
//             __device__ float MapIn(float in, int64_t /*index*/) { return in; }
//             __device__ void Reduce(float next, float& accum) { accum += next; }
//             __device__ float MapOut(float accum) { return accum; }
//         };
//
//     Then, it can be passed to Reduce like: Reduce(MakeReductionKernelArg(input, axis, output), SumImpl{});
template <typename In, typename Out, typename ReductionImpl>
void Reduce(ReductionKernelArg<In, Out> arg, ReductionImpl&& impl) {
    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, ReductionImpl>).block_size;

    int reduce_block_size = static_cast<int>(std::min(
            static_cast<int64_t>(reduce_detail::kMaxReductionBlockSize),
            reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, arg.reduce_indexer.total_size()))));
    int block_size = std::min(kMaxBlockSize, reduce_block_size);
    int64_t total_reduce_blocks = arg.out_indexer.total_size();
    int64_t grid_size = total_reduce_blocks;
    size_t shared_mem_size = sizeof(decltype(impl.Identity())) * reduce_block_size;

    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(arg, reduce_block_size, impl);
}

}  // namespace cuda
}  // namespace xchainer
