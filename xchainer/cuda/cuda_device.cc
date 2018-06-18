#include "xchainer/cuda/cuda_device.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "xchainer/cuda/cublas.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

CudaDevice::~CudaDevice() {
    if (cublas_handle_ != nullptr) {
        cudaSetDevice(index());
        cublasDestroy(cublas_handle_);
    }
    if (cudnn_handle_ != nullptr) {
        cudaSetDevice(index());
        cudnnDestroy(cudnn_handle_);
    }
}

cublasHandle_t CudaDevice::cublas_handle() {
    if (cublas_handle_ == nullptr) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCublasError(cublasCreate(&cublas_handle_));
    }
    return cublas_handle_;
}

cudnnHandle_t CudaDevice::cudnn_handle() {
    if (cudnn_handle_ == nullptr) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCudnnError(cudnnCreate(&cudnn_handle_));
    }
    return cudnn_handle_;
}

internal::ConvContext& CudaDevice::conv_context() {
    if (conv_context_) {
        return *conv_context_;
    }
    conv_context_.emplace(cudnn_handle());
    return *conv_context_;
}

void CudaDevice::Synchronize() {
    CheckCudaError(cudaSetDevice(index()));
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
