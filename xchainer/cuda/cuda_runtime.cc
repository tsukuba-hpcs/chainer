#include "xchainer/cuda/cuda_runtime.h"

#include <cassert>
#include <sstream>
#include <string>

namespace xchainer {
namespace cuda {
namespace {

std::string BuildErrorMessage(cudaError_t error) {
    std::ostringstream os;
    os << cudaGetErrorName(error) << ":" << cudaGetErrorString(error);
    return os.str();
}

}  // namespace

RuntimeError::RuntimeError(cudaError_t error) : XchainerError(BuildErrorMessage(error)), error_(error) {}

void CheckCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        Throw(error);
    }
}

void Throw(cudaError_t error) { throw RuntimeError(error); }

bool IsPointerCudaMemory(const void* ptr) {
    cudaPointerAttributes attr = {};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            if (attr.isManaged == 0) {
                throw XchainerError{"Non-managed GPU memory is not supported"};
            }
            return true;
        case cudaErrorInvalidValue:
            return false;
        default:
            CheckCudaError(status);
            break;
    }
    assert(false);  // should never be reached
}

}  // namespace cuda
}  // namespace xchainer
