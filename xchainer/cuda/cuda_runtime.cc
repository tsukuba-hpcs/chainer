#include "xchainer/cuda/cuda_runtime.h"

#include <sstream>
#include <string>

#include "xchainer/macro.h"

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
    XCHAINER_NEVER_REACH();
}

}  // namespace cuda
}  // namespace xchainer
