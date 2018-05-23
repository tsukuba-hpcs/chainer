#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/native/native_backend.h"

namespace xchainer {
namespace cuda {

constexpr const char* CudaBackend::kDefaultName;
constexpr const size_t CudaBackend::kDefaultMaxWorkspaceSize;
constexpr const char* CudaBackend::kMaxWorkspaceSizeEnvName;

std::string CudaBackend::GetName() const { return kDefaultName; }

int CudaBackend::GetDeviceCount() const {
    int count = 0;
    CheckCudaError(cudaGetDeviceCount(&count));
    return count;
}

std::unique_ptr<Device> CudaBackend::CreateDevice(int index) {
    int device_count = GetDeviceCount();
    if (index >= device_count) {
        throw std::out_of_range{"The index number (= " + std::to_string(index) +
                                ") is not less than the device count (= " + std::to_string(device_count) + ')'};
    }
    return std::make_unique<CudaDevice>(*this, index);
}

bool CudaBackend::SupportsTransfer(Device& src_device, Device& dst_device) {
    Backend& src_backend = src_device.backend();
    Backend& dst_backend = dst_device.backend();
    // TODO(niboshi): Make clearner interface to check whether a backend is native, like dst_backend.is_native()
    if (&src_backend == this) {
        return &dst_backend == this || nullptr != dynamic_cast<native::NativeBackend*>(&dst_backend);
    }
    if (&dst_backend == this) {
        return &src_backend == this || nullptr != dynamic_cast<native::NativeBackend*>(&src_backend);
    }
    return false;
}

void CudaBackend::SetMaxWorkspaceSize(size_t max_workspace_size) { max_workspace_size_ = max_workspace_size; }

size_t CudaBackend::GetMaxWorkspaceSize() {
    if (max_workspace_size_) {
        return *max_workspace_size_;
    }
    const char* env = std::getenv(kMaxWorkspaceSizeEnvName);
    if (env == nullptr) {
        max_workspace_size_ = kDefaultMaxWorkspaceSize;
    } else {
        max_workspace_size_ = std::stoul(env);
    }
    return *max_workspace_size_;
}

}  // namespace cuda
}  // namespace xchainer
