#include "xchainer/context.h"

#include <atomic>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

static std::atomic<Context*> g_default_context{nullptr};

}  // namespace

Backend& Context::GetBackend(const std::string& backend_name) {
    {
        std::lock_guard<std::recursive_mutex> lock{mutex_};
        auto it = backends_.find(backend_name);
        if (it != backends_.end()) {
            return *it->second;
        }
    }

    // Ctor of each backend may call member functions of Context.
    // Lock is released here to avoid any deadlocks.
    std::unique_ptr<Backend> backend;
    if (backend_name == NativeBackend::kDefaultName) {
        backend = std::make_unique<NativeBackend>();
    }
#ifdef XCHAINER_ENABLE_CUDA
    else if (backend_name == cuda::CudaBackend::kDefaultName) {
        backend = std::make_unique<cuda::CudaBackend>();
    }
#endif  // XCHAINER_ENABLE_CUDA
    else {
        throw BackendError("Backend not found: '" + backend_name + "'");
    }

    {
        // In a multi-threaded case, backends_[backend_name] may already exist at this point.
        // In that case, the backend created above is thrown away.
        std::lock_guard<std::recursive_mutex> lock{mutex_};
        auto pair = backends_.emplace(backend_name, std::move(backend));
        return *pair.first->second;
    }
}

Device& Context::GetDevice(const DeviceId& device_id) {
    Backend& backend = GetBackend(device_id.backend_name());
    return backend.GetDevice(device_id.index());
}

Context& GetDefaultContext() {
    Context* context = g_default_context;
    if (context == nullptr) {
        throw ContextError("Default context is not set.");
    }
    return *context;
}

void SetDefaultContext(Context* context) { g_default_context = context; }

// TODO(hvy): Continue next mob programming sessions implementing the following functions.
// TODO(hvy): Then, add Context to Backend.
// Context& GetCurrentContext();
// void SetCurrentContext(Context* context);

}  // namespace xchainer
