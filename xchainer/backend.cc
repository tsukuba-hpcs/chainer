#include "xchainer/backend.h"

#include <string>
#include "xchainer/device.h"

namespace xchainer {

Backend::Backend(Context& context) : context_(context) {}

Device& Backend::GetDevice(int index) {
    if (index < 0) {
        throw std::out_of_range("The index number (= " + std::to_string(index) + ") is negative");
    }
    std::unique_ptr<Device> device = CreateDevice(index);
    std::lock_guard<std::mutex> lock{devices_mutex_};
    if (devices_.size() <= static_cast<size_t>(index)) {
        devices_.resize(index + 1);
    }
    if (devices_[index] == nullptr) {
        devices_[index] = std::move(device);
    }
    return *devices_[index];
}

}  // namespace xchainer
