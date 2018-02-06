#include "xchainer/device.h"

#include "xchainer/error.h"

namespace xchainer {
namespace {

thread_local Device thread_local_device = internal::kNullDevice;
static_assert(std::is_pod<decltype(thread_local_device)>::value, "thread_local_device must be POD");

}  // namespace

namespace internal {

const Device& GetCurrentDeviceNoExcept() noexcept { return thread_local_device; }

}  // namespace internal

Device::Device(const std::string& name, Backend* backend) : name_(), backend_(backend) {
    if (name.size() >= device_detail::kMaxDeviceNameLength) {
        throw DeviceError("device name is too long; should be shorter than 8 characters");
    }
    std::copy(name.begin(), name.end(), static_cast<char*>(name_));
}

bool Device::is_null() const { return *this == internal::kNullDevice; }

const Device& GetCurrentDevice() {
    if (thread_local_device.is_null()) {
        throw XchainerError("No device is available. Please set via SetCurrentDevice()");
    } else {
        return thread_local_device;
    }
}

void SetCurrentDevice(const Device& device) { thread_local_device = device; }

}  // namespace xchainer
