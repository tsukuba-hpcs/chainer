#pragma once

#include <cstring>
#include <string>

namespace xchainer {

class Backend;

constexpr size_t kMaxDeviceNameLength = 8;

struct Device {
    char name[kMaxDeviceNameLength];
    Backend* backend;
};

constexpr Device kDefaultDevice = {};

namespace internal {

Device GetCurrentDeviceNoExcept() noexcept;

}  // namespace internal

inline bool operator==(const Device& lhs, const Device& rhs) {
    return (strncmp(lhs.name, rhs.name, kMaxDeviceNameLength) == 0) && (lhs.backend == rhs.backend);
}

inline bool operator!=(const Device& lhs, const Device& rhs) { return !(lhs == rhs); }

Device MakeDevice(const std::string& name, Backend* backend);

Device GetCurrentDevice();

void SetCurrentDevice(const Device& device);

void SetCurrentDevice(const std::string& name, Backend* backend);

// Scope object that switches the current device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(GetCurrentDevice()) {}
    explicit DeviceScope(Device device) : DeviceScope() { SetCurrentDevice(device); }
    explicit DeviceScope(const std::string& device, Backend* backend) : DeviceScope(MakeDevice(device, backend)) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (orig_ != kDefaultDevice) {
            SetCurrentDevice(orig_);
        }
        orig_ = kDefaultDevice;
    }

private:
    Device orig_;
};

}  // namespace xchainer
