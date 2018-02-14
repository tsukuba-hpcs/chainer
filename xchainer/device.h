#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace xchainer {

namespace device_detail {

constexpr size_t kMaxDeviceNameLength = 8;

}  // device_detail

class Backend;

struct Device {
public:
    Device() = default;  // required to be POD
    Device(const std::string& name, Backend* backend);

    std::string name() const { return name_; }
    Backend* backend() const { return backend_; }

    bool is_null() const;
    std::string ToString() const;

private:
    char name_[device_detail::kMaxDeviceNameLength];
    Backend* backend_;
};

namespace internal {

const Device& GetDefaultDeviceNoExcept() noexcept;

constexpr Device kNullDevice = {};

}  // namespace internal

inline bool operator==(const Device& lhs, const Device& rhs) { return (lhs.name() == rhs.name()) && (lhs.backend() == rhs.backend()); }

inline bool operator!=(const Device& lhs, const Device& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream&, const Device&);

const Device& GetDefaultDevice();

void SetDefaultDevice(const Device& device);

// Scope object that switches the default device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(internal::GetDefaultDeviceNoExcept()), exited_(false) {}
    explicit DeviceScope(Device device) : DeviceScope() { SetDefaultDevice(device); }
    explicit DeviceScope(const std::string& device, Backend* backend) : DeviceScope(Device{device, backend}) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope(DeviceScope&&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDevice(orig_);
            exited_ = true;
        }
    }

private:
    Device orig_;
    bool exited_;
};

void DebugDumpDevice(std::ostream& os, const Device& device);

}  // namespace xchainer
