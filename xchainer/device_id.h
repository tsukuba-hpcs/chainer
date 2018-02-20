#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace xchainer {

class Backend;

struct DeviceId {
public:
    DeviceId() = default;  // required to be POD
    DeviceId(Backend* backend, int index = 0) : backend_(backend), index_(index) {}

    Backend* backend() const { return backend_; }
    int index() const { return index_; }

    bool is_null() const;
    std::string ToString() const;

private:
    Backend* backend_;
    int index_;
};

namespace internal {

const DeviceId& GetDefaultDeviceIdNoExcept() noexcept;

constexpr DeviceId kNullDeviceId = {};

}  // namespace internal

inline bool operator==(const DeviceId& lhs, const DeviceId& rhs) { return lhs.backend() == rhs.backend() && lhs.index() == rhs.index(); }

inline bool operator!=(const DeviceId& lhs, const DeviceId& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream&, const DeviceId&);

const DeviceId& GetDefaultDeviceId();

void SetDefaultDeviceId(const DeviceId& device_id);

// Scope object that switches the default device_id by RAII.
class DeviceIdScope {
public:
    DeviceIdScope() : orig_(internal::GetDefaultDeviceIdNoExcept()), exited_(false) {}
    explicit DeviceIdScope(DeviceId device_id) : DeviceIdScope() { SetDefaultDeviceId(device_id); }
    explicit DeviceIdScope(Backend* backend, int index = 0) : DeviceIdScope(DeviceId{backend, index}) {}

    DeviceIdScope(const DeviceIdScope&) = delete;
    DeviceIdScope(DeviceIdScope&&) = delete;
    DeviceIdScope& operator=(const DeviceIdScope&) = delete;
    DeviceIdScope& operator=(DeviceIdScope&&) = delete;

    ~DeviceIdScope() { Exit(); }

    // Explicitly recovers the original device_id. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDeviceId(orig_);
            exited_ = true;
        }
    }

private:
    DeviceId orig_;
    bool exited_;
};

}  // namespace xchainer
