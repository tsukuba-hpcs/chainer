#include "xchainer/native_device.h"

#include <cstring>  // for std::memcpy

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) { std::memcpy(dst_ptr, src_ptr, bytesize); }

std::shared_ptr<void> NativeDevice::FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

void NativeDevice::Fill(Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        T c_value{value};

        int64_t size = out.GetTotalSize();
        auto* ptr = static_cast<T*>(out.data().get());
        for (int64_t i = 0; i < size; ++i) {
            ptr[i] = c_value;
        }
    });
}

void NativeDevice::Add(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.GetTotalSize();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] + rdata[i];
        }
    });
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.GetTotalSize();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * rdata[i];
        }
    });
}

void NativeDevice::Synchronize() {}

std::tuple<std::shared_ptr<void>, size_t> NativeDevice::TransferDataFrom(Device& src_device, const std::shared_ptr<void>& src_ptr,
                                                                         size_t offset, size_t bytesize) {
    (void)src_device;  // unused
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopy(dst_ptr.get(), &static_cast<int8_t*>(src_ptr.get())[offset], bytesize);
    return std::make_tuple(std::move(dst_ptr), size_t{0});
}

std::tuple<std::shared_ptr<void>, size_t> NativeDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr,
                                                                       size_t offset, size_t bytesize) {
    return dst_device.TransferDataFrom(*this, src_ptr, offset, bytesize);
}

}  // namespace xchainer
