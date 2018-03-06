#include "xchainer/native_device.h"

#include <cstring>  // for std::memcpy

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/scalar.h"

namespace xchainer {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

void NativeDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&dst_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

std::tuple<std::shared_ptr<void>, size_t> NativeDevice::TransferDataFrom(Device& src_device, const std::shared_ptr<void>& src_ptr,
                                                                         size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &static_cast<int8_t*>(src_ptr.get())[offset], bytesize, src_device);
    return std::make_tuple(std::move(dst_ptr), 0);
}

std::tuple<std::shared_ptr<void>, size_t> NativeDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr,
                                                                       size_t offset, size_t bytesize) {
    return dst_device.TransferDataFrom(*this, src_ptr, offset, bytesize);
}

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
        IndexableArray<const T> lhs_indexable_array{lhs};
        IndexableArray<const T> rhs_indexable_array{rhs};
        IndexableArray<T> out_indexable_array{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            out_indexable_array[indexer] = lhs_indexable_array[indexer] + rhs_indexable_array[indexer];
        }
    });
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_indexable_array{lhs};
        IndexableArray<const T> rhs_indexable_array{rhs};
        IndexableArray<T> out_indexable_array{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            out_indexable_array[indexer] = lhs_indexable_array[indexer] * rhs_indexable_array[indexer];
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace xchainer
