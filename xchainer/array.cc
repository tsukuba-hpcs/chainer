#include "xchainer/array.h"

#include <cassert>
#include <cstring>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#include "xchainer/array_math.h"
#include "xchainer/array_repr.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/array_math.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/op_node.h"

namespace xchainer {

namespace {

#ifdef XCHAINER_ENABLE_CUDA
std::shared_ptr<void> AllocateCudaManaged(size_t size) {
    std::shared_ptr<void> ptr{};
    {
        void* raw_ptr = nullptr;
        auto raw_ptr_scope = gsl::finally([&]() {
            if (raw_ptr && !ptr) {
                cudaFree(raw_ptr);
            }
        });
        cuda::CheckError(cudaMallocManaged(&raw_ptr, size, cudaMemAttachGlobal));
        ptr.reset(raw_ptr, cudaFree);
    }
    return ptr;
}
#endif  // XCHAINER_ENABLE_CUDA

std::shared_ptr<void> Allocate(const Device& device, size_t size) {
    if (device == MakeDevice("cpu")) {
        return std::make_unique<uint8_t[]>(size);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        return AllocateCudaManaged(size);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void MemoryCopy(const Device& device, void* dst_ptr, const void* src_ptr, size_t size) {
    if (device == MakeDevice("cpu")) {
        std::memcpy(dst_ptr, src_ptr, size);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        cuda::CheckError(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice));
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

}  // namespace

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, int64_t offset)
    : shape_(shape), is_contiguous_(true), dtype_(dtype), data_(nullptr), offset_(offset), node_(std::make_shared<ArrayNode>()) {
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        data_ = std::move(data);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        size_t size = static_cast<size_t>(shape_.total_size() * GetElementSize(dtype));
        data_ = AllocateCudaManaged(size);
        MemoryCopy(device, data_.get(), data.get(), size);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

Array Array::Empty(const Shape& shape, Dtype dtype) {
    size_t size = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> data = Allocate(GetCurrentDevice(), size);
    return {shape, dtype, data};
}

Array Array::EmptyLike(const Array& array) { return Empty(array.shape(), array.dtype()); }

Array Array::DeepCopy() const {
    auto bytes = total_bytes();
    if (GetCurrentDevice() == MakeDevice("cpu")) {
        std::shared_ptr<void> ret_data = std::make_unique<uint8_t[]>(bytes);
        std::memcpy(ret_data.get(), data_.get(), bytes);
        return {shape_, dtype_, ret_data, offset_};
#ifdef XCHAINER_ENABLE_CUDA
    } else if (GetCurrentDevice() == MakeDevice("cuda")) {
        void* ret_ptr = nullptr;
        cuda::CheckError(cudaMallocManaged(&ret_ptr, bytes, cudaMemAttachGlobal));
        cuda::CheckError(cudaMemcpy(ret_ptr, data_.get(), bytes, cudaMemcpyDeviceToDevice));
        std::shared_ptr<void> ret_data = std::shared_ptr<void>(ret_ptr, ::cudaFree);
        return {shape_, dtype_, ret_data, offset_};
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

Array& Array::operator+=(const Array& rhs) {
    Add(rhs, *this);
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    Mul(rhs, *this);
    return *this;
}

Array Array::operator+(const Array& rhs) const {
    Array out = {shape_, dtype_, std::make_unique<uint8_t[]>(total_bytes())};
    Add(rhs, out, enable_backprop);
    return out;
}

Array Array::operator*(const Array& rhs) const {
    Array out = {shape_, dtype_, std::make_unique<uint8_t[]>(total_bytes())};
    Mul(rhs, out, enable_backprop);
    return out;
}

void Array::Add(const Array& rhs, Array& out) const {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    if (enable_backprop) {
        std::shared_ptr<const ArrayNode> lhs_node = node();
        std::shared_ptr<const ArrayNode> rhs_node = rhs.node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        auto lhs_func = [](const Array& gout) { return gout; };
        auto rhs_func = [](const Array& gout) { return gout; };
        auto functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node =
            std::make_shared<OpNode>("add", std::vector<std::shared_ptr<const ArrayNode>>{lhs_node, rhs_node}, functions);
        out_node->set_next_node(op_node);
    }

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Add(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Add(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void Array::Mul(const Array& rhs, Array& out) const {
    // TODO: dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO: broadcasting
    CheckEqual(shape_, rhs.shape());

    if (enable_backprop) {
        // deep copy for in-place operation to keep original input
        const Array& lhs = (this == &out) ? DeepCopy() : *this;
        std::shared_ptr<const ArrayNode> lhs_node = node();
        std::shared_ptr<const ArrayNode> rhs_node = rhs.node();
        std::shared_ptr<ArrayNode> out_node = out.CreateNewNode();
        auto lhs_func = [rhs](const Array& gout) { return gout.Mul(rhs, false); };
        auto rhs_func = [lhs](const Array& gout) { return gout.Mul(lhs, false); };
        auto functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node =
            std::make_shared<OpNode>("mul", std::vector<std::shared_ptr<const ArrayNode>>{lhs_node, rhs_node}, functions);
        out_node->set_next_node(op_node);
    }

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Mul(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Mul(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

std::string Array::ToString() const { return ArrayRepr(*this); }

namespace {

void DebugDumpComputationalGraph(std::ostream& os, const ArrayNode& array_node, int indent) {
    static const char kIndentChar = ' ';

    os << std::string(static_cast<size_t>(indent * 2), kIndentChar) << "ArrayNode<" << &array_node << ">" << std::endl;

    std::shared_ptr<const OpNode> op = array_node.next_node();
    if (op) {
        os << std::string(static_cast<size_t>((indent + 1) * 2), kIndentChar) << "Op<" << op->name() << ">" << std::endl;
        for (const std::shared_ptr<const ArrayNode>& next_node : op->next_nodes()) {
            DebugDumpComputationalGraph(os, *next_node, static_cast<size_t>(indent + 2));
        }
    }
}

}  // namespace

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, int indent) {
    DebugDumpComputationalGraph(os, *array.node(), indent);
}

}  // namespace xchainer
