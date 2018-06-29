#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/array_repr.h"
#include "xchainer/axes.h"
#include "xchainer/backend.h"
#include "xchainer/backward.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/macro.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
#include "xchainer/routines/linalg.h"
#include "xchainer/routines/logic.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"
#include "xchainer/routines/sorting.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace {

// The global array body leak tracker.
// See xchainer::internal::SetArrayBodyLeakTracker().
xchainer::internal::ArrayBodyLeakTracker* g_array_body_leak_tracker = nullptr;

}  // namespace

namespace internal {

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset) {
    return Array{shape, strides, dtype, device, std::move(data), offset};
}

bool HasArrayNode(const Array& array, const GraphId& graph_id) {
    return std::find_if(array.nodes().begin(), array.nodes().end(), [&graph_id](const auto& array_node) {
               return graph_id == array_node->graph_id();
           }) != array.nodes().end();
}

const std::shared_ptr<ArrayNode>& CreateArrayNode(const Array& array, const GraphId& graph_id) {
    return array.body()->AddNode(std::make_shared<ArrayNode>(array.body(), array.shape(), array.dtype(), array.device(), graph_id));
}

std::shared_ptr<const ArrayNode> GetArrayNode(const Array& array, const GraphId& graph_id) { return GetMutableArrayNode(array, graph_id); }

const std::shared_ptr<ArrayNode>& GetMutableArrayNode(const Array& array, const GraphId& graph_id) {
    auto it = std::find_if(
            array.nodes().begin(), array.nodes().end(), [&graph_id](const auto& node) { return graph_id == node->graph_id(); });
    if (it == array.nodes().end()) {
        throw XchainerError{"Array does not belong to the graph: '", graph_id, "'."};
    }
    return *it;
}

void ArrayBodyLeakTracker::operator()(const std::shared_ptr<internal::ArrayBody>& array_body) {
    // Keep weak pointer
    weak_ptrs_.emplace_back(array_body);
}

void ArrayBodyLeakTracker::CheckAllFreed() {
    std::vector<std::shared_ptr<internal::ArrayBody>> alive_ptrs;

    for (const std::weak_ptr<internal::ArrayBody> weak_ptr : weak_ptrs_) {
        std::shared_ptr<internal::ArrayBody> ptr = weak_ptr.lock();
        if (ptr != nullptr) {
            alive_ptrs.emplace_back(ptr);
        }
    }

    if (!alive_ptrs.empty()) {
        // TODO(niboshi): Output only array bodies that are not referenced from other array bodies
        std::ostringstream os;
        os << "Some array bodies are not freed." << std::endl << "Number of alive array bodies: " << alive_ptrs.size() << std::endl;
        for (const std::shared_ptr<internal::ArrayBody>& array_body : alive_ptrs) {
            Array array{array_body};
            os << "- Unreleased array body: " << array_body.get() << std::endl;
            os << array << std::endl;
            for (const std::shared_ptr<ArrayNode>& array_node : array.nodes()) {
                const GraphId& graph_id = array_node->graph_id();
                DebugDumpComputationalGraph(os, array, graph_id);
            }
        }
        throw GradientCheckError{os.str()};
    }
}

ArrayBodyLeakDetectionScope ::ArrayBodyLeakDetectionScope(ArrayBodyLeakTracker& tracker) {
    assert(g_array_body_leak_tracker == nullptr);  // nested use is not supported
    g_array_body_leak_tracker = &tracker;
}

ArrayBodyLeakDetectionScope ::~ArrayBodyLeakDetectionScope() {
    if (!exited_) {
        g_array_body_leak_tracker = nullptr;
        exited_ = true;
    }
}

}  // namespace internal

Array::Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
    : body_{std::make_shared<internal::ArrayBody>(shape, strides, dtype, device, std::move(data), offset)} {
    if (g_array_body_leak_tracker != nullptr) {
        // TODO(niboshi): Make thread-safe
        (*g_array_body_leak_tracker)(body_);
    }
}

Array Array::operator-() const { return Negative(*this); }

Array Array::operator==(const Array& rhs) const { return Equal(*this, rhs); }

Array& Array::operator+=(const Array& rhs) {
    internal::IAdd(*this, rhs);
    return *this;
}

Array& Array::operator+=(Scalar rhs) {
    internal::IAdd(*this, rhs);
    return *this;
}

Array& Array::operator-=(const Array& rhs) {
    internal::ISubtract(*this, rhs);
    return *this;
}

Array& Array::operator-=(Scalar rhs) {
    internal::ISubtract(*this, rhs);
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    internal::IMultiply(*this, rhs);
    return *this;
}

Array& Array::operator*=(Scalar rhs) {
    internal::IMultiply(*this, rhs);
    return *this;
}

Array& Array::operator/=(const Array& rhs) {
    internal::IDivide(*this, rhs);
    return *this;
}

Array& Array::operator/=(Scalar rhs) {
    internal::IDivide(*this, rhs);
    return *this;
}

const Array& Array::operator+=(const Array& rhs) const {
    internal::IAdd(*this, rhs);
    return *this;
}

const Array& Array::operator+=(Scalar rhs) const {
    internal::IAdd(*this, rhs);
    return *this;
}

const Array& Array::operator-=(const Array& rhs) const {
    internal::ISubtract(*this, rhs);
    return *this;
}

const Array& Array::operator-=(Scalar rhs) const {
    internal::ISubtract(*this, rhs);
    return *this;
}

const Array& Array::operator*=(const Array& rhs) const {
    internal::IMultiply(*this, rhs);
    return *this;
}

const Array& Array::operator*=(Scalar rhs) const {
    internal::IMultiply(*this, rhs);
    return *this;
}

const Array& Array::operator/=(const Array& rhs) const {
    internal::IDivide(*this, rhs);
    return *this;
}

const Array& Array::operator/=(Scalar rhs) const {
    internal::IDivide(*this, rhs);
    return *this;
}

Array Array::operator+(const Array& rhs) const { return xchainer::Add(*this, rhs); }

Array Array::operator+(Scalar rhs) const { return xchainer::Add(*this, rhs); }

Array Array::operator-(const Array& rhs) const { return xchainer::Subtract(*this, rhs); }

Array Array::operator-(Scalar rhs) const { return xchainer::Subtract(*this, rhs); }

Array Array::operator*(const Array& rhs) const { return Multiply(*this, rhs); }

Array Array::operator*(Scalar rhs) const { return Multiply(*this, rhs); }

Array Array::operator/(const Array& rhs) const { return xchainer::Divide(*this, rhs); }

Array Array::operator/(Scalar rhs) const { return xchainer::Divide(*this, rhs); }

Array Array::At(const std::vector<ArrayIndex>& indices) const { return internal::At(*this, indices); }

Array Array::Transpose(const OptionalAxes& axes) const { return xchainer::Transpose(*this, axes); }

Array Array::Reshape(const Shape& newshape) const { return xchainer::Reshape(*this, newshape); }

Array Array::Squeeze(const OptionalAxes& axis) const { return xchainer::Squeeze(*this, axis); }

Array Array::BroadcastTo(const Shape& shape) const { return xchainer::BroadcastTo(*this, shape); }

Array Array::ArgMax(const OptionalAxes& axis) const { return xchainer::ArgMax(*this, axis); }

Array Array::Sum(const OptionalAxes& axis, bool keepdims) const { return xchainer::Sum(*this, axis, keepdims); }

Array Array::Max(const OptionalAxes& axis, bool keepdims) const { return xchainer::AMax(*this, axis, keepdims); }

Array Array::Dot(const Array& b) const { return xchainer::Dot(*this, b); }

Array Array::Take(const Array& indices, int8_t axis) const { return xchainer::Take(*this, indices, axis); }

Array Array::Copy() const { return xchainer::Copy(*this); }

Array Array::MakeView() const {
    Array out{std::make_shared<internal::ArrayBody>(shape(), strides(), dtype(), device(), data(), offset())};
    if (!IsConstant()) {
        BackwardBuilder bb{"view", out};
        bb.Define({*this}, [](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad(); });
    }
    return out;
}

Array Array::ToDevice(Device& dst_device) const {
    Device& src_device = body_->device_;
    Array out;

    // TODO(sonots): Avoid copying data between native devices, e.g., from native:0 to native:1 for performance.
    if (&src_device == &dst_device) {
        // Return an alias.
        out = AsConstant(CopyKind::kView);
    } else {
        // Make a contiguous copy to transfer it to the destination device.
        Array src_contig = AsContiguousArray(AsConstant(CopyKind::kView));

        std::shared_ptr<void> dst_data;
        if (src_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use src backend for transfer.
            dst_data = src_device.TransferDataTo(dst_device, src_contig.data(), src_contig.offset(), src_contig.GetNBytes());
        } else if (dst_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use dst backend for transfer.
            dst_data = dst_device.TransferDataFrom(src_device, src_contig.data(), src_contig.offset(), src_contig.GetNBytes());
        } else {
            // Neither backends support transfer.
            throw XchainerError{"Transfer between devices is not supported: src='", src_device.name(), "' dst='", dst_device.name(), "'."};
        }
        out = Array{src_contig.shape(), src_contig.strides(), src_contig.dtype(), dst_device, std::move(dst_data)};
    }

    assert(out.body() != nullptr);

    // Backward operation is implemented as backward-transfer.
    if (!IsConstant()) {
        BackwardBuilder bb{"transfer", out};
        bb.Define({*this}, [&src_device](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad().ToDevice(src_device); });
    }
    return out;
}

Array Array::ToNative() const {
    Context& context = device().backend().context();
    Device& native_device = context.GetNativeBackend().GetDevice(0);
    return ToDevice(native_device);
}

Array Array::AsConstant(CopyKind kind) const {
    switch (kind) {
        case CopyKind::kCopy: {
            Array out = EmptyLike(*this, device());
            device().Copy(*this, out);

            assert(out.IsContiguous());
            return std::move(out);
        }
        case CopyKind::kView:
            return Array{shape(), strides(), dtype(), device(), body_->data_, offset()};
        default:
            XCHAINER_NEVER_REACH();
    }
}

Array Array::AsConstant(gsl::span<const GraphId> graph_ids, CopyKind kind) const {
    switch (kind) {
        case CopyKind::kCopy: {
            Array out = EmptyLike(*this, device());
            device().Copy(*this, out);

            if (!IsConstant()) {
                BackwardBuilder bb{"copy", out, graph_ids};
                bb.Define({*this}, [](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad(); });
            }

            assert(out.IsContiguous());
            return std::move(out);
        }
        case CopyKind::kView: {
            Array out{std::make_shared<internal::ArrayBody>(shape(), strides(), dtype(), device(), data(), offset())};
            if (!IsConstantAfterStop(graph_ids)) {
                BackwardBuilder bb{"as_constant_view", out, graph_ids};
                bb.Define({*this}, [](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad(); });
            }
            return std::move(out);
        }
        default:
            XCHAINER_NEVER_REACH();
    }
}

Array Array::AsType(Dtype dtype, bool copy) const {
    Dtype src_dtype = this->dtype();
    if (!copy && dtype == src_dtype) {
        return *this;
    }

    Array out = Empty(shape(), dtype, device());
    device().AsType(*this, out);

    if (!IsConstant() && GetKind(dtype) == DtypeKind::kFloat) {
        BackwardBuilder bb{"astype", out};
        bb.Define({*this}, [src_dtype](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad().AsType(src_dtype); });
    }

    assert(out.IsContiguous());
    return out;
}

void Array::Fill(Scalar value) const { device().Fill(*this, value); }

const nonstd::optional<Array>& Array::GetGrad(const GraphId& graph_id) const {
    const nonstd::optional<Array>* grad = body_->GetGrad(graph_id);
    if (grad == nullptr) {
        throw XchainerError{"Array does not belong to the graph: '", graph_id, "'."};
    }
    return *grad;
}

void Array::SetGrad(Array grad, const GraphId& graph_id) const {
    nonstd::optional<Array>* target_grad = body_->GetGrad(graph_id);
    if (target_grad == nullptr) {
        throw XchainerError{"Array does not belong to the graph: '", graph_id, "'."};
    }
    internal::SetGrad(*target_grad, std::move(grad), shape(), dtype(), device());
}

void Array::ClearGrad(const GraphId& graph_id) const { body_->ClearGrad(graph_id); }

std::string Array::ToString() const { return ArrayRepr(*this); }

namespace {

class PrintComputationalGraphImpl {
private:
    using VisitedArrayNodeSet = std::unordered_set<const ArrayNode*>;

    struct State {
        VisitedArrayNodeSet visited_array_nodes;
        int indent;
    };

    // TODO(niboshi): Make the options configurable from outside
    struct Options {
        bool print_metadata{true};
    };

public:
    explicit PrintComputationalGraphImpl(std::ostream& os) : os_{os} {}

    void Run(const ArrayNode& array_node, int indent) {
        State state{{}, indent};
        RunImpl(state, array_node);
    }

    std::string GetArrayNodeName(const ArrayNode& array_node) {
        static constexpr char kChars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        static constexpr size_t kNumChars = sizeof(kChars) / sizeof(kChars[0]) - 1;
        static const auto kLen = static_cast<size_t>(std::ceil(sizeof(size_t) * 8U / std::log2(kNumChars)));
        auto it = array_name_map_.find(&array_node);
        if (it != array_name_map_.end()) {
            return it->second;
        }
        size_t hash = std::hash<const ArrayNode*>{}(&array_node);
        std::string s(kLen, '0');
        // Fill the string from left to right, because hash may be just the raw address and MSBs may be indistinguishable.
        for (auto it_s = s.begin(); hash > 0 && it_s != s.end(); ++it_s) {
            *it_s = gsl::at(kChars, hash % kNumChars);
            hash /= kNumChars;
        }
        return s;
    }

    std::string Indent(int indent) {
        static constexpr char kIndentChar = ' ';
        return std::string(static_cast<size_t>(indent * 2), kIndentChar);
    }

    void RunImpl(State& state, const ArrayNode& array_node) {
        std::string name = GetArrayNodeName(array_node);

        int indent = state.indent;
        VisitedArrayNodeSet& visited_array_nodes = state.visited_array_nodes;
        os_ << Indent(indent) << "ArrayNode<" << name << " " << &array_node << " rank=" << array_node.rank()
            << " shape=" << array_node.shape() << " dtype=" << GetDtypeName(array_node.dtype()) << ">" << std::endl;

        if (visited_array_nodes.end() == visited_array_nodes.find(&array_node)) {
            visited_array_nodes.insert(&array_node);

            if (options_.print_metadata) {
                std::shared_ptr<const internal::ArrayBody> body = array_node.GetBody();
                if (body == nullptr) {
                    os_ << Indent(indent + 2) << "body=(gone)" << std::endl;
                } else {
                    const nonstd::optional<Array>* grad = body->GetGrad(array_node.graph_id());
                    assert(grad != nullptr);
                    if (grad->has_value()) {
                        os_ << Indent(indent + 2) << "grad=<shape=" << (*grad)->shape() << " dtype=" << GetDtypeName((*grad)->dtype())
                            << ">" << std::endl;
                    }
                }
            }

            std::shared_ptr<const OpNode> op = array_node.next_op_node();
            if (op) {
                os_ << Indent(indent + 1) << "Op<" << op->name() << " " << op.get() << " rank=" << op->rank() << ">" << std::endl;
                for (const std::shared_ptr<const ArrayNode>& next_array_node : op->next_array_nodes()) {
                    state.indent += 2;
                    RunImpl(state, *next_array_node);
                    state.indent -= 2;
                }
            }
        }
    }

    void SetArrayName(const ArrayNode& array_node, std::string name) { array_name_map_[&array_node] = std::move(name); }

private:
    std::ostream& os_;
    Options options_{};
    std::unordered_map<const ArrayNode*, std::string> array_name_map_;
};

}  // namespace

void DebugDumpComputationalGraph(
        std::ostream& os,
        const Array& array,
        const GraphId& graph_id,
        int indent,
        const std::vector<std::pair<ConstArrayRef, std::string>>& array_name_map) {
    PrintComputationalGraphImpl impl{os};
    for (const auto& pair : array_name_map) {
        for (const std::shared_ptr<ArrayNode>& array_node : pair.first.get().nodes()) {
            if (array_node->graph_id() == graph_id) {
                impl.SetArrayName(*array_node, pair.second);
            }
        }
    }
    impl.Run(*internal::GetArrayNode(array, graph_id), indent);
}

}  // namespace xchainer
