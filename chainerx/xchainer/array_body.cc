#include "xchainer/array_body.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/array_body_leak_detection.h"
#include "xchainer/array_node.h"
#include "xchainer/backward.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/macro.h"

namespace xchainer {
namespace internal {

std::shared_ptr<ArrayBody> CreateArrayBody(
        const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset) {
    // Trick to use make_shared with private ctor
    struct ArrayBodyWithPublicCtor : ArrayBody {
        ArrayBodyWithPublicCtor(
                const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
            : ArrayBody{shape, strides, dtype, device, std::move(data), offset} {}
    };

    std::shared_ptr<ArrayBody> array_body =
            std::make_shared<ArrayBodyWithPublicCtor>(shape, strides, dtype, device, std::move(data), offset);

    if (internal::ArrayBodyLeakTracker* tracker = internal::ArrayBodyLeakDetectionScope::GetGlobalTracker()) {
        // TODO(niboshi): Make thread-safe
        (*tracker)(array_body);
    }

    return array_body;
}

std::shared_ptr<ArrayBody> CreateArrayBody(ArrayBody::Params params) {
    return CreateArrayBody(params.shape, params.strides, params.dtype, params.device, std::move(params.data), params.offset);
}

ArrayBody::ArrayBody(
        const Shape& shape,  // NOLINT(modernize-pass-by-value)
        const Strides& strides,  // NOLINT(modernize-pass-by-value)
        Dtype dtype,
        Device& device,
        std::shared_ptr<void> data,
        int64_t offset)
    : shape_{shape}, strides_{strides}, dtype_{dtype}, device_{device}, data_{std::move(data)}, offset_{offset} {}

ArrayBody::ArrayBody(Params params)
    : ArrayBody{params.shape, params.strides, params.dtype, params.device, std::move(params.data), params.offset} {}

const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(const std::shared_ptr<ArrayBody>& body, std::shared_ptr<ArrayNode> array_node) {
    body->AssertConsistency();

    // The body must be either unset (the array node is being created normally) or dead (the body is being replaced with a fabricated one,
    // as a retained output of backward)
    XCHAINER_ASSERT(array_node->weak_body().expired());

    auto it = std::find_if(body->nodes_.begin(), body->nodes_.end(), [&array_node](const std::shared_ptr<ArrayNode>& existing_node) {
        return existing_node->backprop_id() == array_node->backprop_id();
    });
    if (it != body->nodes_.end()) {
        return *it;  // Do nothing and return the existing ArrayNode if found for this graph.
    }

    // Connect the new backprop ID and the existing backprop IDs in this array body.
    for (const std::shared_ptr<ArrayNode>& existing_array_node : body->nodes_) {
        existing_array_node->device().context().ConnectBackpropIds(existing_array_node->backprop_id(), array_node->backprop_id());
    }

    array_node->weak_body_ = body;

    body->nodes_.emplace_back(std::move(array_node));
    body->grads_.emplace_back(std::make_unique<nonstd::optional<Array>>(nonstd::nullopt));

    body->AssertConsistency();
    return body->nodes_.back();
}

const std::shared_ptr<ArrayNode>& ArrayBody::CreateArrayNode(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id) {
    XCHAINER_ASSERT(GetKind(body->dtype()) == DtypeKind::kFloat);
    return AddNode(body, std::make_shared<ArrayNode>(body->shape_, body->dtype_, body->device_, backprop_id));
}

void ArrayBody::AssertConsistency() const {
    if (XCHAINER_DEBUG) {
        // Array with integral dtypes can neither have array nodes nor gradients.
        if (GetKind(dtype()) != DtypeKind::kFloat) {
            XCHAINER_ASSERT(nodes_.empty());
            XCHAINER_ASSERT(grads_.empty());
        }

        XCHAINER_ASSERT(nodes_.size() == grads_.size());
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const std::shared_ptr<ArrayNode>& array_node = nodes_[i];
            const nonstd::optional<Array>& grad = *grads_[i];
            XCHAINER_ASSERT(array_node != nullptr);
            XCHAINER_ASSERT(this == array_node->weak_body().lock().get());

            if (grad.has_value()) {
                XCHAINER_ASSERT(internal::GetArrayBody(*grad) != nullptr);
                XCHAINER_ASSERT(grad->shape() == array_node->shape());
                XCHAINER_ASSERT(grad->dtype() == array_node->dtype());
                XCHAINER_ASSERT(&grad->device() == &array_node->device());
            }
        }
    }
}

nonstd::optional<size_t> ArrayBody::GetNodeIndex(const BackpropId& backprop_id) const {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->backprop_id() == backprop_id) {
            return i;
        }
    }
    return nonstd::nullopt;
}

void ArrayBody::SetGrad(Array grad, const BackpropId& backprop_id) {
    nonstd::optional<Array>* target_grad = GetGrad(backprop_id);
    XCHAINER_ASSERT(target_grad != nullptr);
    internal::SetGrad(*target_grad, std::move(grad), shape_, dtype_, device_);
}

void ArrayBody::AccumulateGrad(Array partial_grad, const BackpropId& backprop_id) {
    nonstd::optional<Array>* target_grad = GetGrad(backprop_id);
    XCHAINER_ASSERT(target_grad != nullptr);
    internal::AccumulateGrad(*target_grad, std::move(partial_grad), shape_, dtype_, device_);
}

void ArrayBody::ClearGrad(const BackpropId& backprop_id) {
    nonstd::optional<Array>* grad = GetGrad(backprop_id);
    XCHAINER_ASSERT(grad != nullptr);
    grad->reset();
}

template <typename ThisPtr, typename ReturnType>
ReturnType ArrayBody::GetGradImpl(ThisPtr this_ptr, const BackpropId& backprop_id) {
    nonstd::optional<size_t> i = this_ptr->GetNodeIndex(backprop_id);
    if (!i.has_value()) {
        return nullptr;
    }
    XCHAINER_ASSERT(*i < this_ptr->grads_.size());
    return this_ptr->grads_[*i].get();
}

template nonstd::optional<Array>* ArrayBody::GetGradImpl<ArrayBody*, nonstd::optional<Array>*>(ArrayBody*, const BackpropId&);
template const nonstd::optional<Array>* ArrayBody::GetGradImpl<const ArrayBody*, const nonstd::optional<Array>*>(
        const ArrayBody*, const BackpropId&);

}  // namespace internal
}  // namespace xchainer
