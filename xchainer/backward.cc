#include "xchainer/backward.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"

namespace xchainer {
namespace internal {
namespace {

void CheckGradCompatible(const Array& grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckEqual(dtype, grad.dtype());
    CheckEqual(shape, grad.shape());
    CheckEqual(device, grad.device());
}

}  // namespace

void AccumulateGrad(nonstd::optional<Array>& target_grad, Array partial_grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckGradCompatible(partial_grad, shape, dtype, device);
    if (target_grad.has_value()) {
        target_grad = *target_grad + partial_grad;
    } else {
        target_grad = std::move(partial_grad);
    }
}

void SetGrad(nonstd::optional<Array>& target_grad, Array grad, const Shape& shape, Dtype dtype, Device& device) {
    CheckGradCompatible(grad, shape, dtype, device);
    target_grad = std::move(grad);
}

GradRef::GradRef(ArrayNode& array_node) : original_grad_owner_body_{array_node.GetBody()} {
    if (original_grad_owner_body_ != nullptr) {
        original_grad_ptr_ = original_grad_owner_body_->GetGrad(array_node.graph_id());
    }
}

GradRef::GradRef(nonstd::nullopt_t /*nullopt*/) : temporary_grad_{std::make_unique<nonstd::optional<Array>>()} {}

nonstd::optional<Array>& GradRef::get() {
    if (original_grad_ptr_ == nullptr) {
        if (temporary_grad_ == nullptr) {
            // Original gradient is gone and this is the first accumulation.
            // Initialize the temporary gradient.
            temporary_grad_ = std::make_unique<nonstd::optional<Array>>(nonstd::nullopt);
        }

        // Target of accumulation is the temporary gradient.
        return *temporary_grad_;
    }

    // Target of accumulation is the original gradient.
    return *original_grad_ptr_;
}

}  // namespace internal

RetainedOutputToken::RetainedOutputToken(std::shared_ptr<internal::ArrayBody> data_array_body, size_t output_index)
    : data_array_body_{std::move(data_array_body)}, output_index_{output_index} {
    assert(data_array_body_ != nullptr);
    // TODO(niboshi): Could be written as: assert(data_array_body_.nodes().empty())
    assert(!internal::HasAnyArrayNode(Array{data_array_body_}));
}

const std::shared_ptr<internal::ArrayBody>& RetainedOutputToken::GetFabricatedArrayBodyWithNodes(
        const std::shared_ptr<OpNode>& op_node) const {
    assert(op_node != nullptr);
    std::vector<std::shared_ptr<ArrayNode>> new_prev_array_nodes;

    // Loop over graphs to collect array nodes corresponding to the same output index
    for (const auto& tup : op_node->prev_array_nodes_of_all_graphs()) {
        const GraphId& graph_id = std::get<0>(tup);
        const std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes = std::get<1>(tup);

        // If previous array node is alive, add the node to the array body.
        // Otherwise, create a new array node out of an op node of the corresponding graph.
        std::shared_ptr<ArrayNode> prev_array_node = prev_array_nodes[output_index_].lock();
        if (prev_array_node == nullptr) {
            std::shared_ptr<OpNode> new_op_node{};
            if (op_node->graph_id() == graph_id) {
                // Creating prev array node for "this" graph, based on the current op node
                new_op_node = op_node;
            } else {
                // Creating prev array node for other graph, based on mocked op node in the other graph
                new_op_node = op_node->CloneInOtherGraph(graph_id);
            }
            // Create mocked prev array node which refers to the op node
            const internal::ArrayProps& props = op_node->GetPrevArrayProps(output_index_);
            prev_array_node = std::make_shared<ArrayNode>(props.shape, props.dtype, props.device, graph_id);
            prev_array_node->set_next_op_node(std::move(new_op_node));
        }

        assert(prev_array_node->graph_id() == graph_id);
        assert(prev_array_node->GetBody() == nullptr);
        new_prev_array_nodes.emplace_back(std::move(prev_array_node));
    }

    // Create a new array body with (possibly fabricated) array nodes.
    // The data array body stored in the token is reused as a base.
    for (const std::shared_ptr<ArrayNode>& prev_array_node : new_prev_array_nodes) {
        assert(prev_array_node->GetBody() == nullptr);
        prev_array_node->set_array_body(data_array_body_);
        data_array_body_->AddNode(prev_array_node);
    }

    return data_array_body_;
}

BackwardContext::BackwardContext(
        const std::shared_ptr<OpNode>& op_node,
        gsl::span<ArrayNode*> prev_array_nodes,
        gsl::span<internal::GradRef*> prev_grads,
        std::vector<Array>& input_grads_storage,
        const GraphId& graph_id,
        DoubleBackpropOption double_backprop_option)
    : op_node_{op_node},
      prev_array_nodes_{prev_array_nodes},
      prev_grads_{prev_grads},
      input_grads_storage_{input_grads_storage},
      zero_output_grads_{prev_array_nodes_.size()},
      graph_id_{graph_id},
      double_backprop_option_{double_backprop_option} {
    assert(prev_array_nodes_.size() == prev_grads_.size());
    // Input grads must be initialized with null-body arrays.
    assert(std::all_of(input_grads_storage_.begin(), input_grads_storage_.end(), [](const Array& g) { return g.body() == nullptr; }));

    retained_output_array_bodies_.resize(op_node->prev_node_count());  // Fill with nullptr
};

bool BackwardContext::HasOutputGrad(size_t output_index) const { return gsl::at(prev_grads_, output_index)->get().has_value(); }

const Array& BackwardContext::output_grad(size_t output_index) const {
    // If the output gradient has a propagated value, return it.
    if (HasOutputGrad(output_index)) {
        return *prev_grads_[output_index]->get();
    }

    // If there already is a zero-filled gradient allocated, return it.
    assert(output_index < output_count());
    nonstd::optional<Array>& zero_grad = zero_output_grads_[output_index];
    if (zero_grad.has_value()) {
        return *zero_grad;
    }

    // Allocate new zero-filled gradient and return it.
    const internal::ArrayProps& props = op_node_->GetPrevArrayProps(output_index);
    zero_grad = Zeros(props.shape, props.dtype, props.device);
    return *zero_grad;
}

Array& BackwardContext::input_grad() {
    assert(input_grads_storage_.size() == 1);
    return gsl::at(input_grads_storage_, 0);
}

Array& BackwardContext::input_grad(size_t index) { return gsl::at(input_grads_storage_, index); }

Array BackwardContext::GetRetainedOutput(const RetainedOutputToken& token) {
    assert(token.output_index() < output_count());
    size_t output_index = token.output_index();

    // Retrieve the kept array body for retained output.
    // Note that it's a non-const reference so that the following logic can assign to it to keep it for the repeated retrieval of the
    // retained array.
    std::shared_ptr<internal::ArrayBody>& kept_body = retained_output_array_bodies_[output_index];

    if (kept_body == nullptr) {
        // This is the first retrieval of the retained output.
        // If the original output array body is still alive. Just make a copy of array body with restricted array nodes.
        // Otherwise, a new array body is fabricated.

        // Retrieve the array body of the original output array.
        std::shared_ptr<internal::ArrayBody> array_body{nullptr};
        if (ArrayNode* prev_array_node = prev_array_nodes_[output_index]) {
            array_body = prev_array_node->GetBody();
        }

        if (array_body == nullptr) {
            // Fabricate a new array body
            array_body = token.GetFabricatedArrayBodyWithNodes(op_node_);
        }

        // Cut graphs of the array body
        kept_body = Array{std::move(array_body)}.MakeView().move_body();
    }

    assert(kept_body != nullptr);
    return Array{kept_body};
}

size_t BackwardContext::output_count() const { return zero_output_grads_.size(); }

BackwardBuilder::BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs)
    : op_name_{op_name}, outputs_{outputs.begin(), outputs.end()} {
    // Outputs requiring grad (e.g. in-place ops.) must have been detected and reported before reaching here.
    assert(std::all_of(outputs.begin(), outputs.end(), [](const Array& output) { return !internal::HasAnyArrayNode(output); }));
    // All output arrays must have the same device.
    assert(std::all_of(outputs.begin(), outputs.end(), [&outputs](const Array& output) {
        return &outputs.begin()->get().device() == &output.device();
    }));
    output_array_props_.reserve(outputs_.size());
    std::transform(
            outputs_.begin(), outputs_.end(), std::back_inserter(output_array_props_), [](const Array& output) -> internal::ArrayProps {
                return {output.shape(), output.dtype(), output.device()};
            });
}

void BackwardBuilder::Define(std::initializer_list<ConstArrayRef> inputs, const BackwardFunction& backward_func) {
    // `outputs` may or may not include non-constant arrays, because `BackwardBuilder::Define` may be called repeatedly in a single op.
    // At the beginning of this function, `op_node_map` holds the op nodes created in the previous calls of `BackwardBuilder::Define`
    // for this op.

    // All input arrays must have the same device.
    assert(std::all_of(
            inputs.begin(), inputs.end(), [&inputs](const Array& input) { return &input.device() == &(inputs.begin()->get().device()); }));

    // Collect input array nodes, grouped by graph.
    // However, skip the array nodes that belong to graphs for which gradients should be stopped,
    // by creating a temporary no-backprop scope.
    // If only a subset of input arrays have array nodes that require grad, nullptrs are assigned in place of other array nodes that do not.
    // TODO(niboshi): Probably linear search with a simple vector is faster than hash table.
    using NextArrayNodes = std::vector<const std::shared_ptr<ArrayNode>*>;
    std::unordered_map<GraphId, NextArrayNodes> graph_to_next_array_nodes;
    for (size_t i_input = 0; i_input < inputs.size(); ++i_input) {
        const Array& input = *(inputs.begin() + i_input);

        for (std::shared_ptr<ArrayNode>& next_array_node : input.nodes()) {
            const GraphId& graph_id = next_array_node->graph_id();

            if (!IsBackpropRequired(graph_id)) {
                continue;
            }

            // Add the array node to the mapping
            auto insert_result = graph_to_next_array_nodes.emplace(graph_id, NextArrayNodes{});
            auto& vec = insert_result.first->second;
            if (insert_result.second) {
                // New array node for a graph. Fill all array nodes with nullptr.
                vec.resize(inputs.size());
            }
            // Assign valid pointer to the array node.
            vec[i_input] = &next_array_node;
        }
    }

#ifndef NDEBUG
    for (auto& pair : graph_to_next_array_nodes) {
        const GraphId& graph_id = pair.first;
        const NextArrayNodes& vec = pair.second;
        for (const std::shared_ptr<ArrayNode>* array_node : vec) {
            assert(array_node == nullptr || graph_id == (*array_node)->graph_id());
        }
    }
#endif  // NDEBUG

    // Pointers to op nodes involved in this backward function
    std::vector<OpNode*> op_nodes;

    // Create op node for each graph
    for (auto it_graph = graph_to_next_array_nodes.begin(); it_graph != graph_to_next_array_nodes.end(); ++it_graph) {
        const GraphId& graph_id = it_graph->first;
        const NextArrayNodes& next_array_nodes = it_graph->second;

        // Find op node
        auto insert_result = op_node_map_.emplace(graph_id, nullptr);
        if (insert_result.second) {
            // Create new op instance
            std::vector<std::weak_ptr<ArrayNode>> weak_prev_array_nodes;  // weak pointers to pass to new op node
            std::vector<ArrayNode*> prev_array_nodes;
            weak_prev_array_nodes.reserve(outputs_.size());
            prev_array_nodes.reserve(outputs_.size());
            for (const Array& out : outputs_) {
                const std::shared_ptr<ArrayNode>& prev_array_node = xchainer::internal::HasArrayNode(out, graph_id)
                                                                            ? xchainer::internal::GetMutableArrayNode(out, graph_id)
                                                                            : xchainer::internal::CreateArrayNode(out, graph_id);
                prev_array_nodes.emplace_back(prev_array_node.get());
                weak_prev_array_nodes.emplace_back(prev_array_node);
            }
            // Create new op instance with weakrefs to output nodes
            std::shared_ptr<OpNode>& new_op_node = insert_result.first->second =
                    std::make_shared<OpNode>(op_name_, graph_id, weak_prev_array_nodes, output_array_props_);
            // Add edges from the output nodes
            for (ArrayNode* prev_array_node : prev_array_nodes) {
                assert(prev_array_node->next_op_node() == nullptr);
                prev_array_node->set_next_op_node(new_op_node);
            }
        }

        std::shared_ptr<OpNode>& op_node = insert_result.first->second;

        // Keep the list of op nodes involved in this backward function
        op_nodes.emplace_back(op_node.get());

        // Add edges to the input nodes
        std::vector<std::shared_ptr<ArrayNode>> temp_next_array_nodes;
        temp_next_array_nodes.reserve(next_array_nodes.size());
        std::transform(
                next_array_nodes.begin(),
                next_array_nodes.end(),
                std::back_inserter(temp_next_array_nodes),
                [](const std::shared_ptr<ArrayNode>* array_node) { return array_node == nullptr ? nullptr : *array_node; });
        internal::OpNodeBackwardEntry& backward_entry = op_node->RegisterBackwardFunction(std::move(temp_next_array_nodes), backward_func);

        // Add edges to next array nodes of other graphs.
        // TODO(niboshi): Do this only when BackwardBuilder::RetainOutput() is called.
        for (auto it_exotic_graph = graph_to_next_array_nodes.begin(); it_exotic_graph != graph_to_next_array_nodes.end();
             ++it_exotic_graph) {
            if (it_graph == it_exotic_graph) {
                continue;
            }
            assert(graph_id != it_exotic_graph->first);
            const GraphId& exotic_graph_id = it_exotic_graph->first;
            const NextArrayNodes& exotic_next_array_nodes = it_exotic_graph->second;

            std::vector<std::shared_ptr<ArrayNode>> temp_array_nodes;
            temp_array_nodes.reserve(exotic_next_array_nodes.size());
            std::transform(
                    exotic_next_array_nodes.begin(),
                    exotic_next_array_nodes.end(),
                    std::back_inserter(temp_array_nodes),
                    [](const std::shared_ptr<ArrayNode>* array_node) { return array_node == nullptr ? nullptr : *array_node; });
            backward_entry.AddExoticNextArrayNode(
                    std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>{exotic_graph_id, std::move(temp_array_nodes)});
        }
    }

    // Add weak ptrs from the op nodes to previous array nodes of other graphs.
    // TODO(niboshi): Do this only when BackwardBuilder::RetainOutput() is called.
    if (!any_defined_ && op_nodes.size() >= 2) {  // op_nodes.size() is the number of graphs
        std::unordered_map<GraphId, std::vector<std::shared_ptr<ArrayNode>>> exotic_array_nodes;

        for (const Array& output : outputs_) {
            for (const std::shared_ptr<ArrayNode>& output_array_node : output.nodes()) {
                exotic_array_nodes[output_array_node->graph_id()].emplace_back(output_array_node);
            }
        }

        for (OpNode* op_node : op_nodes) {
            for (const auto& tup : exotic_array_nodes) {
                assert(tup.second.size() == outputs_.size());
                if (tup.first == op_node->graph_id()) {
                    continue;
                }
                std::vector<std::weak_ptr<ArrayNode>> weak_prev_array_nodes;
                weak_prev_array_nodes.reserve(tup.second.size());
                std::transform(
                        tup.second.begin(),
                        tup.second.end(),
                        std::back_inserter(weak_prev_array_nodes),
                        [](const std::shared_ptr<ArrayNode>& array_node) { return std::weak_ptr<ArrayNode>{array_node}; });
                op_node->RegisterExoticPreviousArrayNodes(tup.first, std::move(weak_prev_array_nodes));
            }
        }
    }

    assert(!op_node_map_.empty());
    any_defined_ = true;
}

RetainedOutputToken BackwardBuilder::RetainOutput(const Array& output) {
    // Find the corresponding output index.
    // If there are more than one array with the same array body in outputs, the first one is always chosen, no matter what array the caller
    // actually specified. It doesn't matter because the array GetRetainedOutput would return is the same.

    // TODO(niboshi): It may be costly in ops with many output arrays.
    auto it = std::find_if(outputs_.begin(), outputs_.end(), [&output](const Array& output2) { return output.body() == output2.body(); });
    if (it == outputs_.end()) {
        throw XchainerError{"Cannot retain an array which is not any of output arrays."};
    }
    size_t output_index = std::distance(outputs_.begin(), it);
    std::shared_ptr<internal::ArrayBody> data_array_body = output.AsGradStopped().move_body();
    assert(data_array_body.get() != output.body().get());
    return {std::move(data_array_body), output_index};
}

namespace {

struct OpNodeComparator {
    bool operator()(const std::shared_ptr<OpNode>& lhs, const std::shared_ptr<OpNode>& rhs) const { return lhs->rank() < rhs->rank(); }
};

class BackwardImpl {
public:
    BackwardImpl(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop)
        : outputs_{outputs}, graph_id_{graph_id}, double_backprop_{double_backprop} {
        for (const Array& output : outputs) {
            if (!output.IsGradRequired(graph_id)) {
                throw XchainerError{"Cannot start backprop from an array whose gradient is not required (on graph '", graph_id, "')"};
            }
            output_array_nodes_.emplace_back(internal::GetMutableArrayNode(output, graph_id));
        }
    }

    void Run() {
        // Push initial output array nodes
        for (size_t i = 0; i < outputs_.size(); ++i) {
            const Array& output = outputs_[i];
            const std::shared_ptr<ArrayNode>& array_node = output_array_nodes_[i];

            // Add GradRef for output array nodes
            auto emplace_result = array_node_grad_map_.emplace(array_node.get(), internal::GradRef{*array_node});

            // Set unset output gradients to the default value of one
            if (!emplace_result.first->second.get().has_value()) {
                emplace_result.first->second.get() = OnesLike(output, output.device());
            }

            PushNextOpNode(array_node);
        }

        // Backpropagation
        while (!candidate_op_nodes_.empty()) {
            std::pop_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            std::shared_ptr<OpNode> op_node = std::move(candidate_op_nodes_.back());
            candidate_op_nodes_.pop_back();

            // Add GradRef for next array nodes
            for (const std::shared_ptr<ArrayNode>& next_array_node : op_node->next_array_nodes()) {
                assert(next_array_node != nullptr);
                array_node_grad_map_.emplace(next_array_node.get(), internal::GradRef{*next_array_node});
            }

            // Backpropagate gradients from the previous array nodes into the next array nodes.
            {
                std::vector<nonstd::optional<Array>> gxs = ComputeNextGradients(op_node, graph_id_);
                AccumulateNextGradients(*op_node, std::move(gxs));
            }

            // Push the next op nodes into the queue
            for (const auto& next_array_node : op_node->next_array_nodes()) {
                PushNextOpNode(next_array_node);
            }

            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                op_node->Unchain();
            }

            // Erase the array node's temporarily held grad
            {
                auto range = previous_array_node_keeper_.equal_range(op_node.get());
                for (auto it = range.first; it != range.second; ++it) {
                    size_t n_removed = array_node_grad_map_.erase(it->second.get());
                    (void)n_removed;  // unused
                    assert(n_removed > 0);
                }
            }
        }
    }

private:
    std::vector<nonstd::optional<Array>> ComputeNextGradients(const std::shared_ptr<OpNode>& op_node, const GraphId& graph_id) {
        assert(op_node != nullptr);

        // Determine graph IDs to stop gradients
        std::vector<GraphId> graph_ids_to_stop_gradient;
        if (double_backprop_ == DoubleBackpropOption::kDisable) {
            graph_ids_to_stop_gradient.emplace_back(graph_id);
        }

        // Run backward functions to compute gradients of next array nodes.
        std::vector<nonstd::optional<Array>> next_grads;
        next_grads.resize(op_node->next_array_node_count());

        // Previous array nodes. May be nullptr if the node is gone.
        std::vector<ArrayNode*> prev_array_nodes;

        // `temp_prev_grads` is a set of temporary GradRefs of this op node's previous array nodes.
        // This is used for previous array nodes which are either dead at the moment or alive but have not been involved in the preceding
        // backpropagation.
        // This vector is just a keeper and not used in any other way. prev_grads holds the pointer to it.
        // These GradRefs are only valid in the backward functions of this op node.
        // Be careful not to cause reallocation in this vector. Otherwise the pointers would be invalidated.
        std::vector<internal::GradRef> temp_prev_grads;
        temp_prev_grads.reserve(op_node->prev_array_nodes().size());

        std::vector<internal::GradRef*> prev_grads;
        for (const std::weak_ptr<ArrayNode>& maybe_prev_array_node : op_node->prev_array_nodes()) {
            std::shared_ptr<ArrayNode> prev_array_node = maybe_prev_array_node.lock();
            prev_array_nodes.emplace_back(prev_array_node.get());

            // Get the pointer to the previous gradient.
            if (prev_array_node != nullptr) {
                // Previous array node is alive.
                auto it = array_node_grad_map_.find(prev_array_node.get());
                if (it != array_node_grad_map_.end()) {
                    // The grad mapping has the gradient for the array node.
                    // Keep a pointer to the gradient in the map.
                    prev_grads.emplace_back(&it->second);
                } else {
                    // The grad mapping has no entry for the array node.
                    // Create a new entry in temporary gradients and keep a pointer to it.
                    temp_prev_grads.emplace_back(*prev_array_node);
                    prev_grads.emplace_back(&temp_prev_grads.back());
                }
            } else {
                // Previous array node is dead.
                // Keep a pointer to the temporary gradient vector.
                temp_prev_grads.emplace_back(nonstd::nullopt);
                prev_grads.emplace_back(&temp_prev_grads.back());
            }
        }

        for (const internal::OpNodeBackwardEntry& backward_entry : op_node->backward_entries()) {
            // `next_grads_subset` stores the next gradients (`next_grads`) of the subset of input arrays of this backward
            // call. `BackwardContext` holds it by reference and assignment to BackwardContext::input_grad() stores the
            // gradients there. It initially holds null-body arrays.
            std::vector<Array> next_grads_subset;
            next_grads_subset.resize(backward_entry.next_array_node_count());

            // Call backward.
            BackwardContext bctx{op_node, prev_array_nodes, prev_grads, next_grads_subset, graph_id_, double_backprop_};
            {
                NoBackpropModeScope scope{graph_ids_to_stop_gradient};
                backward_entry.backward_func()(bctx);
            }

            for (auto it = next_grads_subset.begin(); it != next_grads_subset.end(); ++it) {
                // TODO(sonots): Allow backward without setting input grads
                assert(it->body() != nullptr);
                // Make a view if the next gradient is identical to one of other prev or next gradients.
                // TODO(niboshi): Check node identity instead of body identity.
                if (std::any_of(
                            prev_array_nodes.begin(),
                            prev_array_nodes.end(),
                            [it, this](const ArrayNode* prev_array_node) {
                                if (prev_array_node == nullptr) {
                                    return false;
                                }
                                std::shared_ptr<const internal::ArrayBody> body = prev_array_node->GetBody();
                                if (body == nullptr) {
                                    return false;
                                }
                                const nonstd::optional<Array>* prev_grad = body->GetGrad(graph_id_);
                                return prev_grad != nullptr && prev_grad->has_value() && it->body() == (*prev_grad)->body();
                            }) ||
                    std::any_of(next_grads_subset.begin(), it, [it](const Array& next_grad) { return next_grad.body() == it->body(); })) {
                    // TODO(niboshi): View is needed to make new nodes. Come up with a solution to avoid extra backward insertion.
                    *it = it->MakeView();
                }
            }

            // Accumulate grads from `next_grads_subset`.
            for (size_t i = 0; i < backward_entry.next_array_node_count(); ++i) {
                nonstd::optional<size_t> i_next_grad = backward_entry.next_array_node_indices()[i];
                if (!i_next_grad.has_value()) {
                    continue;  // grad is not required for this input
                }
                nonstd::optional<Array>& target_grad = next_grads[*i_next_grad];
                const ArrayNode& next_array_node = *op_node->next_array_nodes()[*i_next_grad];

                internal::AccumulateGrad(
                        target_grad,
                        std::move(next_grads_subset[i]),
                        next_array_node.shape(),
                        next_array_node.dtype(),
                        next_array_node.device());
            }
        }

        // If previous array nodes are not output nodes of backward, clear their gradients
        for (ArrayNode* prev_array_node : prev_array_nodes) {
            if (prev_array_node == nullptr) {
                continue;
            }
            if (std::find_if(
                        output_array_nodes_.begin(),
                        output_array_nodes_.end(),
                        [prev_array_node](const std::shared_ptr<ArrayNode>& out_node) { return prev_array_node == out_node.get(); }) ==
                output_array_nodes_.end()) {
                if (prev_array_node != nullptr) {
                    std::shared_ptr<internal::ArrayBody> body = prev_array_node->GetBody();
                    if (body != nullptr) {
                        body->ClearGrad(prev_array_node->graph_id());
                    }
                }
            }
        }

        // Erase processed OpNode from the map
        previous_array_node_keeper_.erase(op_node.get());

        return next_grads;
    }

    void AccumulateNextGradients(const OpNode& op_node, std::vector<nonstd::optional<Array>> gxs) {
        gsl::span<const std::shared_ptr<ArrayNode>> next_array_nodes = op_node.next_array_nodes();
        assert(next_array_nodes.size() == gxs.size());
        for (size_t i = 0; i < next_array_nodes.size(); ++i) {
            const ArrayNode& next_array_node = *next_array_nodes[i];
            nonstd::optional<Array>& gx = gxs[i];
            if (gx.has_value()) {
                // Retrieve the pointer to the next gradient.
                internal::GradRef& next_grad = array_node_grad_map_.at(next_array_nodes[i].get());
                internal::AccumulateGrad(
                        next_grad.get(), std::move(*gx), next_array_node.shape(), next_array_node.dtype(), next_array_node.device());
            }
        }
    }

    void PushNextOpNode(const std::shared_ptr<ArrayNode>& array_node) {
        // When double backprop is enabled, array_node releases the pointer to the next node here. After this operation, array_node will
        // look like a leaf node of the graph. Note that this move does not invalidates the array_node object itself; it is guaranteed
        // by the standard that shared_ptr becomes null after move-assigned to another.
        std::shared_ptr<OpNode> next_op_node =
                double_backprop_ == DoubleBackpropOption::kEnable ? array_node->next_op_node() : array_node->move_next_op_node();

        if (next_op_node) {
            auto range = previous_array_node_keeper_.equal_range(next_op_node.get());
            if (std::none_of(range.first, range.second, [&array_node](const auto& pair) { return pair.second == array_node; })) {
                // First appearance of the combination of op node and next node.
                bool is_first_visit = range.first == range.second;
                previous_array_node_keeper_.emplace(next_op_node.get(), array_node);  // Iterators are invalidated here.
                if (is_first_visit) {
                    // First appearance of this op node. Push it to the queue.
                    candidate_op_nodes_.push_back(std::move(next_op_node));
                    std::push_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
                }
            }
        }
    }

    // Op nodes to be visited. This is a max heap ordered by the rank of each op node (see OpNodeComparator).
    std::vector<std::shared_ptr<OpNode>> candidate_op_nodes_;

    // This mapping is used to keep previous array nodes alive (referenced from op nodes as weak pointers).
    std::unordered_multimap<const OpNode*, std::shared_ptr<ArrayNode>> previous_array_node_keeper_;

    // Mapping from array nodes to the corresponding gradients. Gradients may be genuine gradients held by array bodies or temporary
    // gradients which are only valid during backward computation at most.
    std::unordered_map<ArrayNode*, internal::GradRef> array_node_grad_map_;

    // Arguments to Backward().
    // Be careful that references require the referred objects alive (it should be guaranteed by Backward()).
    const std::vector<ConstArrayRef>& outputs_;
    std::vector<std::reference_wrapper<const std::shared_ptr<ArrayNode>>> output_array_nodes_;
    const GraphId& graph_id_;  // NOLINT: intentionally holding a reference
    DoubleBackpropOption double_backprop_;
};

}  // namespace

void Backward(const Array& output, const GraphId& graph_id, DoubleBackpropOption double_backprop) {
    std::vector<ConstArrayRef> outputs{output};  // Do not inline it; we need to guarantee that the vector is alive until Run() finishes.
    BackwardImpl{outputs, graph_id, double_backprop}.Run();
}

void Backward(const std::vector<ConstArrayRef>& outputs, const GraphId& graph_id, DoubleBackpropOption double_backprop) {
    BackwardImpl{outputs, graph_id, double_backprop}.Run();
}

}  // namespace xchainer
