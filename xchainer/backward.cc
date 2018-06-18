#include "xchainer/backward.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"

namespace xchainer {

BackwardContext::BackwardContext(
        const OpNode& op_node,
        gsl::span<const std::reference_wrapper<ArrayNode>> prev_nodes,
        gsl::span<const GraphId> stop_graph_ids,
        std::vector<Array>& input_grads_storage,
        bool next_backward_required)
    : op_node_{op_node},
      prev_nodes_{prev_nodes},
      stop_graph_ids_{stop_graph_ids},
      input_grads_storage_{input_grads_storage},
      zero_output_grads_{prev_nodes_.size()},
      next_backward_required_{next_backward_required} {
    assert(input_grads_storage_.size() <= op_node.next_node_count());
    // Input grads must be initialized with null-body arrays.
    assert(std::all_of(input_grads_storage_.begin(), input_grads_storage_.end(), [](const Array& g) { return g.body() == nullptr; }));
};

bool BackwardContext::HasOutputGrad(int output_index) const { return gsl::at(prev_nodes_, output_index).get().grad().has_value(); }

const Array& BackwardContext::output_grad(int output_index) const {
    // If the output gradient has a propagated value, return it.
    if (HasOutputGrad(output_index)) {
        return *prev_nodes_[output_index].get().grad();
    }

    // If there already is a zero-filled gradient allocated, return it.
    assert(output_index < static_cast<int>(zero_output_grads_.size()));
    nonstd::optional<Array>& zero_grad = zero_output_grads_[output_index];
    if (zero_grad.has_value()) {
        return *zero_grad;
    }

    // Allocate new zero-filled gradient an return it.
    const ArrayNode& prev_node = prev_nodes_[output_index];
    zero_grad = Zeros(prev_node.shape(), prev_node.dtype(), prev_node.device());
    return *zero_grad;
}

Array BackwardContext::Cut(const Array& a) const {
#ifndef NDEBUG
    for (const ArrayNode& prev_node : prev_nodes_) {
        assert((!prev_node.grad().has_value() || &(*prev_node.grad()) != &a) && "Output grads do not have to be cut");
    }
#endif /*NDEBUG*/
    return a.AsConstant(stop_graph_ids_);
}

BackwardBuilder::BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs, gsl::span<const GraphId> stop_graph_ids)
    : op_name_{op_name}, outputs_{outputs.begin(), outputs.end()}, stop_graph_ids_{stop_graph_ids.begin(), stop_graph_ids.end()} {
    // Non-const outputs (e.g. in-place ops.) must have been detected and repored before reaching here.
    assert(std::all_of(outputs.begin(), outputs.end(), [](const Array& output) { return output.IsConstant(); }));
    // All output arrays must have the same device.
    assert(std::all_of(outputs.begin(), outputs.end(), [&outputs](const Array& output) {
        return &outputs.begin()->get().device() == &output.device();
    }));
}

void BackwardBuilder::Define(std::initializer_list<ConstArrayRef> inputs, const BackwardFunction& backward_func) {
    // `outputs` may or may not include non-constant arrays, because `BackwardBuilder::Define` may be called repeatedly in a single op.
    // At the beginning of this function, `op_node_map` holds the op nodes created in the previous calls of `BackwardBuilder::Define`
    // for this op.

    // All input arrays must have the same device.
    assert(std::all_of(
            inputs.begin(), inputs.end(), [&inputs](const Array& input) { return &input.device() == &(inputs.begin()->get().device()); }));

    // Collect input nodes, grouped by graph
    // TODO(niboshi): Probably linear search with a simple vector is faster than hash table.
    using NextArrayNodes = std::vector<std::reference_wrapper<std::shared_ptr<ArrayNode>>>;
    std::unordered_map<GraphId, NextArrayNodes> graph_to_next_array_nodes;
    for (const Array& input : inputs) {
        for (std::shared_ptr<ArrayNode>& next_array_node : input.nodes()) {
            const GraphId& graph_id = next_array_node->graph_id();

            // Skip if the node belong to the graphs to stop gradients
            if (stop_graph_ids_.end() != std::find(stop_graph_ids_.begin(), stop_graph_ids_.end(), graph_id)) {
                continue;
            }

            // Add the array node to the mapping
            auto& vec = graph_to_next_array_nodes[graph_id];
            vec.emplace_back(next_array_node);
        }
    }

    // Create op node for each graph
    for (auto& pair : graph_to_next_array_nodes) {
        const GraphId& graph_id = pair.first;
        NextArrayNodes& next_array_nodes = pair.second;

        // Find op node
        auto insert_result = op_node_map_.emplace(graph_id, nullptr);
        if (insert_result.second) {
            // Create new op instance
            std::vector<std::shared_ptr<ArrayNode>> prev_array_nodes;
            for (const Array& out : outputs_) {
                const std::shared_ptr<ArrayNode>& prev_array_node = xchainer::internal::HasArrayNode(out, graph_id)
                                                                            ? xchainer::internal::GetMutableArrayNode(out, graph_id)
                                                                            : xchainer::internal::CreateArrayNode(out, graph_id);
                prev_array_nodes.emplace_back(prev_array_node);
            }
            // Create new op instance with weakrefs to output nodes
            std::shared_ptr<OpNode>& new_op_node = insert_result.first->second = std::make_shared<OpNode>(op_name_, prev_array_nodes);
            // Add edges from the output nodes
            for (std::shared_ptr<ArrayNode>& prev_array_node : prev_array_nodes) {
                assert(prev_array_node->next_node() == nullptr);
                prev_array_node->set_next_node(new_op_node);
            }
        }

        // Add edges to the input nodes
        std::shared_ptr<OpNode>& op_node = insert_result.first->second;
        op_node->RegisterBackwardFunction(next_array_nodes, backward_func);
    }

    assert(!op_node_map_.empty());
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

            if (!array_node->grad()) {
                array_node->set_grad(OnesLike(output, output.device()));
            }
            PushNextOpNode(array_node);
        }

        // Backpropagation
        while (!candidate_op_nodes_.empty()) {
            std::pop_heap(candidate_op_nodes_.begin(), candidate_op_nodes_.end(), OpNodeComparator{});
            std::shared_ptr<OpNode> op_node = std::move(candidate_op_nodes_.back());
            candidate_op_nodes_.pop_back();

            {
                std::vector<nonstd::optional<Array>> gxs = ComputeNextGradients(*op_node, graph_id_);
                AccumulateNextGradients(*op_node, std::move(gxs));
            }

            for (const auto& next_array_node : op_node->next_nodes()) {
                PushNextOpNode(next_array_node);
            }

            if (double_backprop_ == DoubleBackpropOption::kDisable) {
                op_node->Unchain();
            }
        }
    }

private:
    std::vector<nonstd::optional<Array>> ComputeNextGradients(const OpNode& op_node, const GraphId& graph_id) {
        // Determine graph IDs to stop gradients
        std::vector<GraphId> graph_ids_to_stop_gradient;
        if (double_backprop_ == DoubleBackpropOption::kDisable) {
            graph_ids_to_stop_gradient.emplace_back(graph_id);
        }

        // Run backward functions to compute gradients of next array nodes.
        std::vector<nonstd::optional<Array>> next_grads;
        next_grads.resize(op_node.next_node_count());

        std::vector<std::reference_wrapper<ArrayNode>> prev_nodes;
        for (const std::weak_ptr<ArrayNode>& weak_prev_node : op_node.prev_nodes()) {
            std::shared_ptr<ArrayNode> prev_node = weak_prev_node.lock();
            assert(prev_node != nullptr);  // Weak ptrs should be alive
            prev_nodes.emplace_back(*prev_node);
        }

        for (const internal::OpNodeBackwardEntry& backward_entry : op_node.backward_entries()) {
            // `next_grads_subset` stores the next gradients (`next_grads`) of the subset of input arrays of this backward
            // call. `BackwardContext` holds it by reference and assignment to BackwardContext::input_grad() stores the
            // gradients there. It initially holds null-body arrays.
            std::vector<Array> next_grads_subset;
            next_grads_subset.resize(backward_entry.next_node_count());

            // Call backward.
            BackwardContext bctx{
                    op_node, prev_nodes, graph_ids_to_stop_gradient, next_grads_subset, double_backprop_ == DoubleBackpropOption::kEnable};
            backward_entry.backward_func()(bctx);

            for (auto it = next_grads_subset.begin(); it != next_grads_subset.end(); ++it) {
                // TODO(sonots): Allow backward without setting input grads
                assert(it->body() != nullptr);
                // Make a view if the next gradient is identical to one of other prev or next gradients.
                // TODO(niboshi): Check node identity instead of body identity.
                if (std::any_of(
                            prev_nodes.begin(),
                            prev_nodes.end(),
                            [it](const ArrayNode& prev_node) {
                                return prev_node.grad().has_value() && it->body() == prev_node.grad()->body();
                            }) ||
                    std::any_of(next_grads_subset.begin(), it, [it](const Array& next_grad) { return next_grad.body() == it->body(); })) {
                    // TODO(niboshi): Copy is needed to make new nodes. Come up with a solution to avoid copy.
                    *it = it->Copy();
                }
            }

            // Accumulate grads from `next_grads_subset`.
            for (size_t i = 0; i < backward_entry.next_node_count(); ++i) {
                size_t i_next_grad = backward_entry.next_node_indices()[i];
                nonstd::optional<Array>& target_grad = next_grads[i_next_grad];
                if (target_grad.has_value()) {
                    target_grad = *target_grad + next_grads_subset[i];
                } else {
                    target_grad = std::move(next_grads_subset[i]);
                }
            }
        }

        // If previous array nodes are output nodes of backward, clear their gradients
        for (ArrayNode& prev_node : prev_nodes) {
            if (std::find_if(
                        output_array_nodes_.begin(), output_array_nodes_.end(), [&prev_node](const std::shared_ptr<ArrayNode>& out_node) {
                            return &prev_node == out_node.get();
                        }) == output_array_nodes_.end()) {
                prev_node.ClearGrad();
            }
        }

        // Erase processed OpNode from the map
        previous_array_node_keeper_.erase(&op_node);

        return next_grads;
    }

    void AccumulateNextGradients(const OpNode& op_node, std::vector<nonstd::optional<Array>> gxs) {
        gsl::span<const std::shared_ptr<ArrayNode>> next_array_nodes = op_node.next_nodes();
        assert(next_array_nodes.size() == gxs.size());
        for (size_t i = 0; i < next_array_nodes.size(); ++i) {
            nonstd::optional<Array>& gx = gxs[i];
            if (gx.has_value()) {
                next_array_nodes[i]->AccumulateGrad(std::move(*gx));
            }
        }
    }

    void PushNextOpNode(const std::shared_ptr<ArrayNode>& array_node) {
        // When double backprop is enabled, array_node releases the pointer to the next node here. After this operation, array_node will
        // look like a leaf node of the graph. Note that this move does not invalidates the array_node object itself; it is guaranteed by
        // the standard that shared_ptr becomes null after move-assigned to another.
        std::shared_ptr<OpNode> next_op_node =
                double_backprop_ == DoubleBackpropOption::kEnable ? array_node->next_node() : array_node->move_next_node();

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
