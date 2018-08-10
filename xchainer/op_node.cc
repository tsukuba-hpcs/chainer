#include "xchainer/op_node.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

ArrayProps::ArrayProps(const Array& array) : shape{array.shape()}, dtype{array.dtype()}, device{array.device()} {}
ArrayProps::ArrayProps(const ArrayNode& array_node) : shape{array_node.shape()}, dtype{array_node.dtype()}, device{array_node.device()} {}

OpNodeBackwardEntry::OpNodeBackwardEntry(OpNode& op_node, std::vector<size_t> input_array_node_indices, BackwardFunction backward_func)
    : op_node_{op_node}, input_array_node_indices_{std::move(input_array_node_indices)}, backward_func_{std::move(backward_func)} {}

void OpNodeBackwardEntry::AddExoticInputArrayNode(std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>> input_array_nodes) {
    assert(std::get<0>(input_array_nodes) != op_node_.backprop_id());
    exotic_input_array_nodes_.emplace_back(std::move(input_array_nodes));
}

std::shared_ptr<ArrayNode> FabricateOutputArrayNode(std::shared_ptr<OpNode> op_node, size_t output_array_node_index) {
    assert(output_array_node_index < op_node->output_array_node_count());
    assert(op_node->output_array_nodes()[output_array_node_index].expired());

    const ArrayProps& props = op_node->GetOutputArrayProps(output_array_node_index);
    auto output_array_node = std::make_shared<ArrayNode>(props.shape, props.dtype, props.device, op_node->backprop_id());

    op_node->output_array_nodes()[output_array_node_index] = output_array_node;
    output_array_node->set_creator_op_node(std::move(op_node));

    return output_array_node;
}

// static
std::shared_ptr<OpNode> OpNode::CreateWithOutputArrayNodes(
        std::string name, BackpropId backprop_id, size_t input_count, const std::vector<ConstArrayRef>& outputs) {
    // Trick to use make_shared with private ctor
    struct OpNodeWithPublicCtor : OpNode {
        OpNodeWithPublicCtor(std::string name, BackpropId backprop_id, size_t input_count)
            : OpNode{std::move(name), backprop_id, input_count} {}
    };
    std::shared_ptr<OpNode> op_node = std::make_shared<OpNodeWithPublicCtor>(std::move(name), backprop_id, input_count);

    for (const Array& out : outputs) {
        const std::shared_ptr<ArrayBody>& out_body = GetArrayBody(out);
        assert(!out_body->HasArrayNode(backprop_id));
        const std::shared_ptr<ArrayNode>& output_array_node = ArrayBody::CreateArrayNode(out_body, backprop_id);
        op_node->output_array_props_.emplace_back(*output_array_node);
        op_node->output_array_nodes_.emplace_back(output_array_node);
        output_array_node->set_creator_op_node(op_node);
    }
    op_node->AssertConsistency();
    return op_node;
}

OpNode::OpNode(std::string name, BackpropId backprop_id, size_t input_array_node_count)
    : name_{std::move(name)},
      backprop_id_{backprop_id},
      input_array_nodes_{input_array_node_count}  // Initialize with nullptrs
{}

#ifndef NDEBUG

namespace {

bool IsAllArrayNodesMatchBackpropId(
        const BackpropId& outer_backprop_id, const std::vector<std::shared_ptr<ArrayNode>>& outer_graphs_array_nodes) {
    return std::all_of(
            outer_graphs_array_nodes.begin(),
            outer_graphs_array_nodes.end(),
            [&outer_backprop_id](const std::shared_ptr<ArrayNode>& array_node) {
                return array_node == nullptr || array_node->backprop_id() == outer_backprop_id;
            });
}

bool AssertOuterGraphsArrayNodesConsistency(
        const BackpropId& backprop_id,
        size_t array_node_count,
        const std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>>& outer_graphs_array_nodes,
        bool is_output) {
    assert(array_node_count > 0);

    // No pair of entries may have the same backprop ID.
    assert(std::all_of(outer_graphs_array_nodes.begin(), outer_graphs_array_nodes.end(), [&outer_graphs_array_nodes](const auto& tup1) {
        return std::all_of(outer_graphs_array_nodes.begin(), outer_graphs_array_nodes.end(), [&tup1](const auto& tup2) {
            BackpropId backprop_id1 = std::get<0>(tup1);
            BackpropId backprop_id2 = std::get<0>(tup2);
            return &tup1 == &tup2 || backprop_id1 != backprop_id2;
        });
    }));

    // All the outer graphs linked from this op node must be outer (lower backprop ordinal).
    assert(std::all_of(outer_graphs_array_nodes.begin(), outer_graphs_array_nodes.end(), [&backprop_id](const auto& tup) {
        BackpropId outer_backprop_id = std::get<0>(tup);
        return outer_backprop_id < backprop_id;
    }));

    // Corresponding array nodes across graphs (corresponding to the same input/output array) should have the same array body, if it's
    // alive.
    for (size_t i = 0; i < array_node_count; ++i) {
        nonstd::optional<ArrayBody*> array_body{};
        for (const auto& tup : outer_graphs_array_nodes) {
            const std::vector<std::shared_ptr<ArrayNode>>& vec = std::get<1>(tup);
            assert(vec.size() == array_node_count);
            const std::shared_ptr<ArrayNode>& array_node = vec.at(i);
            if (is_output) {
                // If the output is retained, array nodes of the output for all the outer graphs are not null.
                // Otherwise, they are all null.
                assert((array_node == nullptr) == (std::get<1>(outer_graphs_array_nodes.front()).at(i) == nullptr));
            }
            if (array_node == nullptr) {
                // Outer graph references can be null for array nodes for arrays that are not retained.
                continue;
            }
            std::shared_ptr<ArrayBody> body = array_node->weak_body().lock();
            if (!array_body.has_value()) {
                array_body = body.get();
            } else {
                assert(*array_body == body.get());
            }
        }
    }
}

}  // namespace

void OpNode::AssertConsistency() const {
    AssertOuterGraphsArrayNodesConsistency(backprop_id_, input_array_node_count(), outer_graphs_input_array_nodes_, false);
    AssertOuterGraphsArrayNodesConsistency(backprop_id_, output_array_node_count(), outer_graphs_output_array_nodes_, true);
}

#else

void OpNode::AssertConsistency() const {}

#endif  // NDEBUG

std::vector<std::shared_ptr<ArrayNode>>& OpNode::input_array_nodes() {
    assert(std::all_of(input_array_nodes_.begin(), input_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node == nullptr || arr_node->backprop_id() == backprop_id_;
    }));
    return input_array_nodes_;
}

const std::vector<std::shared_ptr<ArrayNode>>& OpNode::input_array_nodes() const {
    assert(std::all_of(input_array_nodes_.begin(), input_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node == nullptr || arr_node->backprop_id() == backprop_id_;
    }));
    return input_array_nodes_;
}

OpNodeBackwardEntry& OpNode::RegisterBackwardFunction(
        std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> input_array_nodes, BackwardFunction backward_func) {
    AssertConsistency();
    assert(!input_array_nodes.empty());
    assert(std::all_of(input_array_nodes.begin(), input_array_nodes.end(), [this](const auto& tup) {
        const std::shared_ptr<ArrayNode>& input_array_node = std::get<1>(tup);
        // input_array_node could be nullptr, if the corresponding input array does not require grad.
        return input_array_node == nullptr || input_array_node->backprop_id() == backprop_id_;
    }));

    // Update the rank of op node
    for (const auto& tup : input_array_nodes) {
        const std::shared_ptr<ArrayNode>& input_array_node = std::get<1>(tup);
        if (input_array_node != nullptr) {
            rank_ = std::max(rank_, input_array_node->rank() + 1);
        }
    }

    // Store input nodes and record indices of them
    std::vector<size_t> input_array_node_indices;
    input_array_node_indices.reserve(input_array_nodes.size());
    for (auto& tup : input_array_nodes) {
        size_t input_index = std::get<0>(tup);
        std::shared_ptr<ArrayNode>& input_array_node = std::get<1>(tup);

        input_array_node_indices.emplace_back(input_index);
        if (input_array_node != nullptr) {
            assert(gsl::at(input_array_nodes_, input_index) == nullptr);
            gsl::at(input_array_nodes_, input_index) = std::move(input_array_node);
        }
    }

    // Add backward entry
    backward_entries_.emplace_back(*this, std::move(input_array_node_indices), std::move(backward_func));

    AssertConsistency();
    return backward_entries_.back();
}

void OpNode::AddEdgesToInputArrayNodesOfOuterGraph(
        const BackpropId& outer_backprop_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_input_array_nodes) {
    AssertConsistency();
    assert(outer_backprop_id < backprop_id_);
    assert(outer_graphs_input_array_nodes.size() == input_array_nodes_.size());
    assert(IsAllArrayNodesMatchBackpropId(outer_backprop_id, outer_graphs_input_array_nodes));

    outer_graphs_input_array_nodes_.emplace_back(outer_backprop_id, std::move(outer_graphs_input_array_nodes));

    AssertConsistency();
}

void OpNode::AddEdgesToOutputArrayNodesOfOuterGraph(
        const BackpropId& outer_backprop_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_output_array_nodes) {
    AssertConsistency();
    assert(outer_backprop_id < backprop_id_);
    assert(outer_graphs_output_array_nodes.size() == output_array_props_.size());
    assert(IsAllArrayNodesMatchBackpropId(outer_backprop_id, outer_graphs_output_array_nodes));

    outer_graphs_output_array_nodes_.emplace_back(outer_backprop_id, std::move(outer_graphs_output_array_nodes));

    AssertConsistency();
}

}  // namespace internal
}  // namespace xchainer
