#pragma once

#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/op_node.h"

namespace xchainer {

class ArrayNode {
public:
    ArrayNode() = default;
    ArrayNode(GraphId graph_id) : next_node_(), rank_(0), grad_(), graph_id_(graph_id) {}

    const std::shared_ptr<OpNode>& next_node() { return next_node_; }
    std::shared_ptr<const OpNode> next_node() const { return next_node_; }
    std::shared_ptr<OpNode> move_next_node() { return std::move(next_node_); }

    void set_next_node(std::shared_ptr<OpNode> next_node) { next_node_ = std::move(next_node); }

    int64_t rank() const { return rank_; }

    void set_rank(int64_t rank) { rank_ = rank; }

    const nonstd::optional<Array>& grad() const noexcept { return grad_; }

    void set_grad(Array grad) { grad_.emplace(std::move(grad)); }

    GraphId graph_id() const { return graph_id_; }

    void set_graph_id(GraphId graph_id) { graph_id_ = std::move(graph_id); }

    void ClearGrad() noexcept { grad_.reset(); }

private:
    std::shared_ptr<OpNode> next_node_;
    int64_t rank_{0};
    nonstd::optional<Array> grad_;
    GraphId graph_id_;
};

}  // namespace xchainer
