#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array_node.h"

namespace xchainer {

class OpNode {
public:
    OpNode() = default;
    OpNode(std::string name, int64_t rank, std::vector<std::shared_ptr<ArrayNode>> next_nodes,
           std::vector<std::function<Array(const Array&)>> backward_functions)
        : name_(std::move(name)), rank_(rank), next_nodes_(std::move(next_nodes)), backward_functions_(std::move(backward_functions)) {}

    gsl::span<const std::shared_ptr<ArrayNode>> next_nodes() const { return gsl::make_span(next_nodes_); }
    gsl::span<const std::function<Array(const Array&)>> backward_functions() const { return gsl::make_span(backward_functions_); }

    std::string name() const { return name_; }

    int64_t rank() const { return rank_; }

private:
    std::string name_;
    int64_t rank_;
    std::vector<std::shared_ptr<ArrayNode>> next_nodes_;
    std::vector<std::function<Array(const Array&)>> backward_functions_;
};

}  // namespace xchainer
