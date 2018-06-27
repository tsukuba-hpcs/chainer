#pragma once

#include <functional>
#include <initializer_list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"

namespace xchainer {

class BackwardContext;
class OpNode;

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

using BackwardFunction = std::function<void(BackwardContext&)>;

class BackwardContext {
public:
    // Ctor
    //
    // `input_grads_storage` is where input gradients returned by backward functions will be stored.
    // Its size must be equal to the number of input arrays whose gradients are to be returned in this single backward function (1 in most
    // ordinary functions).
    BackwardContext(
            const OpNode& op_node,
            gsl::span<const std::reference_wrapper<ArrayNode>> prev_array_nodes,
            gsl::span<const GraphId> stop_graph_ids,
            std::vector<Array>& input_grads_storage,
            bool next_backward_required);

    // Indicates whether the next order of backward is required. It reflects DoubleBackpropOption.
    bool next_required() const { return next_backward_required_; }

    // Returns whether the output has a propagated gradient.
    // If there is only one output, the output always has the propagated gradient, therefore you do not have to call this function in that
    // case.
    bool HasOutputGrad(int output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad(int output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad() const {
        assert(prev_array_nodes_.size() == 1);
        return output_grad(0);
    }

    // Returns the reference to the input gradient.
    Array& input_grad() {
        assert(input_grads_storage_.size() == 1);
        return gsl::at(input_grads_storage_, 0);
    }

    // Returns the reference to the input gradient.
    Array& input_grad(size_t index) { return gsl::at(input_grads_storage_, index); }

    // Given an array, cuts the graphs to stop gradients and returns the resulting array.
    Array Cut(const Array& a) const;

private:
    const OpNode& op_node_;
    gsl::span<const std::reference_wrapper<ArrayNode>> prev_array_nodes_;
    gsl::span<const GraphId> stop_graph_ids_;

    // A reference to the storage of input gradient arrays.
    // Gradient passed in input_grad() will be put into this storage.
    // Unset gradients will have null array body.
    std::vector<Array>& input_grads_storage_;

    // Holds zero-filled arrays for outputs without actual gradients.
    // The arrays are allocated on-demand in output_grad.
    mutable std::vector<nonstd::optional<Array>> zero_output_grads_;

    bool next_backward_required_;
};

class BackwardBuilder {
public:
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs, gsl::span<const GraphId> stop_graph_ids);
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs) : BackwardBuilder{op_name, outputs, {}} {}
    BackwardBuilder(const char* op_name, const Array& output, gsl::span<const GraphId> stop_graph_ids)
        : BackwardBuilder{op_name, std::initializer_list<ConstArrayRef>{output}, stop_graph_ids} {}
    BackwardBuilder(const char* op_name, const Array& output) : BackwardBuilder{op_name, {output}, {}} {}

    // Defines a backward function with respect to specified input arrays.
    // For multi-input ops, usually this function is called for each of independent subsets of input arrays.
    void Define(std::initializer_list<ConstArrayRef> inputs, const BackwardFunction& backward_func);

private:
    const char* op_name_;

    // Output arrays of the op.
    std::vector<ConstArrayRef> outputs_;

    // A collection of op nodes, each of which corresponds to a graph.
    // This record is increasingly populated as new graphs are encountered in multiple Define() calls.
    std::unordered_map<GraphId, std::shared_ptr<OpNode>> op_node_map_;

    std::vector<GraphId> stop_graph_ids_;
};

void Backward(
        const Array& output,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

void Backward(
        const std::vector<ConstArrayRef>& outputs,
        const GraphId& graph_id = kDefaultGraphId,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kDisable);

namespace internal {

struct BackpropMode {
public:
    BackpropMode(const nonstd::optional<GraphId>& graph_id, bool backprop) : graph_id_(graph_id), backprop_(backprop) {}
    BackpropMode(const GraphId& graph_id, bool backprop) : graph_id_(graph_id), backprop_(backprop) {}

    const nonstd::optional<GraphId>& graph_id() const { return graph_id_; }
    bool backprop() const { return backprop_; }

    bool operator==(const BackpropMode& other) const { return backprop_ == other.backprop_ && graph_id_ == other.graph_id_; }
    bool operator!=(const BackpropMode& other) const { return !operator==(other); }

private:
    nonstd::optional<GraphId> graph_id_;
    bool backprop_;  // false for NoBackpropMode, and true for ForceBackpropMode
};

using BackpropModeStack = std::vector<BackpropMode>;
using BackpropModeContextStack = std::unordered_map<const Context*, BackpropModeStack>;

void SetBackpropModeContextStack(BackpropModeContextStack* backprop_mode_context_stack);
BackpropModeContextStack* GetBackpropModeContextStack();
BackpropModeStack* GetBackpropModeStack(const Context* context = GetDefaultContextNoExcept());

}  // namespace internal

// Make a context which disables back-propagation.
class NoBackpropModeScope {
public:
    // No backprop mode for all graphs
    NoBackpropModeScope() : NoBackpropModeScope(nonstd::nullopt) {}

    // No backprop mode for specified graph
    explicit NoBackpropModeScope(const GraphId& graph_id) : NoBackpropModeScope(std::move(nonstd::optional<GraphId>{graph_id})) {}

    NoBackpropModeScope(const NoBackpropModeScope&) = delete;
    NoBackpropModeScope(NoBackpropModeScope&& other) = delete;
    NoBackpropModeScope& operator=(const NoBackpropModeScope&) = delete;
    NoBackpropModeScope& operator=(NoBackpropModeScope&& other) = delete;

    ~NoBackpropModeScope() {
        assert(internal::GetBackpropModeContextStack() != nullptr);
        assert(internal::GetBackpropModeStack(context_) != nullptr);
        internal::GetBackpropModeStack(context_)->pop_back();
        // Recover thread local variable to nullptr on exiting from the outer-most scope.
        if (is_outer_most()) {
            internal::SetBackpropModeContextStack(nullptr);
        }
    }

private:
    explicit NoBackpropModeScope(nonstd::optional<GraphId> graph_id) : context_{internal::GetDefaultContextNoExcept()} {
        // The outer-most scope creates and holds an instance of BackpropModeContextStack.
        internal::BackpropModeContextStack* context_stack = internal::GetBackpropModeContextStack();
        if (context_stack == nullptr) {
            context_stack_.emplace();  // allocates
            context_stack = &context_stack_.value();
            internal::SetBackpropModeContextStack(context_stack);
        }
        (*context_stack)[context_].emplace_back(internal::BackpropMode{std::move(graph_id), false});
    }

    bool is_outer_most() { return context_stack_.has_value(); }

    Context* context_;
    nonstd::optional<internal::BackpropModeContextStack> context_stack_{};
};

// Make a context which enables back-propagation.
//
// When you want to enable back-propagation in NoBackpropModeScope, use this scope.
class ForceBackpropModeScope {
public:
    explicit ForceBackpropModeScope(const GraphId& graph_id) : context_{internal::GetDefaultContextNoExcept()} {
        internal::BackpropModeContextStack* context_stack = internal::GetBackpropModeContextStack();
        if (context_stack == nullptr) {
            throw XchainerError{"can be called only inside of no backprop mode context"};
        }
        (*context_stack)[context_].emplace_back(internal::BackpropMode{std::move(graph_id), true});
    }

    ~ForceBackpropModeScope() {
        assert(internal::GetBackpropModeContextStack() != nullptr);
        assert(internal::GetBackpropModeStack(context_) != nullptr);
        internal::GetBackpropModeStack(context_)->pop_back();
    }

    ForceBackpropModeScope(const ForceBackpropModeScope&) = delete;
    ForceBackpropModeScope(ForceBackpropModeScope&& other) = delete;
    ForceBackpropModeScope& operator=(const ForceBackpropModeScope&) = delete;
    ForceBackpropModeScope& operator=(ForceBackpropModeScope&& other) = delete;

private:
    Context* context_;
};

}  // namespace xchainer
