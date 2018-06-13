#pragma once

#include <functional>
#include <initializer_list>
#include <unordered_map>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"

namespace xchainer {

enum class DoubleBackpropOption : bool {
    kDisable = false,
    kEnable = true,
};

class BackwardContext;

namespace backward_detail {

using BackwardFunc = std::function<void(BackwardContext&)>;

}  // namespace backward_detail

class OpNode;

namespace internal {
class ArrayBody;
}  // namespace internal

class BackwardContext {
public:
    // Ctor
    //
    // `input_grads_storage` is where input gradients returned by backward functions will be stored.
    // Its size must be equal to the number of input arrays whose gradients are to be returned in this single backward function (1 in most
    // ordinary functions).
    BackwardContext(
            const OpNode& op_node,
            gsl::span<const std::reference_wrapper<ArrayNode>> prev_nodes,
            gsl::span<const GraphId> stop_graph_ids,
            std::vector<Array>& input_grads_storage);

    // Returns whether the output has a propagated gradient.
    // If there is only one output, the output always has the propagated gradient, therefore you do not have to call this function in that
    // case.
    bool HasOutputGrad(int output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad(int output_index) const { return GetOutputGrad(output_index); }

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& output_grad() const {
        assert(prev_nodes_.size() == 1);
        return GetOutputGrad(0);
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
    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& GetOutputGrad(int output_index) const;

    // Returns the reference to an output gradient array if it has a propagated value.
    // Otherwise, an zero-filled array is allocated and a reference to it is returned.
    const Array& GetOutputGrad() const {
        assert(prev_nodes_.size() == 1);
        return GetOutputGrad(0);
    }

    const OpNode& op_node_;
    gsl::span<const std::reference_wrapper<ArrayNode>> prev_nodes_;
    gsl::span<const GraphId> stop_graph_ids_;

    // A reference to the storage of input gradient arrays.
    // Gradient passed in input_grad() will be put into this storage.
    // Unset gradients will have null array body.
    std::vector<Array>& input_grads_storage_;

    // Holds zero-filled arrays for outputs without actual gradients.
    // The arrays are allocated on-demand in GetOutputGrad.
    mutable std::vector<nonstd::optional<Array>> zero_output_grads_;
};

class BackwardBuilder {
public:
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs, gsl::span<const GraphId> stop_graph_ids);
    BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs) : BackwardBuilder{op_name, outputs, {}} {}
    BackwardBuilder(const char* op_name, const Array& output, gsl::span<const GraphId> stop_graph_ids)
        : BackwardBuilder{op_name, {output}, stop_graph_ids} {};
    BackwardBuilder(const char* op_name, const Array& output) : BackwardBuilder{op_name, {output}, {}} {};

    // Defines a backward function with respect to specified input arrays.
    // For multi-input ops, usually this function is called for each of independent subsets of input arrays.
    template <typename BackwardFunc>
    void Define(std::initializer_list<ConstArrayRef> inputs, BackwardFunc&& backward_func) {
        static_assert(
                std::is_same<std::result_of_t<BackwardFunc(BackwardContext&)>, void>::value,
                "The result type of backward functions must be void.");

        DefineImpl(inputs, std::forward<BackwardFunc>(backward_func));
    }

private:
    // Defines a backward function with respect to specified input arrays.
    // For multi-input ops, usually this function is called for each of independent subsets of input arrays.
    void DefineImpl(std::initializer_list<ConstArrayRef> inputs, backward_detail::BackwardFunc&& backward_func);

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

}  // namespace xchainer
