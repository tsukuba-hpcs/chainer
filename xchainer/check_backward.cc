#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backward.h"
#include "xchainer/error.h"
#include "xchainer/numeric.h"
#include "xchainer/numerical_gradient.h"

namespace xchainer {
namespace {

std::vector<nonstd::optional<Array>> BackwardGradients(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        std::vector<Array>& inputs,
        const nonstd::optional<std::vector<Array>>& grad_outputs,
        const GraphId& graph_id,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kEnable) {
    for (const auto& input : inputs) {
        if (internal::HasArrayNode(input, graph_id) && internal::GetArrayNode(input, graph_id)->next_node()) {
            throw GradientCheckError{"BackwardGradients: All inputs must be leaf nodes of computational graph"};
        }
    }

    std::vector<Array> outputs = func(inputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < outputs.size(); ++j) {
            if (inputs[i].body() == outputs[j].body() && inputs[i].IsGradRequired(graph_id)) {
                throw GradientCheckError{"BackwardGradients: Input ", i, " and output ", j, " of the forward function are identical."};
            }
        }
    }

    if (grad_outputs) {
        const std::size_t nout = outputs.size();
        if (nout != grad_outputs->size()) {
            throw GradientCheckError{"BackwardGradients: Size of function outputs: ",
                                     nout,
                                     " and size of grad outputs: ",
                                     grad_outputs->size(),
                                     " must be same"};
        }

        for (std::size_t i = 0; i < nout; ++i) {
            if (outputs[i].IsGradRequired(graph_id)) {
                outputs[i].SetGrad((*grad_outputs)[i], graph_id);
            }
        }
    }

    // Clear gradients which may exist if func calls backward inside of itself.
    for (Array& input : inputs) {
        if (input.IsGradRequired(graph_id)) {
            input.ClearGrad(graph_id);
        }
    }

    std::vector<ConstArrayRef> outputs_ref{outputs.begin(), outputs.end()};
    std::vector<ConstArrayRef> outputs_requiring_grad;
    std::copy_if(outputs_ref.begin(), outputs_ref.end(), std::back_inserter(outputs_requiring_grad), [&graph_id](const Array& a) {
        return a.IsGradRequired(graph_id);
    });
    Backward(outputs_requiring_grad, graph_id, double_backprop);

    std::vector<nonstd::optional<Array>> backward_grads;
    std::transform(
            inputs.begin(), inputs.end(), std::back_inserter(backward_grads), [&graph_id](const Array& input) -> nonstd::optional<Array> {
                if (!input.IsGradRequired(graph_id)) {
                    return nonstd::nullopt;
                }
                return input.GetGrad(graph_id);
            });

    return backward_grads;
}

void CheckDoubleBackpropOption(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const GraphId& graph_id) {
    std::ostringstream failure_os;

    // make it nonlinear to be double differentiable so that this utility can be used even for non double differentiable functions
    auto nonlinear_func = [&func](const std::vector<Array>& inputs) {
        std::vector<Array> nonlinear_outputs;
        for (const auto& output : func(inputs)) {
            nonlinear_outputs.emplace_back(output * output);
        }
        return nonlinear_outputs;
    };

    // Disable double backprop
    {
        std::vector<Array> inputs_copy{inputs};
        std::vector<nonstd::optional<Array>> grads =
                BackwardGradients(nonlinear_func, inputs_copy, nonstd::nullopt, graph_id, DoubleBackpropOption::kDisable);

        for (size_t i = 0; i < grads.size(); ++i) {
            if (grads[i]) {
                if (grads[i]->IsGradRequired(graph_id)) {
                    failure_os << "Gradient " << i << " / " << grads.size() << " is connected to the graph '" << graph_id
                               << "' even when double-backprop is disabled.";
                }
            }
        }
    }

    // Enable double backprop
    {
        std::vector<Array> inputs_copy{inputs};
        std::vector<nonstd::optional<Array>> grads =
                BackwardGradients(nonlinear_func, inputs_copy, nonstd::nullopt, graph_id, DoubleBackpropOption::kEnable);

        for (size_t i = 0; i < grads.size(); ++i) {
            if (grads[i]) {
                if (!grads[i]->IsGradRequired(graph_id)) {
                    failure_os << "Gradient " << i << " / " << grads.size() << " is not connected to the graph '" << graph_id
                               << "' even when double-backprop is enabled.";
                }
            }
        }
    }

    // Do nothing unless failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        throw GradientCheckError{failure_message};
    }
}

void CheckBackwardComputation(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const GraphId& graph_id) {
    std::vector<Array> inputs_copy{inputs};

    // Compute backward gradients
    const std::vector<nonstd::optional<Array>> backward_grads = BackwardGradients(func, inputs_copy, grad_outputs, graph_id);
    if (backward_grads.size() != inputs.size()) {
        throw GradientCheckError{"Number of input gradients does not match the input arrays."};
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!backward_grads[i].has_value()) {
            continue;
        }
        const Array& backward_grad = *backward_grads[i];
        if (backward_grad.shape() != inputs[i].shape()) {
            throw GradientCheckError{"Shape of input gradient ",
                                     i,
                                     " of ",
                                     inputs.size(),
                                     " ",
                                     backward_grad.shape(),
                                     " does not match the corresponding input shape ",
                                     inputs[i].shape(),
                                     "."};
        }
        if (backward_grad.dtype() != inputs[i].dtype()) {
            throw GradientCheckError{"Dtype of input gradient ",
                                     i,
                                     " of ",
                                     inputs.size(),
                                     " ",
                                     GetDtypeName(backward_grad.dtype()),
                                     " does not match the corresponding input dtype ",
                                     GetDtypeName(inputs[i].dtype()),
                                     "."};
        }
    }

    // Compute numerical gradients
    const std::vector<Array> numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps, graph_id);

    // If you're trapped in any of these asserts, numerical gradiends must be implemented incorrectly.
    assert(numerical_grads.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        assert(numerical_grads[i].shape() == inputs[i].shape());
        assert(numerical_grads[i].dtype() == inputs[i].dtype());
    }

    // Check consistency between backward gradients and numeric gradients.
    std::ostringstream failure_os;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!backward_grads[i].has_value()) {
            continue;
        }

        const Array& backward_grad = *backward_grads[i];
        const Array& numerical_grad = numerical_grads[i];
        if (!AllClose(backward_grad, numerical_grad, atol, rtol)) {
            failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                       << "Graph name: " << graph_id << "\n"
                       << "Atol: " << atol << "\n"
                       << "Rtol: " << rtol << "\n"
                       << "Eps (perturbation):\n"
                       << eps[i] << "\n"
                       << "Error:\n"
                       << backward_grad - numerical_grad << "\n"  // TODO(niboshi): Use abs
                       << "Backward gradients:\n"
                       << backward_grad << "\n"
                       << "Numerical gradients:\n"
                       << numerical_grad;
        }
    }

    // Do nothing if all backward-numerical gradient pairs were close, else generate a nonfatal failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        throw GradientCheckError{failure_message};
    }
}

}  // namespace

void CheckBackward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const GraphId& graph_id) {
    CheckDoubleBackpropOption(func, inputs, graph_id);
    CheckBackwardComputation(func, inputs, grad_outputs, eps, atol, rtol, graph_id);
}

void CheckDoubleBackwardComputation(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& grad_grad_inputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const GraphId& graph_id) {
    // LIMITATION: All inputs must require gradients unlike CheckBackwardComputation
    for (const auto& input : inputs) {
        if (!input.IsGradRequired(graph_id)) {
            throw GradientCheckError{"All inputs must require gradients"};
        }
    }

    const std::size_t nin = inputs.size();
    const std::size_t n_grad_outputs = grad_outputs.size();

    // Just merge inputs and grad_outputs into inputs_and_grad_outputs to pass into below `first_order_grad_func`.
    // Since move assignment operator of Array is deleted, we can not use std::vector::insert. Instead use reserve, and std::copy
    std::vector<Array> inputs_and_grad_outputs;
    inputs_and_grad_outputs.reserve(nin + n_grad_outputs);
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_and_grad_outputs));
    std::copy(grad_outputs.begin(), grad_outputs.end(), std::back_inserter(inputs_and_grad_outputs));

    auto first_order_grad_func = [&func, nin, n_grad_outputs, &graph_id](const std::vector<Array>& inputs_and_grad_outputs) {
        // Just revert (split) inputs_and_grad_outputs into inputs and grad_outputs
        std::vector<Array> inputs{inputs_and_grad_outputs.begin(), inputs_and_grad_outputs.begin() + nin};
        std::vector<Array> grad_outputs{inputs_and_grad_outputs.begin() + nin, inputs_and_grad_outputs.end()};

        std::vector<nonstd::optional<Array>> optional_backward_grads = BackwardGradients(func, inputs, grad_outputs, graph_id);

        // Just convert std::vector<nonstd::optional<Array>> to std::vector<Array> so that CalculateNumericalGradient can accept
        std::vector<Array> backward_grads;
        std::transform(
                optional_backward_grads.begin(),
                optional_backward_grads.end(),
                std::back_inserter(backward_grads),
                [](const nonstd::optional<Array>& optional_backward_grad) {
                    if (!optional_backward_grad.has_value()) {
                        throw GradientCheckError{"All gradients must exist"};
                    }
                    return *optional_backward_grad;
                });
        return backward_grads;
    };

    const std::vector<Array> numerical_grads =
            CalculateNumericalGradient(first_order_grad_func, inputs_and_grad_outputs, grad_grad_inputs, eps, graph_id);
    const std::vector<nonstd::optional<Array>> backward_grads =
            BackwardGradients(first_order_grad_func, inputs_and_grad_outputs, grad_grad_inputs, graph_id);
    if (backward_grads.size() != numerical_grads.size()) {
        throw GradientCheckError{"Number of gradient arrays mismatched between backprop and numerical grad"};
    }

    std::ostringstream failure_os;
    for (size_t i = 0; i < backward_grads.size(); ++i) {
        if (!backward_grads[i].has_value()) {
            failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                       << "Graph name: " << graph_id << "\n"
                       << "Missing gradients. Maybe the given function was not twice differentiable.";
        } else if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
            failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                       << "Graph name: " << graph_id << "\n"
                       << "Atol: " << atol << "\n"
                       << "Rtol: " << rtol << "\n"
                       << "Eps (perturbation):\n"
                       << eps[i] << "\n"
                       << "Backward gradients:\n"
                       << *backward_grads[i] << "\n"
                       << "Numerical gradients:\n"
                       << numerical_grads[i];
        }
    }

    // Do nothing if all backward-numerical gradient pairs were close, else generate a nonfatal failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        throw GradientCheckError{failure_message};
    }
}

}  // namespace xchainer
