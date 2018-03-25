import contextlib

import chainer

from scipy import stats # fixme: remove after debug
import numpy as np # fixme: remove after debug

# These function are intended to by called from chainer.FunctionNode and
# chainer.Variable. They should not be directly called from user code.


def is_static_func(func):
    """Check if the function node is included in a static schedule.

    Returns:
        bool: True if the supplied function is included in a static
            schedule. Otherwise, return False.
    """
    return hasattr(func, 'schedule_func')


def get_static_schedule(func):
    """Get the forward static schedule that contains the supplied function node.

    If the supplied function node is contained in a static schedule, return
    the static schedule. Otherwise, return ``None``. Note in order for
    ``func`` to be contained in a static schedule, ``func`` must have already
    been called in the forward pass from a ``@static_graph``-decorated
    chain.

    Args:
        func (FunctionNode): The supplied function node.

    Returns:
        StaticScheduleFunction or None: Depending on whether or not the
        supplied function is contained in a static schedule.
    """
    return getattr(func, 'schedule_func', None)


def is_trace_mode():
    """Check if trace mode is on.

    If this function is called by the define-by-run code of a @static_graph
    decorated ``__call__()`` of a chain, return True.

    Returns:
        bool: True if trace mode is on. Otherwise, return False.
    """
    return chainer.config.schedule_func is not None


def mark_static_vars(input_vars):
    """Mark variables as static if inside a static chain.

    If trace mode is currently on, set the ``is_static`` attribute of
    each variable in ``input_vars`` to True.
    This is needed since cleargrads() sets the `grad` members of
    the parameters to `None`. Since the static subgraph optimizations
    currently assumes that all parameters in a static chain are allocated
    statically, setting the `grad` members to `None` would break the
    functionality and so this function is used to mark parameters in
    a static chain as being "static" and modifying `cleargrads()` to
    check only set non-static parmaeters' `grad` members to `None`.

    Args:
        input_vars (list of variable): The supplied list of variables
            (including parameters).

    """
    if is_trace_mode():
        for var in input_vars:
            # todo: consider only marking a variable if it is a parameter.
            var.is_static = True


def static_schedule_func(*dec_args, **dec_kwargs):
    """Decorator to mark a function for inclusion in the forward schedule.

    This decorator is used to wrap a function `func` that is a forward-pass
    method of a sub-class of Function. This will cause it to be added to
    the forward static schedule when the `static_graph` feature is
    enabled on a Chain that deeply calls it from the chain's
    `__call__()` method.

    The function to be wrapped should only return `None` because any return value
    will be ignored. Instead of returning its results, any result arrays must
    be supplied as input arguments and must have already have been initialized
    to the appropriate dimensions and data types.

    Usage:

    Typical usage is to allocate any required arrays (Numpy or CuPy) in Python
    code in an instance of FunctionNode (See `LinearFunction` function for an example).
    Generally, this will involve first allocating storage for the results arrays
    in the `forward()` method of a sub-class of Function. Then, the
    FunctionNode.foward()
     method should call another
    (private) method that is wrapped using this decorator. The
    decorated function will take all required input and output arrays as
    arguments and will not return anything (that is, `None` will be implicitly
    returned).

    Note that by following this usage convention, all input and output activations,
    along with any parameter arrays will have been statically allocated by the
    end of the first forward pass. Since the the forward-pass functions that
    are used inside the forward static schedule (that is, the functions that
    use this decorator) do not allocate any results arrays, this results in code that
    looks like 'define-by-run' to the user, and which can be debugged during
    the first iteration, but then becomes static in terms of memory allocations and
    scheduling starting from the second iteration. Thus, we get the benefit of
    both ease of use and optimized performance.

    It is important that all of the required computations that occur during the
    second  and later forward passes must be contained inside the functions
    that use this decorator. That is, any other code (that is not wrapped inside this
    decorator) in the various FunctionNode and Link instances can be viewed as
    setup code that only checks types, allocates result arrays, initializes
    parameters etc., but does not perform any computations that must
    be repeated after the first forward pass.

    The reason for this is that after the first iteration (that is, starting
    from the second forward pass), when the chain's `__call__()` is called,
    the forward static schedule will be invoked and it will only call the
    functions that were wrapped with this decorator. Note that this can potentially
    lead to difficult to find bugs if one forgets to decorate a required function,
    since the corresponding computations would no longer execute after the
    first iteration. As a general rule, any code that is intended to exectue on
    each iteration should be called by a function that uses this decorator.

    Args:
        delay_call (bool): Optional keyword argument. If True, don't call the wrapped
        function during the define by run pass, but only add it to the static schedule.
        Default value is False.

        func: A forward-pass method of a sub-class of FunctionNode that will be inserted
            into the static forward schedule when `static_graph` is enabled. The function
            must not return anything because any return values will be ignored.

    Returns: The wrapped function.

    """
    delay_call = False
    zero_args = False
    if len(dec_args) == 1 and not dec_kwargs and callable(dec_args[0]):
        callable_arg = dec_args[0]
        zero_args = True
    elif dec_kwargs:
        if 'delay_call' in dec_kwargs:
            delay_call = dec_kwargs['delay_call']

    def debug_print_stats(args):
        for arg in args:
            if isinstance(arg, np.ndarray):
                print('no-arg-func stats: ', stats.describe(arg, axis=None))
                print('id of array: ', id(arg))
            elif isinstance(arg, (list, tuple)):
                debug_print_stats(arg)
            else:
                print('no-arg-func name: ', str(arg))


    def wrap(func):
        def wrapped_func(*args, **kwargs):
            # Save arguments, function, and results pointers/references to the schedule list:
            def no_arg_func():
                #print('In no_arg_func: Calling: ', func)
                #debug_print_stats(args)
                ret = func(*args, **kwargs)
                if ret is not None:
                    raise RuntimeError("This function is not supposed to return anything: ", func)
                # print("Arguments were: %s, %s" % (args, kwargs))

            # no_arg_func() requires no arguments to call since the arguments of the decorated function
            # are captured by the closure.
            if not delay_call:
                no_arg_func()

            schedule_function = chainer.config.schedule_func
            # If trace mode is on, add to schedule.
            if schedule_function is not None:
                schedule_function.append_function(no_arg_func)
                # Add the schedule function as an attribute of the FunctionNode instance
                # that contains the wrapped function as a method
                # This attribute will be needed by the corresponding @static_backward
                # function.
                instance = args[0]
                # assert isinstance(instance, chainer.function_node.FunctionNode)
                instance._supports_static_optimizations = True
                # print('static_forward: instance: ', instance)
                instance.schedule_func = schedule_function

        return wrapped_func

    if zero_args:
        return wrap(callable_arg)
    else:
        return wrap


def static_forward_optimizations(func, in_vars, in_data, outputs):
    """Perform checks needed for creation of a static schedule.

    For each variable ``x`` in ``in_vars``, check if ``x`` is an
    input variable to a static chain. If so, then save the
    information to the function so that it can be used during the
    backward pass schedule creation.

    Also check if `func` supports static graph optimizations. If not, try
    to automatically wrap it to be compatible.

    This function should be called from the ``FunctionNode`` apply() method
    just after func.forward() is called.

    Args:
        func (instance of FunctionNode):
        in_vars (tuple of variable): input variables to func.apply()
        in_data (tuple of ndarray): input arrays to func
        outputs (tuple of ndarray): outputs of func.

    """

    schedule_function = chainer.config.schedule_func
    if schedule_function is not None:
        if not func._supports_static_optimizations:
            if schedule_function.verbosity_level >= 2:
                print("Adding automatic static graph support to function: ",
                    func)
            # func does not support static optimizations, so let's try to wrap it
            # automatically.
            @static_schedule_func(delay_call=True)
            def generic_static_forward(func, in_data, out_data):
                """
                    fixme
                in_arrs: tuple of input arrays
                out_arrs: tuple of output arrays
                func: compatible with out_arrs = func(in_arrs)
                """
                temp_out_data = func.forward(in_data)
                for ind, static_ar in enumerate(out_data):
                    static_ar[...] = temp_out_data[ind]

            generic_static_forward(func, in_data, outputs)




