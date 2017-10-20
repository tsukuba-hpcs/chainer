import math
import warnings

import numpy
import six

from chainer import configuration
from chainer import cuda
from chainer.functions.math import identity
from chainer import testing
from chainer import variable


def _copy_arrays(xs):
    xp = cuda.get_array_module(*xs)
    return [xp.array(x, order='C', dtype=numpy.float64, copy=True) for x in xs]


def numerical_grad(f, inputs, grad_outputs, eps=1e-3):
    """Computes numerical gradient by finite differences.

    This function is used to implement gradient check. For usage example, see
    unit tests of :mod:`chainer.functions`.

    Args:
        f (function): Python function with no arguments that runs forward
            computation and returns the result.
        inputs (tuple of arrays): Tuple of arrays that should be treated as
            inputs. Each element of them is slightly modified to realize
            numerical gradient by finite differences.
        grad_outputs (tuple of arrays): Tuple of arrays that are treated as
            output gradients.
        eps (float): Epsilon value of finite differences.

    Returns:
        tuple: Numerical gradient arrays corresponding to ``inputs``.

    """
    assert eps > 0
    for x in inputs:
        if x.dtype.kind != 'f':
            raise RuntimeError(
                'The dtype of input arrays must be kind of float')

    inputs = tuple(inputs)
    grad_outputs = tuple(grad_outputs)
    gpu = any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs)
    cpu = any(isinstance(x, numpy.ndarray) for x in inputs + grad_outputs)

    if gpu and cpu:
        raise RuntimeError('Do not mix GPU and CPU arrays in `numerical_grad`')

    if gpu:
        xp = cuda.cupy
        numerical_grad_kernel = cuda.reduce(
            'T y1, T y2, U gy, T eps', 'V gxi',
            '(y1 - y2) * gy', 'a + b', 'gxi += a / (eps * 2)', '0',
            'numerical_grad_kernel'
        )
    else:
        xp = numpy
    grads = [xp.zeros(x.shape, numpy.float64) for x in inputs]

    with configuration.using_config('type_check', False):
        for x, gx in six.moves.zip(inputs, grads):
            orig_x = x.copy()  # hold original value
            for i in numpy.ndindex(x.shape):
                orig = orig_x[i]
                x[i] = orig + eps
                ys1 = _copy_arrays(f())
                x[i] = orig - eps
                ys2 = _copy_arrays(f())
                x[i] = orig
                for y1, y2, gy in six.moves.zip(ys1, ys2, grad_outputs):
                    if gy is not None:
                        if (gpu and isinstance(y1, cuda.ndarray) and
                                isinstance(y2, cuda.ndarray) and
                                isinstance(gy, cuda.ndarray)):
                            numerical_grad_kernel(y1, y2, gy, eps, gx[i])
                        else:
                            dot = ((y1 - y2) * gy).sum()
                            gx[i] += dot / (2 * eps)

    return [g.astype(x.dtype, copy=False)
            for g, x in six.moves.zip(grads, inputs)]


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    warnings.warn(
        'chainer.gradient_check.assert_allclose is deprecated.'
        'Use chainer.testing.assert_allclose instead.',
        DeprecationWarning)
    testing.assert_allclose(x, y, atol, rtol, verbose)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _filter_list(lst, ignore_list):
    return [x for x, ignore in six.moves.zip(lst, ignore_list) if not ignore]


def check_backward(func, x_data, y_grad, params=(),
                   eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None, dtype=None):
    """Test backward procedure of a given function.

    This function automatically checks the backward-process of a given function
    to ensure that the computed gradients are approximately correct.
    For example, assuming you've defined a :class:`~chainer.FunctionNode` class
    ``MyFunc``, that takes two arguments and returns one value, you can wrap
    it in a ordinary function and check its gradient computations as follows::

    >> def test_my_func(self):
    >>
    >>     def func(xs):
    >>         y, = MyFunc().apply(xs)
    >>         return y
    >>
    >>   x1_data = xp.array(...)
    >>   x2_data = xp.array(...)
    >>   gy_data = xp.array(...)
    >>   check_backward(func, (x1_data, x2_data), gy_data)

    This method creates :class:`~chainer.Variable` objects with ``x_data``
    and calls ``func`` with the :class:`~chainer.Variable`\\ s to get its
    result as :class:`~chainer.Variable`.
    Then, it sets ``y_grad`` array to ``grad`` attribute of the result and
    calls ``backward`` method to get gradients of the inputs.
    To check correctness of the gradients, the function calls
    :func:`numerical_grad` to calculate numerically the gradients and compares
    the types of gradients with :func:`chainer.testing.assert_allclose`.

    To reduce computational time, it uses directional derivative along a
    random vector. A function
    :math:`g: \\mathbb{R} \\rightarrow \\mathbb{R}^n` is defined as
    :math:`g(\\delta) = f(x + \\delta r)`, where
    :math:`\\delta \\in \\mathbb{R}`, :math:`r \\in \\mathbb{R}^n`
    is a random vector
    and :math:`f` is a function which you want to test.
    Its gradient is

    .. math::
       g'(\\delta) = f'(x + \\delta r) \\cdot r.

    Therefore, :math:`g'(0) = f'(x) \\cdot r`.
    So we can check the correctness of back propagation of :math:`f` indirectly
    by comparing this equation with the gradient of :math:`g` numerically
    calculated and that of :math:`f` computed by backprop.
    If :math:`r` is chosen from uniform distribution, we can conclude with
    high probability that the gradient of :math:`f` itself is correct.

    If input objects (``x1_data`` or/and ``x2_data`` in this example) represent
    integer variables, their gradients are ignored.

    You can simplify a test when ``MyFunc`` gets only one argument::

    >>   check_backward(func, x1_data, gy_data)

    If ``MyFunc`` is a loss function which returns a zero-dimensional
    array, pass ``None`` to ``gy_data``. In this case, it sets ``1`` to
    ``grad`` attribute of the result::

    >>   check_backward(my_loss_func, (x1_data, x2_data), None)

    If ``MyFunc`` returns multiple outputs, pass all gradients for outputs
    as a tuple::

    >>   gy1_data = xp.array(...)
    >>   gy2_data = xp.array(...)
    >>   check_backward(func, x1_data, (gy1_data, gy2_data))

    You can also test a :class:`~chainer.Link`.
    To check gradients of parameters of the link, set a tuple of the parameters
    to ``params`` arguments::

    >>   check_backward(my_link, (x1_data, x2_data), gy_data,
    >>                  (my_link.W, my_link.b))

    Note that ``params`` are not ``ndarray``\\ s,
    but :class:`~chainer.Variables`\\ s.

    Function objects are acceptable as ``func`` argument::

    >>   check_backward(lambda x1, x2: f(x1, x2),
    >>                  (x1_data, x2_data), gy_data)

    .. note::

       ``func`` is called many times to get numerical gradients for all inputs.
       This function doesn't work correctly when ``func`` behaves randomly as
       it gets different gradients.


    Args:
        func (callable): A function which gets :class:`~chainer.Variable`\\ s
            and returns :class:`~chainer.Variable`\\ s. ``func`` must returns
            a tuple of :class:`~chainer.Variable`\\ s or one
            :class:`~chainer.Variable`. You can use a
            :class:`~chainer.Function`, :class:`~chainer.FunctionNode` or a
            :class:`~chainer.Link` object or any other function satisfying the
            condition.
        x_data (ndarray or tuple of ndarrays): A set of ``ndarray``\\ s to be
            passed to ``func``. If ``x_data`` is one ``ndarray`` object, it is
            treated as ``(x_data,)``.
        y_grad (ndarray or tuple of ndarrays or None):
            A set of ``ndarray``\\ s representing gradients of return-values of
            ``func``. If ``y_grad`` is one ``ndarray`` object, it is
            treated as ``(y_grad,)``. If ``func`` is a loss-function,
            ``y_grad`` should be set to ``None``.
        params (~chainer.Variable or tuple of ~chainder.Variable):
            A set of :class:`~chainer.Variable`\\ s whose gradients are
            checked. When ``func`` is a :class:`~chainer.Link` object,
            set its parameters as ``params``.
            If ``params`` is one :class:`~chainer.Variable` object,
            it is treated as ``(params,)``.
        eps (float): Epsilon value to be passed to :func:`numerical_grad`.
        atol (float): Absolute tolerance to be passed to
            :func:`chainer.testing.assert_allclose`.
        rtol (float): Relative tolerance to be passed to
            :func:`chainer.testing.assert_allclose`.
        no_grads (list of bool): Flag to skip variable for gradient assertion.
            It should be same length as ``x_data``.
        dtype (~numpy.dtype): ``x_data``, ``y_grad`` and ``params`` are casted
            to this dtype when calculating numerical gradients. Only float
            types and ``None`` are allowed.

    .. seealso::
       :func:`numerical_grad`
    """
    if dtype is not None and numpy.dtype(dtype).kind != 'f':
        raise ValueError('`dtype` is allowed only float type')

    x_data = _as_tuple(x_data)
    if y_grad is not None:
        y_grad = _as_tuple(y_grad)
    params = _as_tuple(params)

    xs = [variable.Variable(x) for x in x_data]
    y = func(*xs)
    y = _as_tuple(y)

    # All creators of `y` need to be the same because we only call
    # `y[0].backward` to call `backward` method of the creator.
    # To do so we need to insert a dummy function `Ident` to the
    # computational graph.
    # Note that `func` may not be a `Function` object.
    y = identity.Identity().apply(y)

    y_grad = _set_y_grad(y, y_grad)

    # Clear gradients which may exist if func calls backward inside of itself.
    _clear_grads(xs)
    _clear_grads(params)

    # We only need to call `backward` for one result `Variable`.
    # `Variable.backward` method calls `Function.backward` of its creator.
    y[0].backward()

    if no_grads is None:
        no_grads = [x.dtype.kind != 'f' for x in xs]
    else:
        if len(no_grads) != len(xs):
            raise ValueError(
                'Length of no_grads param and xs should be same.\n'
                'Actual: {0} != {1}'.format(len(no_grads), len(xs)))

    for skip, x in six.moves.zip(no_grads, xs):
        if skip:
            if x.grad is not None:
                raise RuntimeError(
                    'gradient of int variable must be None')
        else:
            if x.grad is None:
                raise RuntimeError(
                    'gradients of some arguments are not calculated')

    if len(xs) - no_grads.count(True) + len(params) == 0:
        # When there is no float variables, we need not to check gradient
        # values
        return

    variables = _filter_list(xs, no_grads) + list(params)
    # Keep the gradient arrays of params which may be overwritten by func
    grads = [x.grad for x in variables]

    if dtype is None:
        casted_data = [x.data for x in variables]
    else:
        if numpy.dtype(dtype).kind != 'f':
            raise ValueError('`dtype` is allowed only float type')
        casted_data = [x.data.astype(dtype, copy=False) for x in variables]

        # Even skipped variable must have the same dtype.
        for x, skip in six.moves.zip(xs, no_grads):
            if skip and x.data.dtype.kind == 'f':
                x.data = x.data.astype(dtype, copy=False)

    xp = cuda.get_array_module(*xs)
    directions = [xp.random.normal(size=x.shape) for x in variables]
    # Use unit vector
    norm = math.sqrt(sum([xp.square(d).sum() for d in directions]))
    if norm != 0:
        # norm could be zero if input arrays are 0-sized.
        scale = 1. / norm
        directions = [d * scale for d in directions]

    delta = xp.array(0., 'd')

    def g():
        # This functions is called twice in `numerical_grad`.
        # `delta` is `epsilon` or `-epsilon` in these calls.
        # See the document of `numerical_grad`.
        for x, data, direction in six.moves.zip(
                variables, casted_data, directions):
            # astype is require to store data with the given type
            data = (data.astype('d') +
                    delta * direction).astype(data.dtype)
            if numpy.isscalar(data):
                data = xp.array(data)
            x.data = data

        # Clear gradients to support func that calls backward inside of itself.
        _clear_grads(xs)
        _clear_grads(params)

        ys = func(*xs)
        ys = _as_tuple(ys)
        ys_data = tuple(y.data for y in ys)
        for x, data in six.moves.zip(variables, casted_data):
            x.data = data
        return ys_data

    gx, = numerical_grad(g, (delta,), y_grad, eps=eps)
    gx_accum = 0
    for g, direction in six.moves.zip(grads, directions):
        gx_accum += (g.astype('d') * direction).sum()

    try:
        testing.assert_allclose(gx, gx_accum, atol=atol, rtol=rtol)
    except AssertionError as e:
        f = six.StringIO()
        f.write('check_backward failed (eps={} atol={} rtol={})\n'.format(
            eps, atol, rtol))
        for i, x_ in enumerate(xs):
            f.write('inputs[{}]:\n'.format(i))
            f.write('{}\n'.format(x_))
        for i, gy_ in enumerate(y_grad):
            f.write('grad_outputs[{}]:\n'.format(i))
            f.write('{}\n'.format(gy_))
        f.write('gradients (numeric):  {}\n'.format(gx))
        f.write('gradients (backward): {}\n'.format(gx_accum))
        f.write('\n')
        f.write(str(e))
        raise AssertionError(f.getvalue())


def check_double_backward(func, x_data, y_grad, x_grad_grad, params=(),
                          params_grad_grad=(), eps=1e-3, atol=1e-4, rtol=1e-3,
                          no_grads=None, dtype=None):
    """Test twice differentiation of a given procedure.

    This function automatically checks if the backward procedure of ``func``
    is correctly implemented for further differentiation. It first computes the
    gradient of ``func`` w.r.t. its inputs in the same way as
    :func:`~chainer.gradient_check.check_backward`. This function then further
    invokes the backward procedure against the gradient variables, starting
    from the initial gradient given by ``x_grad_grad``. It also computes the
    second gradient using :func:`~chainer.gradient_check.numerical_grad`. The
    resulting gradients are compared to confirm if the second-order gradients
    are approximately correct.

    Note that this function **DOES NOT** check if the first-order
    differentiation is correct; the numerical gradient assumes that the
    first-order gradient given by the usual :meth:`chainer.Variable.backward`
    is correct. The implementation of each differentiable function should be
    tested by :func:`~chainer.gradient_check.check_backward` first, and then
    should be tested by this function if neccessary.

    For the details of the arguments, see
    :func:`~chainer.gradient_check.check_backward`. The additional arguments
    ``x_grad_grad`` and ``params_grad_grad`` are (tuples of)
    :class:`~chainer.Variable` (s) that include the initial gradient
    corresponding to the first-order gradient of each input and parameter. Note
    that the default error tolerance ``atol`` and ``rtol`` are slightly larger
    than those of :func:`~chainer.gradient_check.check_backward` because the
    numerical gradients of the second order differentiation are less accurate
    than those of the first order gradients.

    """
    x_data = _as_tuple(x_data)
    params = _as_tuple(params)
    y_grad = _as_tuple(y_grad)
    x_grad_grad = _as_tuple(x_grad_grad)
    params_grad_grad = _as_tuple(params_grad_grad)
    n_x = len(x_data)

    def first_order_grad(*inputs):
        xs = inputs[:n_x]
        gys = inputs[n_x:]

        y = _as_tuple(func(*xs))
        # Let all elements of y share the same creator.
        # See the comment in check_backward.
        y = identity.Identity().apply(y)

        _set_y_grad(y, gys)
        y[0].backward(enable_double_backprop=True)

        return tuple([x.grad_var for x in xs] + [p.grad_var for p in params])

    inputs = x_data + y_grad
    grad_grad = x_grad_grad + params_grad_grad
    try:
        check_backward(first_order_grad, inputs, grad_grad, params=params,
                       eps=eps, atol=atol, rtol=rtol, no_grads=no_grads,
                       dtype=dtype)
    except AssertionError as e:
        f = six.StringIO()
        f.write('check_double_backward failed '
                '(eps={} atol={} rtol={})\n'.format(eps, atol, rtol))
        for i, x_ in enumerate(x_data):
            f.write('input[{}]:\n'.format(i))
            f.write('{}\n'.format(x_))
        for i, gy_ in enumerate(y_grad):
            f.write('grad_output[{}]:\n'.format(i))
            f.write('{}\n'.format(gy_))
        for i, ggx_ in enumerate(x_grad_grad):
            f.write('grad_grad_input[{}]:\n'.format(i))
            f.write('{}\n'.format(ggx_))
        for i, ggp_ in enumerate(params_grad_grad):
            f.write('grad_grad_param[{}]:\n'.format(i))
            f.write('{}\n'.format(ggp_))
        f.write('\n')
        f.write(str(e))
        raise AssertionError(f.getvalue())


def _set_y_grad(y, y_grad):
    if y_grad is not None:
        if len(y) != len(y_grad):
            raise ValueError(
                '`y_grad` must have the same length of output values')
        for iy, igy in six.moves.zip(y, y_grad):
            if isinstance(igy, variable.Variable):
                iy.grad_var = igy
            else:
                iy.grad = igy
    else:
        if len(y) != 1:
            raise ValueError(
                'When `y_grad` is `None`, the function must return a'
                'zero-dimentional array')
        y_grad = (1,)
    return y_grad


def _clear_grads(xs):
    for x in xs:
        x.grad_var = None
