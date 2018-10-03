import copy

import chainer
from chainer.functions.array import where
from chainer.functions.math import exponential
from chainer import utils
import warnings


class ModifiedXLogX(chainer.function_node.FunctionNode):
    def __init__(self, logx):
        self._logx = logx

    def forward(self, inputs):
        x, = inputs
        self.x_zero = utils.force_array(x == 0)
        y = utils.force_array(x * self._logx.array)
        y[self.x_zero] = 0.
        return y,

    def backward(self, indexes, grad_outputs):
        if self.x_zero.any():
            warnings.warn(
                'cannot calculate gradient for zero input.',
                RuntimeWarning)
        gy, = grad_outputs
        dx = (1 + self._logx) * (1 - self.x_zero)
        return gy * dx,


def _modified_xlogx(x):
    x = chainer.as_variable(x)
    xp = x.xp
    return ModifiedXLogX(exponential.log(
        where.where(utils.force_array(x.array > 0),
                    x, xp.ones_like(x.array)))).apply((x,))[0]


class cached_property(object):
    """Cache a result of computation of Chainer functions"""

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        with chainer.using_config('enable_backprop', True):
            value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
