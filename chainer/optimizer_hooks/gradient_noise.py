import numpy

from chainer import cuda


def exponential_decay_noise(xp, shape, dtype, hook, opt):
    """Time-dependent annealed Gaussian noise function from the paper:

    `Adding Gradient Noise Improves Learning for Very Deep Networks
    <https://arxiv.org/pdf/1511.06807>`_.
    """
    std = numpy.sqrt(hook.eta / numpy.power(1 + opt.t, 0.55))
    return xp.random.normal(0, std, shape).astype(dtype)


class GradientNoise(object):
    """Optimizer/UpdateRule hook function for adding gradient noise.

    This hook function simply adds noise generated by the ``noise_func``
    to the gradient. By default it adds time-dependent annealed Gaussian
    noise to the gradient at every training step:

    .. math::

        g_t \\leftarrow g_t + N(0, \\sigma_t^2)

    where

    .. math::

        \\sigma_t^2 = \\frac{\\eta}{(1+t)^\\gamma}

    with :math:`\\eta` selected from {0.01, 0.3, 1.0} and
    :math:`\\gamma = 0.55`.

    Args:
        eta (float): Parameter that defines the scale of the noise, which for
            the default noise function is recommended to be either 0.01, 0.3
            or 1.0.
        noise_func (function): Noise generating function which by default
            is given by `Adding Gradient Noise Improves Learning for Very Deep\
            Networks <https://arxiv.org/pdf/1511.06807>`_.

    Attributes:
        ~optimizer_hooks.GradientNoise.timing (string): Specifies
                         when this hook should be called by the
                         Optimizer/UpdateRule. Valid values are
                         'pre' (before any updates) and 'post' (after any
                         updates).
        ~optimizer_hooks.GradientNoise.call_for_each_param (bool): Specifies
                         if this hook is called for each parameter (`True`)
                         or only once (`False`) by an optimizer to
                         which this hook is registered. This function does
                         not expect users to switch the value from default one,
                         which is `True`.

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'GradientNoise'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, eta, noise_func=exponential_decay_noise):
        self.eta = eta
        self.noise_func = noise_func

    def __call__(self, rule, param):
        g = param.grad
        if g is None:
            return
        xp = cuda.get_array_module(g)
        with cuda.get_device_from_array(g) as dev:
            noise = self.noise_func(xp, g.shape, g.dtype, self, rule)
            if int(dev) == -1:
                g += noise
            else:
                kernel = cuda.elementwise(
                    'T noise', 'T g', 'g += noise', 'gradient_noise')
                kernel(noise, g)
