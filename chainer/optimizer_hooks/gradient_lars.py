from chainer import cuda


class GradientLARS(object):

    """Optimizer/UpdateRule hook function for layer wise adaptive rate scaling.

    See: `Large Batch Training of Convolutional Networks \
          <https://arxiv.org/abs/1708.03888>`_
       : `Convergence Analysis of Gradient Descent Algorithms \
          with Proportional Updates \
          <https://arxiv.org/abs/1801.03137>`_

    This hook function scales all gradient arrays to fit to the weight norm.

    In <https://arxiv.org/abs/1708.03888>,
        v_{t+1} = m * v_t + \\gamma * \\lambda * (\\nabla L(w_t) + \\beta w_t)
        w_{t+1} = w_{t} - v_{t+1}
        where
           \\gamma : learning_rate
           m       : momentum
           \\beta  : weight_decay
           \\eta   : lars_coeeficient
           \\lamda : local_lr
                     = \\eta * \\frac{\|w_t\|}{\|\\nabla L(w_t)\|
                       + \\beta * \|w_t\|}.
    As lr in chainer.optimizers.SGD or chainer.optimizers.MomentumSGD
    corresponds to \\gamma * \\eta, we define clip_rate as
    \\frac{\|w_t\|}{\|\\nabla L(w_t)\| + \\beta * \|w_t\|}
    and reformulate the aforementioned formula as:
    v_{t+1} = m * v_t + lr * clip_rate * (\\nabla L(w_t) + \\beta w_t)
    and implement in this way. So you do not set lars_coeeficient.

    Args:
        threashold (float): If weight norm is more than threshold,
            this function scales all gradient arrays to fit weight norm.
            (See <https://arxiv.org/abs/1801.03137>)
        weight_decay (float): Coefficient for the weight decay.
        eps (float): Small value for the numerical stability.
            (See <https://arxiv.org/abs/1801.03137>)

    Attributes:
        threashold (float): If weight norm is more than threshold,
            this function scales all gradient arrays to fit weight norm.
            (See <https://arxiv.org/abs/1801.03137>)
        weight_decay (float): Coefficient for the weight decay.
        eps (float): Small value for the numerical stability.
            (See <https://arxiv.org/abs/1801.03137>)
    """
    name = 'GradientLARS'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, threshold=1e-2, weight_decay=0.0, eps=1e-9):
        self.threshold = threshold
        self.weight_decay = weight_decay
        self.eps = eps

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return

        xp = cuda.get_array_module(p)

        # weight norm
        p_norm = xp.linalg.norm(p)
        # grad norm
        g_norm = xp.linalg.norm(g)
        local_rate = p_norm / (self.eps + g_norm + self.weight_decay * p_norm)
        rate = xp.where(p_norm > self.threshold, local_rate, 1.0)
        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.weight_decay * p
                g *= rate
            else:
                kernel = cuda.elementwise(
                    'T p, T rate, T weight_decay',
                    'T g',
                    'g += weight_decay * p; g *= rate;',
                    'lars')
                kernel(p, rate, self.weight_decay, g)
