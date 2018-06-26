import numpy

from chainer.backends import cuda
from chainer import configuration
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import batch_renormalization
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer import variable


class BatchRenormalization(BatchNormalization):

    """Batch renormalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_renormalization` and
    :func:`~chainer.functions.fixed_batch_renormalization` functions.

    This is an extension of batch normalization, which ensures that the
    training and inference models generate the same outputs that depend on
    individual examples rather than the entire minibatch.

    See: `Batch Renormalization: Towards Reducing Minibatch Dependence in \
          Batch-Normalized Models <https://arxiv.org/abs/1702.03275>`_

    .. seealso::
       :func:`~chainer.functions.batch_renormalization`,
       :func:`~chainer.functions.fixed_batch_renormalization`
       :func:`~chainer.functions.batch_normalization`,

    """

    def __init__(self, size, rmax=1, dmax=0, decay=0.9, eps=2e-5,
                 dtype=numpy.float32, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(BatchRenormalization, self).__init__(size, decay, eps, dtype,
                                                   use_gamma, use_beta,
                                                   initial_gamma, initial_beta)
        self.rmax = rmax  # maximum allowed correction of variance
        self.dmax = dmax  # maximum allowed correction of mean
        self.r = None
        self.d = None

    def __call__(self, x, finetune=False):
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype)

        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype)

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = batch_renormalization.batch_renormalization(
                x, gamma, beta, self.rmax, self.dmax,
                self.eps, self.avg_mean, self.avg_var, decay,
                update_statistics=True)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            var = self.avg_var
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret
