import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Orthogonal(initializer.Initializer):
    """Initializes array with an orthogonal system.

    This initializer first makes a matrix of the same shape as the
    array to be initialized whose elements are drawn independently from
    standard Gaussian distribution.
    Next, it applies Singular Value Decomposition (SVD) to the matrix.
    Then, it initializes the array with either side of resultant
    orthogonal matrices, depending on the shape of the input array.
    Finally, the array is multiplied by the constant ``scale``.

    If the ``ndim`` of the input array is more than 2, we consider the array
    to be a matrix by concatenating all axes except the first one.

    The number of vectors consisting of the orthogonal system
    (i.e. first element of the shape of the array) must be equal to or smaller
    than the dimension of each vector (i.e. second element of the shape of
    the array).

    Attributes:
        scale (float): A constant to be multiplied by.

    Reference: Saxe et al., http://arxiv.org/abs/1312.6120

    """

    def __init__(self, scale=1.1, **kwargs):
        super(Orthogonal, self).__init__(**kwargs)
        self.scale = scale

    # TODO(Kenta Oono)
    # How do we treat overcomplete base-system case?
    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            if len(shape) == 0:
                return xp.full(shape, self.scale, self.dtype)
        else:
            assert self.dtype is None or array.dtype == self.dtype
            xp = cuda.get_array_module(array)
            if array.size == 0:
                raise ValueError('Array to be initialized must be non-empty.')
            if not array.shape:  # 0-dim case
                array[...] = self.scale
                return
            shape = array.shape
        flat_shape = (shape[0], int(numpy.prod(shape[1:])))
        if flat_shape[0] > flat_shape[1]:
            raise ValueError(
                'Cannot make orthogonal system because # of vectors ({}) is '
                'larger than that of dimensions ({})'.format(
                    flat_shape[0], flat_shape[1]))
        a = numpy.random.normal(size=flat_shape)
        # we do not have cupy.linalg.svd for now
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        if array is None:
            return xp.asarray(q.reshape(shape) * self.scale, self.dtype)
        array[...] = xp.asarray(q.reshape(shape))
        array *= self.scale
