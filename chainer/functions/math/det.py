import numpy.linalg

from chainer import cuda
from chainer import function
from chainer.functions.array.reshape import Reshape
from chainer.functions.math.inv import _inv_gpu
from chainer.functions.math.matmul import _as_batch_mat
from chainer.functions.math.matmul import _check_ndim
from chainer.functions.math.matmul import _convert_type
from chainer.functions.math.matmul import _get_ld
from chainer.functions.math.matmul import _mat_ptrs
from chainer import utils


def _det_gpu(b):
    # We do a batched LU decomposition on the GPU to compute
    # and compute the determinant by multiplying the diagonal
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = _as_batch_mat(b).copy()
    n = int(a.shape[1])
    n_matrices = int(a.shape[0])
    # Pivot array
    p = cuda.cupy.zeros((n, n_matrices), dtype='int32')
    # Output array
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info1 = cuda.cupy.zeros(n_matrices, dtype=numpy.intp)
    ap = _mat_ptrs(a)
    _, lda = _get_ld(a)
    cuda.cublas.sgetrf(cuda.Device().cublas_handle, n, ap.data.ptr, lda,
                       p.data.ptr, info1.data.ptr, n_matrices)
    # The determinant is the result of the diagonal entries multiplied
    # in each row of the minibatch
    det = cuda.cupy.prod(a.diagonal(axis1=1, axis2=2), axis=1)
    return det


class BatchDet(function.Function):
    @property
    def label(self):
        return 'det'

    def check_type_forward(self, in_types):
        utils.type_check.expect(in_types.size() == 1)
        a_type, = in_types
        a_type = _convert_type(a_type)
        utils.type_check.expect(a_type.dtype.kind == 'f')
        # Only a minibatch of 2D array shapes allowed
        _check_ndim(a_type, lower=3, upper=3)
        # Matrix inversion only allowed for square matrices
        # so assert the last two dimensions are equal
        utils.type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    def forward_cpu(self, x):
        self.detx = utils.force_array(numpy.linalg.det(x[0]))
        return self.detx,

    def forward_gpu(self, x):
        self.detx = _det_gpu(x[0])
        return self.detx,

    def backward_gpu(self, x, gy):
        # For matrix x, z, y the gradient is
        # d[det(x)]/dz = gy det(x) (x^-1).T
        x, = x
        gy, = gy
        grad = (gy[:, None, None] * self.detx[:, None, None] *
                _inv_gpu(x.transpose([0, 2, 1])))
        return utils.force_array(grad),

    def backward_cpu(self, x, gy):
        x, = x
        gy, = gy
        grad = (gy[:, None, None] * self.detx[:, None, None] *
                numpy.linalg.inv(x.transpose([0, 2, 1])))
        return utils.force_array(grad),


def batch_det(a):
    """Computes the determinant of a batch of square matrices.

    Args:
        a (Variable): Input array to compute the determinant for.
        The first dimension should iterate over each matrix and be
        of the batchsize.

    Returns:
        ~chainer.Variable: vector of determinants for every matrix
        in the batch.
    """
    return BatchDet()(a)


def det(a):
    """Computes the determinant of a single square matrix.

    Args:
        a (Variable): Input array to compute the determinant for.

    Returns:
        ~chainer.Variable: Scalar determinant of the matrix a.
    """
    shape = (1, a.data.shape[0], a.data.shape[1])
    batched_a = Reshape(shape)(a)
    return BatchDet()(batched_a)
