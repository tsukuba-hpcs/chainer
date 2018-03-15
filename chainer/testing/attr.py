import os
import unittest


try:
    import pytest
    _error = None
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def get_error():
    return _error


if _error is None:
    _gpu_limit = int(os.getenv('CHAINER_TEST_GPU_LIMIT', '-1'))

    cudnn = pytest.mark.cudnn
    ideep = pytest.mark.ideep
    slow = pytest.mark.slow

else:
    def _dummy_callable(*args, **kwargs):
        check_available()
        assert False  # Not reachable

    cudnn = _dummy_callable
    ideep = _dummy_callable
    slow = _dummy_callable


def multi_gpu(gpu_num):
    """Decorator to indicate number of GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@multi_gpu(2)``) to
    declare number of GPUs required to run. When running tests, if
    ``CHAINER_TEST_GPU_LIMIT`` environment variable is set to value greater
    than or equals to 0, test cases that require GPUs more than the limit will
    be skipped.
    """

    check_available()
    return unittest.skipIf(
        0 <= _gpu_limit < gpu_num,
        reason='{} GPUs required'.format(gpu_num))


def gpu(f):
    """Decorator to indicate that GPU is required to run the test.

    Tests can be annotated with this decorator (e.g., ``@gpu``) to
    declare that one GPU is required to run.
    """

    check_available()
    return multi_gpu(1)(pytest.mark.gpu(f))
