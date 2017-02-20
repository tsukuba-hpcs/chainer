from cupy import core


def count_nonzero(x, axis=None):
    """Counts the number of non-zero values in the array.

    Args:
        x (cupy.ndarray): The array for which to count non-zeros.
        axis (int or tuple): optionalAxis or tuple of axes along which to count
            non-zeros. Default is None, meaning that non-zeros will be counted
            along a flattened version of ``a``
    Returns:
        int or cupy.ndarray of int: Number of non-zero values in the array
            along a given axis. Otherwise, the total number of non-zero values
            in the array is returned.
    """

    if axis is None:
        return int(_count_nonzero(x))
    else:
        return _count_nonzero(x, axis=axis)


_count_nonzero = core.create_reduction_func(
    'cupy_count_nonzero',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('in0 != 0', 'a + b', 'out0 = a', None), 0)
