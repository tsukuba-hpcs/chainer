import collections

import numpy
import six

from chainer.backends import cuda


def to_device(device, x):
    """Send an array to a given device.

    This method sends a given array to a given device. This method is used in
    :func:`~chainer.dataset.concat_examples`.
    You can also use this method in a custom converter method used in
    :class:`~chainer.training.Updater` and :class:`~chainer.training.Extension`
    such as :class:`~chainer.training.updaters.StandardUpdater` and
    :class:`~chainer.training.extensions.Evaluator`.

    See also :func:`chainer.dataset.concat_examples`.

    Args:
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.
        x (numpy.ndarray or cupy.ndarray): An array to send.

    Returns:
        Converted array.

    """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def concat_examples(batch, device=None, padding=None):
    """Concatenates a list of examples into array(s).

    This function converts an "array of tuples" into a "tuple of arrays".
    Specifically, given a list of examples each of which consists of
    a list of elements, this function first makes an array
    by taking the element in the same position from each example
    and concatenates them along the newly-inserted first axis
    (called `batch dimension`) into one array.
    It repeats this for all positions and returns the resulting arrays.

    The output type depends on the type of examples in ``batch``.
    For instance, consider each example consists of two arrays ``(x, y)``.
    Then, this function concatenates ``x`` 's into one array, and ``y`` 's
    into another array, and returns a tuple of these two arrays. Another
    example: consider each example is a dictionary of two entries whose keys
    are ``'x'`` and ``'y'``, respectively, and values are arrays. Then, this
    function concatenates ``x`` 's into one array, and ``y`` 's into another
    array, and returns a dictionary with two entries ``x`` and ``y`` whose
    values are the concatenated arrays.

    When the arrays to concatenate have different shapes, the behavior depends
    on the ``padding`` value. If ``padding`` is ``None`` (default), it raises
    an error. Otherwise, it builds an array of the minimum shape that the
    contents of all arrays can be substituted to. The padding value is then
    used to the extra elements of the resulting arrays.

    .. admonition:: Example

       >>> import numpy as np
       >>> from chainer import dataset
       >>> x = [([1, 2], 1),
       ...      ([3, 4], 2),
       ...      ([5, 6], 3)]
       >>> dataset.concat_examples(x)
       (array([[1, 2],
              [3, 4],
              [5, 6]]), array([1, 2, 3]))
       >>>
       >>> y = [(np.array([1, 2]), 0),
       ...      (np.array([3]), 1),
       ...      (np.array([]), 2)]
       >>> dataset.concat_examples(y, padding=100)
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([0, 1, 2]))
       >>>
       >>> z = [(np.array([1, 2]), np.array([0])),
       ...      (np.array([3]), np.array([])),
       ...      (np.array([]), np.array([2]))]
       >>> dataset.concat_examples(z, padding=(100, 200))
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([[  0],
              [200],
              [  2]]))
       >>> w = [{'feature': np.array([1, 2]), 'label': 0},
       ...      {'feature': np.array([3, 4]), 'label': 1},
       ...      {'feature': np.array([5, 6]), 'label': 2}]
       >>> dataset.concat_examples(w)  # doctest: +SKIP
       {'feature': array([[1, 2],
              [3, 4],
              [5, 6]]), 'label': array([0, 1, 2])}

    Args:
        batch (list): A list of examples. This is typically given by a dataset
            iterator.
        device (int): Device ID to which each array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, all arrays are
            left in the original device.
        padding: Scalar value for extra elements. If this is None (default),
            an error is raised on shape mismatch. Otherwise, an array of
            minimum dimensionalities that can accommodate all arrays is
            created, and elements outside of the examples are padded by this
            value.

    Returns:
        Array, a tuple of arrays, or a dictionary of arrays. The type depends
        on the type of each example in the batch.

    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(device, _concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return to_device(device, _concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int or float.
    if not isinstance(arrays[0], numpy.ndarray) and\
       not isinstance(arrays[0], cuda.ndarray):
        arrays = numpy.asarray(arrays)
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src

    return result


class ConcatWithAsyncTransfer(object):

    """Interface to concatenate data and transfer them to GPU asynchronously.

    It enables to transfer next batch of input data to GPU while GPU is
    running kernels for training using current batch of input data.

    An instance of this class is mainly intended to be used as a converter
    function of an updater like below.

    .. doctest::

        from chainer.dataset import convert
        ...
        updater = chainer.training.updaters.StandardUpdater(
                       ...,
                       converter=convert.ConcatWithAsyncTransfer(),
                       ...)

    Args:
        stream (cupy.cuda.Stream): CUDA stream. If ``None``, a stream is
            automatically created on the first call. Data transfer operation
            is launched acynchrnously using the stream.
    """

    def __init__(self, stream=None):
        self._stream = stream
        self._device = None
        self._conveyor = collections.defaultdict(
            lambda: Conveyor(self._device, self._stream))

    def __call__(self, batch, device=None, padding=None):
        """Concatenate data and transfer them to GPU asynchronously.

        See also :func:`chainer.dataset.concat_examples`.

        Args:
            batch (list): A list of examples.
            device (int): Device ID to which each array is sent.
            padding: Scalar value for extra elements.

        Returns:
            Array, a tuple of arrays, or a dictionary of arrays.
            The type depends on the type of each example in the batch.
        """
        if len(batch) == 0:
            raise ValueError('batch is empty')
        first_elem = batch[0]

        if len(self._conveyor) == 0:
            self._device = device  # device is set at first call
            if device is not None and device >= 0 and self._stream is None:
                with cuda.get_device_from_id(device):
                    self._stream = cuda.Stream(non_blocking=True)
        if device is not self._device:
            raise ValueError('device is different')

        with cuda.get_device_from_id(device):
            if isinstance(first_elem, tuple):
                result = []
                if not isinstance(padding, tuple):
                    padding = [padding] * len(first_elem)

                for i in six.moves.range(len(first_elem)):
                    self._conveyor[i].put(_concat_arrays(
                        [example[i] for example in batch], padding[i]))

                for i in six.moves.range(len(first_elem)):
                    result.append(self._conveyor[i].get())

                return tuple(result)

            elif isinstance(first_elem, dict):
                result = {}
                if not isinstance(padding, dict):
                    padding = {key: padding for key in first_elem}

                for key in first_elem:
                    self._conveyor[key].put(_concat_arrays(
                        [example[key] for example in batch], padding[key]))

                for key in first_elem:
                    result[key] = self._conveyor[key].get()

                return result

            else:
                return to_device(device, _concat_arrays(batch, padding))


class Conveyor(object):

    """Interface to handle asynchronous data transfer using double buffering.

    An asynchrous data transfer is initiated by :meth:`put`, and the result,
    the array transferred to a target device, is obtained by :meth:`get`.
    You should call :meth:`put` followed by :meth:`get`.

    Args:
        device (int): Device ID to which an array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, the array is
            left in the original device. Asynchronous data transfer is used
            only when device ID >= 0.
        stream (cupy.cuda.Stream): CUDA stream. An array is sent to GPU
            asynchronously using this stream. If ``None``, asynchronous data
            transfer is not used.
    """

    def __init__(self, device=None, stream=None):
        self._device = device
        self._stream = stream
        self._array_set = [[None, None], [None, None]]
        self._ret_array = []

    def put(self, array):
        """Initiates asynchrous transfer of an array to a target device.

        This method assumes that the input array is a numpy array and
        on host memory without page-locked. So, it first copys the data
        to page-locked host memory (so called pinned memory), then initiates
        asynchronous data transfer to a target device.

        The intermediate arrays on pinned memory and cupy arrays on the
        target device are retained at self._array_set in order to reduce number
        of memory allocation/release, and they are to be reused for subsequent
        data transfer as long as the size are the same.

        Double buffering scheme is used here, so you can initiate next data
        transfer safely even when current data is still used on the target
        device.
        """
        if self._device is None or self._device < 0 or self._stream is None:
            self._ret_array.append(to_device(self._device, array))
            return

        pin_array, cp_array = self._array_set.pop(0)
        if pin_array is not None:
            if pin_array.nbytes != array.nbytes:
                pin_array = None

        with cuda.get_device_from_id(self._device):
            if pin_array is None:
                # The global synchronization below is necceary to ensure ALL
                # operations including compute and data transfer submitted
                # to GPU so far have been completed, in order to avoid possible
                # memory corruption due to race condition among operations that
                # use different CUDA streams.
                # You can also solve this sort of race condition by preparing a
                # memory pool for each CUDA stream and using it carefully.
                cuda.cupy.cuda.runtime.deviceSynchronize()
                pin_mem = cuda.cupy.cuda.alloc_pinned_memory(array.nbytes)
                pin_array = numpy.frombuffer(pin_mem,
                                             array.dtype,
                                             array.size
                                             ).reshape(array.shape)
                cp_array = cuda.cupy.empty_like(array)

            pin_array[...] = array  # copy(CPU): paged -> pinned
            cp_array.set(pin_array, self._stream)  # copy: CPU to GPU

        self._array_set.append([pin_array, cp_array])
        self._ret_array.append(cp_array)

    def get(self):
        """Returns the array of data transferred to a target device asynchronously.

        This method first waits for completion of asynchrnous data trasfer
        initiated by :meth:`put`, then returns the array on the target
        device.

        Global synchronizaton (deviceSynchronize()) is used to ensure
        completion of asynchronous data transfer for safer reason.
        If a caller function is correctly handling the synchronization,
        local synchronization (self._stream.synchronize()) may be enough.
        """
        if (self._device is not None and self._device >= 0 and
                self._stream is not None):
            cuda.cupy.cuda.runtime.deviceSynchronize()
        return self._ret_array.pop(0)
