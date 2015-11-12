from __future__ import division
import collections
import ctypes
import sys

import numpy
import six

from cupy import flags
from cupy import util

cimport cython
from libcpp cimport vector

from cupy.cuda cimport cublas
from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy.cuda cimport module


DEF MAX_NDIM = 25

@cython.profile(False)
cpdef inline tuple _get_size(object size):
    if size is None:
        return ()
    if isinstance(size, collections.Sequence):
        return tuple(size)
    if isinstance(size, int):
        return size,
    raise ValueError('size should be None, collections.Sequence, or int')


@cython.profile(False)
cdef inline _should_use_rop(x, y):
    xp = getattr(x, '__array_priority__', 0)
    yp = getattr(y, '__array_priority__', 0)
    return xp < yp and not isinstance(y, ndarray)


@cython.profile(False)
cdef inline bint _vector_equal(
        vector.vector[Py_ssize_t]& x, vector.vector[Py_ssize_t]& y):
    cdef Py_ssize_t n = x.size()
    if n != y.size():
        return False
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True


cdef class ndarray:

    """Multi-dimensional array on a CUDA device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content on the
    current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memptr (cupy.cuda.MemoryPointer): Pointer to the array content head.
        strides (tuple of ints): The strides for axes.

    Attributes:
        data (cupy.cuda.MemoryPointer): Pointer to the array content head.
        base (None or cupy.ndarray): Base array from which this array is
            created as a view.

    """

    cdef:
        public Py_ssize_t _size
        public vector.vector[Py_ssize_t] _shape
        public vector.vector[Py_ssize_t] _strides
        readonly bint _c_contiguous
        readonly bint _f_contiguous
        readonly object _dtype
        readonly memory.MemoryPointer data
        readonly ndarray base


    def __init__(self, shape, dtype=float, memptr=None):
        cdef Py_ssize_t size
        self._shape = _get_size(shape)
        ndim = self._shape.size()
        for x in self._shape:
            if x < 0:
                raise ValueError('Negative dimensions are not allowed')
        self._dtype = numpy.dtype(dtype)
        self._size = _prod_ssize_t(self._shape)
        self._strides = _get_contiguous_strides(self._shape, self.itemsize)

        if memptr is None:
            self.data = memory.alloc(self._size * self._dtype.itemsize)
        else:
            self.data = memptr
        self.base = None

        self._c_contiguous = True
        self._update_f_contiguity()


    # The definition order of attributes and methods are borrowed from the
    # order of documentation at the following NumPy document.
    # http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

    # -------------------------------------------------------------------------
    # Memory layout
    # -------------------------------------------------------------------------
    @property
    def flags(self):
        """Object containing memory-layout information.

        It only contains ``c_contiguous``, ``f_contiguous``, and ``owndata``
        attributes. All of these are read-only. Accessing by indexes is also
        supported.

        .. seealso:: :attr:`numpy.ndarray.flags`

        """
        return flags.Flags(self._c_contiguous, self._f_contiguous,
                           self.base is not None)

    property shape:
        """Lengths of axes.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        def __get__(self):
            return tuple(self._shape)

        def __set__(self, newshape):
            cdef vector.vector[Py_ssize_t] shape, strides
            shape = _infer_unknown_dimension(newshape, self._size)
            strides = _get_strides_for_nocopy_reshape(self, newshape)
            if strides.size() != shape.size():
                raise AttributeError('incompatible shape')
            self._shape = shape
            self._strides = strides
            self._update_f_contiguity()

    @property
    def strides(self):
        """Strides of axes in bytes.

        .. seealso:: :attr:`numpy.ndarray.strides`

        """
        return tuple(self._strides)

    @property
    def ndim(self):
        """Number of dimensions.

        ``a.ndim`` is equivalent to ``len(a.shape)``.

        .. seealso:: :attr:`numpy.ndarray.ndim`

        """
        return self._shape.size()

    @property
    def size(self):
        """Number of elements this array holds.

        This is equivalent to product over the shape tuple.

        .. seealso:: :attr:`numpy.ndarray.size`

        """
        return self._size

    @property
    def itemsize(self):
        """Size of each element in bytes.

        .. seealso:: :attr:`numpy.ndarray.itemsize`

        """
        return self._dtype.itemsize

    @property
    def nbytes(self):
        """Size of whole elements in bytes.

        It does not count skips between elements.

        .. seealso:: :attr:`numpy.ndarray.nbytes`

        """
        return self._size * self.itemsize

    # -------------------------------------------------------------------------
    # Data type
    # -------------------------------------------------------------------------
    @property
    def dtype(self):
        """Dtype object of element type.

        .. seealso::
           `Data type objects (dtype) \
           <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_

        """
        return self._dtype

    # -------------------------------------------------------------------------
    # Other attributes
    # -------------------------------------------------------------------------
    @property
    def T(self):
        """Shape-reversed view of the array.

        If ndim < 2, then this is just a reference to the array itself.

        """
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

    __array_priority__ = 100

    # -------------------------------------------------------------------------
    # Array interface
    # -------------------------------------------------------------------------
    # TODO(beam2d): Implement __array_interface__

    # -------------------------------------------------------------------------
    # foreign function interface
    # -------------------------------------------------------------------------
    @property
    def cstruct(self):
        """C representation of the array.

        This property is used for sending an array to CUDA kernels. The type of
        returned C structure is different for different dtypes and ndims. The
        definition of C type is written in ``cupy/carray.cuh``.

        """
        return to_carray(self.data.ptr, self.size, self.shape, self.strides)

    # -------------------------------------------------------------------------
    # Array conversion
    # -------------------------------------------------------------------------
    # TODO(okuta): Implement item

    def tolist(self):
        """Converts the array to a (possibly nested) Python list.

        Returns:
            list: The possibly nested Python list of array elements.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        return self.get().tolist()

    # TODO(okuta): Implement itemset
    # TODO(okuta): Implement tostring
    # TODO(okuta): Implement tobytes

    def tofile(self, fid, sep='', format='%s'):
        """Writes the array to a file.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        self.get().tofile(fid, sep, format)

    def dump(self, file):
        """Dumps a pickle of the array to a file.

        Dumped file can be read back to cupy.ndarray by
        :func:`cupy.load`.

        """
        six.moves.cPickle.dump(self, file, -1)

    def dumps(self):
        """Dumps a pickle of the array to a string."""
        return six.moves.cPickle.dumps(self, -1)

    def astype(self, dtype, copy=True):
        """Casts the array to given data type.

        Args:
            dtype: Type specifier.
            copy (bool): If it is False and no cast happens, then this method
                returns the array itself. Otherwise, a copy is returned.

        Returns:
            If ``copy`` is False and no cast is required, then the array itself
            is returned. Otherwise, it returns a (possibly casted) copy of the
            array.

        .. note::
           This method currently does not support ``order``, ``casting``, and
           ``subok`` arguments.

        .. seealso:: :meth:`numpy.ndarray.astype`

        """
        # TODO(beam2d): Support ordering, casting, and subok option
        dtype = numpy.dtype(dtype)
        if dtype == self._dtype:
            if copy:
                return self.copy()
            else:
                return self
        else:
            newarray = ndarray(self.shape, dtype=dtype)
            elementwise_copy(self, newarray)
            return newarray

    # TODO(okuta): Implement byteswap

    cpdef ndarray copy(self):
        """Returns a copy of the array.

        .. seealso::
           :func:`cupy.copy` for full documentation,
           :meth:`numpy.ndarray.copy`

        """
        cdef ndarray a, newarray
        # TODO(beam2d): Support ordering option
        if self._size == 0:
            return ndarray(self.shape, self._dtype)

        a = self
        if not self._c_contiguous:
            a = ascontiguousarray(self)
            if a.data.device.id == device.get_device_id():
                return a
        newarray = ndarray(a.shape, a.dtype)
        newarray.data.copy_from_device(a.data, a.nbytes)
        return newarray

    cpdef ndarray view(self, dtype=None):
        """Returns a view of the array.

        Args:
            dtype: If this is different from the data type of the array, the
                returned view reinterpret the memory sequence as an array of
                this type.

        Returns:
            cupy.ndarray: A view of the array. A reference to the original
            array is stored at the :attr:`~ndarray.base` attribute.

        .. seealso:: :meth:`numpy.ndarray.view`

        """
        # Use __new__ instead of __init__ to skip recomputation of contiguity
        cdef ndarray v
        v = ndarray.__new__(ndarray)
        v._size = self._size
        v._shape = self._shape
        v._strides = self._strides
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v._dtype = self._dtype
        v.data = self.data
        v.base = self.base if self.base is not None else self
        return v

    # TODO(okuta): Implement getfield
    # TODO(okuta): Implement setflags

    cpdef fill(self, value):
        """Fills the array with a scalar value.

        Args:
            value: A scalar value to fill the array content.

        .. seealso:: :meth:`numpy.ndarray.fill`

        """
        elementwise_copy(value, self, dtype=self._dtype)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------
    def reshape(self, *shape):
        """Returns an array of a different shape and the same content.

        .. seealso::
           :func:`cupy.reshape` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        # TODO(beam2d): Support ordering option
        if len(shape) == 1 and isinstance(shape[0], collections.Sequence):
            shape = shape[0]
        cdef vector.vector[Py_ssize_t] newshape, newstrides
        newshape = shape
        # TODO(beam2d): Support ordering option
        if isinstance(newshape, collections.Sequence):
            newshape = tuple(newshape)
        else:
            newshape = newshape,

        newshape = _infer_unknown_dimension(newshape, self.size)
        if _vector_equal(newshape, self._shape):
            return self.view()

        cdef ndarray newarray
        newstrides = _get_strides_for_nocopy_reshape(self, newshape)
        if newstrides.size() == newshape.size():
            newarray = self.view()
        else:
            newarray = self.copy()
            newstrides = _get_strides_for_nocopy_reshape(newarray, newshape)

        assert newshape.size() == newstrides.size()
        newarray._shape = newshape
        newarray._strides = newstrides
        newarray._update_f_contiguity()
        return newarray

    # TODO(okuta): Implement resize

    def transpose(self, *axes):
        """Returns a view of the array with axes permuted.

        .. seealso::
           :func:`cupy.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        cdef ndarray ret
        cdef vector.vector[Py_ssize_t] vec_axes, a_axes, temp_axes
        cdef Py_ssize_t ndim, axis
        if len(axes) == 1:
            a = axes[0]
            if a is None or isinstance(a, collections.Sequence):
                axes = a

        ndim = self._shape.size()
        ret = self.view()
        if not axes:
            ret._shape.assign(self._shape.rbegin(), self._shape.rend())
            ret._strides.assign(self._strides.rbegin(), self._strides.rend())
            ret._c_contiguous = self._f_contiguous
            ret._f_contiguous = self._c_contiguous
            return ret

        if ndim != len(axes):
            raise ValueError('Invalid axes value: %s' % str(axes))

        for i in axes:
            axis = <Py_ssize_t?>i
            if axis < -ndim or axis >= ndim:
                raise IndexError('Axes overrun')
            vec_axes.push_back(axis % ndim)

        for i in range(ndim):
            a_axes.push_back(i)
        if _vector_equal(a_axes, vec_axes):
            return ret
        temp_axes.assign(vec_axes.rbegin(), vec_axes.rend())
        if _vector_equal(a_axes, temp_axes):
            ret._shape.assign(self._shape.rbegin(), self._shape.rend())
            ret._strides.assign(self._strides.rbegin(), self._strides.rend())
            ret._c_contiguous = self._f_contiguous
            ret._f_contiguous = self._c_contiguous
            return ret

        if ndim != len(set(temp_axes)):
            raise ValueError('Invalid axes value: %s' % str(axes))

        ret._shape.clear()
        ret._strides.clear()
        for axis in vec_axes:
            ret._shape.push_back(self._shape[axis])
            ret._strides.push_back(self._strides[axis])
        ret._update_contiguity()

        return ret

    cpdef ndarray swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2):
        """Returns a view of the array with two axes swapped.

        .. seealso::
           :func:`cupy.swapaxes` for full documentation,
           :meth:`numpy.ndarray.swapaxes`

        """
        cdef Py_ssize_t ndim=self.ndim
        if axis1 >= ndim or axis2 >= ndim:
            raise ValueError('Axis out of range')
        axes = list(range(ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self.transpose(axes)

    cpdef ndarray flatten(self):
        """Returns a copy of the array flatten into one dimension.

        It currently supports C-order only.

        Returns:
            cupy.ndarray: A copy of the array with one dimension.

        .. seealso:: :meth:`numpy.ndarray.flatten`

        """
        # TODO(beam2d): Support ordering option
        if self._c_contiguous:
            newarray = self.copy()
        else:
            newarray = ndarray(self.shape, self.dtype)
            elementwise_copy(self, newarray)

        newarray._shape.assign(1, self._size)
        newarray._strides.assign(1, self.itemsize)
        newarray._update_contiguity()
        return newarray

    cpdef ndarray ravel(self):
        """Returns an array flattend into one dimension.

        .. seealso::
           :func:`cupy.ravel` for full documentation,
           :meth:`numpy.ndarray.ravel`

        """
        # TODO(beam2d): Support ordering option
        return self.reshape(self.size)

    cpdef ndarray squeeze(self, axis=None):
        """Returns a view with size-one axes removed.

        .. seealso::
           :func:`cupy.squeeze` for full documentation,
           :meth:`numpy.ndarray.squeeze`

        """
        cdef vector.vector[Py_ssize_t] vec_axis, newshape, newstrides
        cdef Py_ssize_t j, n
        if axis is None:
            for i in range(self._shape.size()):
                n = self._shape[i]
                if n == 1:
                    vec_axis.push_back(i)
        elif isinstance(axis, int):
            vec_axis.push_back(axis)
        else:
            vec_axis = axis

        if vec_axis.size() == 0:
            return self
        j = 0
        for i in range(self._shape.size()):
            n = self._shape[i]
            if j < vec_axis.size() and i == vec_axis[j]:
                if n != 1:
                    raise RuntimeError('Cannot squeeze dimension of size > 1')
                j += 1
            else:
                newshape.push_back(n)
                newstrides.push_back(self._strides[i])

        v = self.view()
        v._shape = newshape
        v._strides = newstrides
        v._update_contiguity()
        return v

    # -------------------------------------------------------------------------
    # Item selection and manipulation
    # -------------------------------------------------------------------------
    cpdef ndarray take(self, indices, axis=None, out=None):
        """Returns an array of elements at given indices along the axis.

        .. seealso::
           :func:`cupy.take` for full documentation,
           :meth:`numpy.ndarray.take`

        """
        return take(self, indices, axis, out)

    # TODO(okuta): Implement put
    # TODO(okuta): Implement repeat
    # TODO(okuta): Implement choose
    # TODO(okuta): Implement sort
    # TODO(okuta): Implement argsort
    # TODO(okuta): Implement partition
    # TODO(okuta): Implement argpartition
    # TODO(okuta): Implement searchsorted
    # TODO(okuta): Implement nonzero
    # TODO(okuta): Implement compress

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Returns a view of the specified diagonals.

        .. seealso::
           :func:`cupy.diagonal` for full documentation,
           :meth:`numpy.ndarray.diagonal`

        """
        return diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    def max(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the maximum along a given axis.

        .. seealso::
           :func:`cupy.amax` for full documentation,
           :meth:`numpy.ndarray.max`

        """
        return amax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def argmax(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the indices of the maximum along a given axis.

        .. seealso::
           :func:`cupy.argmax` for full documentation,
           :meth:`numpy.ndarray.argmax`

        """
        return argmax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def min(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the minimum along a given axis.

        .. seealso::
           :func:`cupy.amin` for full documentation,
           :meth:`numpy.ndarray.min`

        """
        return amin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def argmin(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the indices of the minimum along a given axis.

        .. seealso::
           :func:`cupy.argmin` for full documentation,
           :meth:`numpy.ndarray.argmin`

        """
        return argmin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    # TODO(okuta): Implement ptp

    def clip(self, a_min, a_max, out=None):
        """Returns an array with values limited to [a_min, a_max].

        .. seealso::
           :func:`cupy.clip` for full documentation,
           :meth:`numpy.ndarray.clip`

        """
        return _clip(self, a_min, a_max, out=out)

    # TODO(okuta): Implement round

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Returns the sum along diagonals of the array.

        .. seealso::
           :func:`cupy.trace` for full documentation,
           :meth:`numpy.ndarray.trace`

        """
        d = self.diagonal(offset, axis1, axis2)
        return d.sum(-1, dtype, out, False)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the sum along a given axis.

        .. seealso::
           :func:`cupy.sum` for full documentation,
           :meth:`numpy.ndarray.sum`

        """
        return _sum(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumsum

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the mean along a given axis.

        .. seealso::
           :func:`cupy.mean` for full documentation,
           :meth:`numpy.ndarray.mean`

        """
        return _mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Returns the variance along a given axis.

        .. seealso::
           :func:`cupy.var` for full documentation,
           :meth:`numpy.ndarray.var`

        """
        return var(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Returns the standard deviation along a given axis.

        .. seealso::
           :func:`cupy.std` for full documentation,
           :meth:`numpy.ndarray.std`

        """
        return std(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=None):
        """Returns the product along a given axis.

        .. seealso::
           :func:`cupy.prod` for full documentation,
           :meth:`numpy.ndarray.prod`

        """
        return _prod(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumprod

    def all(self, axis=None, out=None, keepdims=False):
        return _all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return _any(self, axis=axis, out=out, keepdims=keepdims)

    # -------------------------------------------------------------------------
    # Arithmetic and comparison operations
    # -------------------------------------------------------------------------
    # Comparison operators:

    def __richcmp__(object self, object other, int op):
        if op == 0:
            return less(self, other)
        if op == 1:
            return less_equal(self, other)
        if op == 2:
            return equal(self, other)
        if op == 3:
            return not_equal(self, other)
        if op == 4:
            return greater(self, other)
        if op == 5:
            return greater_equal(self, other)
        return NotImplemented

    # Truth value of an array (bool):

    def __nonzero__(self):
        if self._size == 0:
            return False
        elif self._size == 1:
            return bool(self.get())
        else:
            msg = 'The truth value of an array with more than one element is ' \
                  'ambiguous. Use a.any() or a.all()'
            raise ValueError(msg)

    # Unary operations:

    def __neg__(self):
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __invert__(self):
        return invert(self)

    # Arithmetic:

    def __add__(x, y):
        if _should_use_rop(x, y):
            return y.__radd__(x)
        else:
            return add(x, y)

    def __sub__(x, y):
        if _should_use_rop(x, y):
            return y.__rsub__(x)
        else:
            return subtract(x, y)

    def __mul__(x, y):
        if _should_use_rop(x, y):
            return y.__rmul__(x)
        else:
            return multiply(x, y)

    def __div__(x, y):
        if _should_use_rop(x, y):
            return y.__rdiv__(x)
        else:
            return divide(x, y)

    def __truediv__(x, y):
        if _should_use_rop(x, y):
            return y.__rtruediv__(x)
        else:
            return true_divide(x, y)

    def __floordiv__(x, y):
        if _should_use_rop(x, y):
            return y.__rfloordiv__(x)
        else:
            return floor_divide(x, y)

    def __mod__(x, y):
        if _should_use_rop(x, y):
            return y.__rmod__(x)
        else:
            return remainder(x, y)

    def __divmod__(x, y):
        if _should_use_rop(x, y):
            return y.__rdivmod__(x)
        else:
            return divmod(x, y)

    def __pow__(x, y, modulo):
        # Note that we ignore the modulo argument as well as NumPy.
        if _should_use_rop(x, y):
            return y.__rpow__(x)
        else:
            return power(x, y)

    def __lshift__(x, y):
        if _should_use_rop(x, y):
            return y.__rlshift__(x)
        else:
            return left_shift(x, y)

    def __rshift__(x, y):
        if _should_use_rop(x, y):
            return y.__rrshift__(x)
        else:
            return right_shift(x, y)

    def __and__(x, y):
        if _should_use_rop(x, y):
            return y.__rand__(x)
        else:
            return bitwise_and(x, y)

    def __or__(x, y):
        if _should_use_rop(x, y):
            return y.__ror__(x)
        else:
            return bitwise_or(x, y)

    def __xor__(x, y):
        if _should_use_rop(x, y):
            return y.__rxor__(x)
        else:
            return bitwise_xor(x, y)

    # Arithmetic, in-place:

    def __iadd__(self, other):
        return add(self, other, self)

    def __isub__(self, other):
        return subtract(self, other, self)

    def __imul__(self, other):
        return multiply(self, other, self)

    def __idiv__(self, other):
        return divide(self, other, self)

    def __itruediv__(self, other):
        return true_divide(self, other, self)

    def __ifloordiv__(self, other):
        return floor_divide(self, other, self)

    def __imod__(self, other):
        return remainder(self, other, self)

    def __ipow__(self, other):
        return power(self, other, self)

    def __ilshift__(self, other):
        return left_shift(self, other, self)

    def __irshift__(self, other):
        return right_shift(self, other, self)

    def __iand__(self, other):
        return bitwise_and(self, other, self)

    def __ior__(self, other):
        return bitwise_or(self, other, self)

    def __ixor__(self, other):
        return bitwise_xor(self, other, self)

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------
    # For standard library functions:

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __reduce__(self):
        return array, (self.get(),)

    # Basic customization:

    # cupy.ndarray does not define __new__

    def __array__(self, dtype=None):
        if dtype is None or self._dtype == dtype:
            return self
        else:
            return self.astype(dtype)

    # TODO(okuta): Implement __array_wrap__

    # Container customization:

    def __len__(self):
        if self._shape.size() == 0:
            raise TypeError('len() of unsized object')
        return self._shape[0]

    def __getitem__(self, slices):
        # It supports the basic indexing (by slices, ints or Ellipsis) only.
        # TODO(beam2d): Support the advanced indexing of NumPy.
        if not isinstance(slices, tuple):
            slices = [slices]
        else:
            slices = list(slices)

        if six.moves.builtins.any(isinstance(s, ndarray) for s in slices):
            raise ValueError('Advanced indexing is not supported')

        # Expand ellipsis into empty slices
        n_newaxes = slices.count(newaxis)
        n_ellipses = slices.count(Ellipsis)
        ndim = self._shape.size()
        if n_ellipses > 0:
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed in index')
            ellipsis = slices.index(Ellipsis)
            ellipsis_size = ndim - (len(slices) - n_newaxes - 1)
            slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size

        slices += [slice(None)] * (ndim - len(slices) + n_newaxes)

        # Create new shape and stride
        shape = []
        strides = []

        j = 0
        offset = 0
        for i, s in enumerate(slices):
            if s is newaxis:
                shape.append(1)
                if j < ndim:
                    strides.append(self._strides[j])
                elif ndim > 0:
                    strides.append(self._strides[-1])
                else:
                    strides.append(self.itemsize)
            elif isinstance(s, slice):
                s = complete_slice(s, self._shape[j])
                if s.step > 0:
                    dim = (s.stop - s.start - 1) // s.step + 1
                else:
                    dim = (s.stop - s.start + 1) // s.step + 1

                shape.append(dim)
                strides.append(self._strides[j] * s.step)

                offset += s.start * self._strides[j]
                j += 1
            elif numpy.isscalar(s):
                ind = int(s)
                if ind < 0:
                    ind += self._shape[j]
                if not (0 <= ind < self._shape[j]):
                    msg = 'Index %s is out of bounds for axis %s with size %s' \
                          % (s, j, self._shape[j])
                    raise IndexError(msg)
                offset += ind * self._strides[j]
                j += 1
            else:
                raise TypeError('Invalid index type: %s' % type(slices[i]))

        v = self.view()
        v._shape = shape
        v._strides = strides
        v._size = internal_prod(shape)
        v.data = self.data + offset
        v._update_contiguity()

        return v

    def __setitem__(self, slices, value):
        cdef ndarray v, x, y
        v = self[slices]
        if isinstance(value, ndarray):
            y, x = broadcast(v, value).values
            if (_vector_equal(y._shape, x._shape) and
                    _vector_equal(y._strides, x._strides)):
                if y.data.ptr == x.data.ptr:
                    return  # Skip since x and y are the same array
                elif y._c_contiguous and x.dtype == y.dtype:
                    y.data.copy_from(x.data, x.nbytes)
                    return
            elementwise_copy(x, y)
        else:
            v.fill(value)

    # TODO(okuta): Implement __getslice__
    # TODO(okuta): Implement __setslice__
    # TODO(okuta): Implement __contains__

    # Conversion:

    def __int__(self):
        return int(self.get())

    if sys.version_info < (3,):
        def __long__(self):
            # Avoid using long() for flake8
            return self.get().__long__()

    def __float__(self):
        return float(self.get())

    def __oct__(self):
        return oct(self.get())

    def __hex__(self):
        return hex(self.get())

    # String representations:

    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        return str(self.get())

    # -------------------------------------------------------------------------
    # Methods outside of the ndarray main documentation
    # -------------------------------------------------------------------------
    def dot(self, b, out=None):
        """Returns the dot product with given array.

        .. seealso::
           :func:`cupy.dot` for full documentation,
           :meth:`numpy.ndarray.dot`

        """
        return dot(self, b, out)

    # -------------------------------------------------------------------------
    # Cupy specific attributes and methods
    # -------------------------------------------------------------------------
    @property
    def device(self):
        """CUDA device on which this array resides."""
        return self.data.device

    cpdef get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            numpy.ndarray: Copy of the array on host memory.

        """
        a_gpu = ascontiguousarray(self)
        a_cpu = numpy.empty(self._shape, dtype=self._dtype)
        ptr = a_cpu.ctypes.data_as(ctypes.c_void_p)
        if stream is None:
            a_gpu.data.copy_to_host(ptr, a_gpu.nbytes)
        else:
            a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes, stream)
        return a_cpu

    cpdef set(self, arr, stream=None):
        """Copies an array on the host memory to cuda.ndarray.

        Args:
            arr (numpy.ndarray): The source array on the host memory.
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        """
        if not isinstance(arr, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to cupy.ndarray')
        if self._dtype != arr.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                arr.dtype, self._dtype))
        if self.shape != arr.shape:
            raise ValueError('Shape mismatch')
        if not self.flags.c_contiguous:
            raise RuntimeError('Cannot set to non-contiguous array')

        arr = numpy.ascontiguousarray(arr)
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        if stream is None:
            self.data.copy_from_host(ptr, self.nbytes)
        else:
            self.data.copy_from_host_async(ptr, self.nbytes, stream)

    cpdef ndarray reduced_view(self, dtype=None):
        """Returns a view of the array with minimum number of dimensions.

        Args:
            dtype: Data type specifier. If it is given, then the memory
                sequence is reinterpreted as the new type.

        Returns:
            cupy.ndarray: A view of the array with reduced dimensions.

        """
        cdef vector.vector[Py_ssize_t] shape, strides
        cdef Py_ssize_t ndim
        ndim = self._shape.size()
        if ndim <= 1:
            return self
        _get_reduced_dims(
            self._shape, self._strides, self.itemsize, shape, strides)
        if ndim == shape.size():
            return self

        view = self.view(dtype=dtype)
        view._shape = shape
        view._strides = strides
        view._update_f_contiguity()
        return view

    cpdef _update_c_contiguity(self):
        self._c_contiguous = _get_c_contiguity(
            self._shape, self._strides, self.itemsize)

    cpdef _update_f_contiguity(self):
        cdef vector.vector[Py_ssize_t] rev_shape, rev_strides
        rev_shape.assign(self._shape.rbegin(), self._shape.rend())
        rev_strides.assign(self._strides.rbegin(), self._strides.rend())
        self._f_contiguous = _get_c_contiguity(
           rev_shape, rev_strides, self.itemsize)

    cpdef _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()


newaxis = numpy.newaxis  # == None

six_zip = six.moves.zip


@cython.profile(False)
cdef inline Py_ssize_t _prod_ssize_t(
        vector.vector[Py_ssize_t]& arr, Py_ssize_t init=1):
    for a in arr:
        init *= a
    return init


@cython.profile(False)
cpdef inline Py_ssize_t internal_prod(args, Py_ssize_t init=1) except *:
    cdef long long arg
    for arg in args:
        init *= arg
    return init


cpdef void _get_reduced_dims(
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize, vector.vector[Py_ssize_t]& reduced_shape,
        vector.vector[Py_ssize_t]& reduced_strides) except *:
    cdef vector.vector[Py_ssize_t] tmp_shape, tmp_strides
    cdef Py_ssize_t i, ndim, sh, st, prev_st, index
    assert shape.size() == strides.size()
    ndim = shape.size()
    reduced_shape.clear()
    reduced_strides.clear()
    if ndim == 0:
        return

    for i in range(ndim):
        sh = shape[i]
        if sh == 0:
            reduced_shape.push_back(0)
            reduced_strides.push_back(itemsize)
            return
        if sh != 1:
            tmp_shape.push_back(sh)
            tmp_strides.push_back(strides[i])
    if tmp_shape.size() == 0:
        return

    reduced_shape.push_back(tmp_shape[0])
    reduced_strides.push_back(tmp_strides[0])
    index = 0
    for i in range(tmp_shape.size() - 1):
        sh = tmp_shape[i + 1]
        st = tmp_strides[i + 1]
        if tmp_strides[i] == sh * st:
            reduced_shape[index] *= sh
            reduced_strides[index] = st
        else:
            reduced_shape.push_back(sh)
            reduced_strides.push_back(st)
            index += 1


cpdef vector.vector[Py_ssize_t] _get_strides_for_nocopy_reshape(
        ndarray a, vector.vector[Py_ssize_t]& newshape) except *:
    cdef vector.vector[Py_ssize_t] newstrides
    cdef Py_ssize_t size, itemsize, ndim, dim, last_stride
    size = a.size
    if size != _prod_ssize_t(newshape):
        return newstrides

    itemsize = a.itemsize
    if size == 1:
        newstrides.assign(newshape.size(), itemsize)
        return newstrides

    cdef vector.vector[Py_ssize_t] shape, strides
    _get_reduced_dims(a._shape, a._strides, itemsize, shape, strides)

    ndim = shape.size()
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = shape[0] * strides[0]
    for size in newshape:
        if size <= 1:
            newstrides.push_back(last_stride)
            continue
        if dim >= ndim or shape[dim] % size != 0:
            newstrides.clear()
            break
        shape[dim] //= size
        last_stride = shape[dim] * strides[dim]
        newstrides.push_back(last_stride)
        if shape[dim] == 1:
            dim += 1
    return newstrides


cpdef vector.vector[Py_ssize_t] _get_contiguous_strides(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t itemsize) except *:
    cdef vector.vector[Py_ssize_t] strides
    cdef Py_ssize_t st, sh
    strides.resize(shape.size(), 0)
    st = itemsize
    for i in range(shape.size() - 1, -1, -1):
        strides[i] = st
        sh = shape[i]
        if sh > 1:
            st *= sh
    return strides


cpdef bint _get_c_contiguity(
        vector.vector[Py_ssize_t]& shape, vector.vector[Py_ssize_t]& strides,
        Py_ssize_t itemsize) except *:
    cdef vector.vector[Py_ssize_t] r_shape, r_strides
    cpdef Py_ssize_t ndim
    _get_reduced_dims(shape, strides, itemsize, r_shape, r_strides)
    ndim = r_strides.size()
    return ndim == 0 or (ndim == 1 and r_strides[0] == itemsize)


cpdef vector.vector[Py_ssize_t] _infer_unknown_dimension(
        vector.vector[Py_ssize_t]& shape, Py_ssize_t size) except *:
    cdef vector.vector[Py_ssize_t] ret = shape
    cdef Py_ssize_t cnt=0, index=-1, new_size=1
    for i in range(shape.size()):
        if shape[i] < 0:
            cnt += 1
            index = i
        else:
            new_size *= shape[i]
    if cnt == 0:
        return ret
    if cnt > 1:
        raise ValueError('can only specify only one unknown dimension')
    if (size != 0 and new_size == 0) or size % new_size != 0:
        raise ValueError('total size of new array must be unchanged')
    ret[index] = size // new_size
    return ret


cpdef slice complete_slice(slice slc, Py_ssize_t dim):
    cpdef Py_ssize_t start=0, stop=0, step=0
    cpdef bint start_none, stop_none
    if slc.step is None:
        step = 1
    else:
        try:
            step = int(slc.step)
        except TypeError:
            raise IndexError(
                'slice.step must be int or None: {}'.format(slc))

    if step == 0:
        raise ValueError('Slice step must be nonzero.')

    start_none = slc.start is None
    if not start_none:
        try:
            start = int(slc.start)
        except TypeError:
            raise IndexError(
                'slice.start must be int or None: {}'.format(slc))
        if start < 0:
            start += dim

    stop_none = slc.stop is None
    if not stop_none:
        try:
            stop = int(slc.stop)
        except TypeError:
            raise IndexError(
                'slice.stop must be int or None: {}'.format(slc))
        if stop < 0:
            stop += dim

    if step > 0:
        start = 0 if start_none else max(0, min(dim, start))
        stop = dim if stop_none else max(start, min(dim, stop))
    else:
        start = dim - 1 if start_none else max(0, min(dim, start))
        stop = -1 if stop_none else max(0, min(start, stop))

    return slice(start, stop, step)


include "carray.pxi"
include "elementwise.pxi"
include "reduction.pxi"


# =============================================================================
# Routines
# =============================================================================

_id = 'out0 = in0'

elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    _id)


elementwise_copy_where = create_ufunc(
    'cupy_copy_where',
    ('??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'),
    'if (in1) out0 = in0')


_divmod_float = '''
    out0_type a = _floor_divide(in0, in1);
    out0 = a;
    out1 = in0 - a * in1'''


divmod = create_ufunc(
    'cupy_divmod',
    ('bb->bb', 'BB->BB', 'hh->hh', 'HH->HH', 'ii->ii', 'II->II', 'll->ll',
     'LL->LL', 'qq->qq', 'QQ->QQ',
     ('ee->ee', _divmod_float),
     ('ff->ff', _divmod_float),
     ('dd->dd', _divmod_float)),
    '''
    if (in1 == 0) {
        out0 = 0;
        out1 = 0;
    } else {
        out0_type a = _floor_divide(in0, in1);
        out0 = a;
        out1 = in0 - a * in1;
    }''')


_min_max_preamble = '''
struct min_max_st{
    type_in0_raw value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(type_in0_raw v) : value(v), index(0) { }
    __device__ min_max_st(type_in0_raw v, int i) : value(v), index(i) { }
};
__device__ min_max_st my_min(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(min(a.value, b.value));
}
__device__ min_max_st my_max(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(max(a.value, b.value));
}
__device__ min_max_st my_argmin(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value) return min_max_st(a.value, min(a.index, b.index));
    return (a.value <= b.value) ? a : b;
}
__device__ min_max_st my_argmax(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value) return min_max_st(a.value, min(a.index, b.index));
    return (a.value >= b.value) ? a : b;
}'''


amin = create_reduction_func(
    'cupy_min',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_min(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)


amax = create_reduction_func(
    'cupy_max',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_max(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)


argmin = create_reduction_func(
    'cupy_argmin',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)


argmax = create_reduction_func(
    'cupy_argmax',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)


# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------

def array(obj, dtype=None, copy=True, ndmin=0):
    # TODO(beam2d): Support order and subok options
    if isinstance(obj, ndarray):
        if dtype is None:
            dtype = obj.dtype
        a = obj.astype(dtype, copy)

        ndim = a.ndim
        if ndmin > ndim:
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a
    else:
        a_cpu = numpy.array(obj, dtype=dtype, copy=False, ndmin=ndmin)
        if a_cpu.ndim > 0:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        a = ndarray(a_cpu.shape, dtype=a_cpu.dtype)
        a.data.copy_from_host(a_cpu.ctypes.data_as(ctypes.c_void_p), a.nbytes)
        if a_cpu.dtype == a.dtype:
            return a
        else:
            return a.view(dtype=a_cpu.dtype)


cpdef ndarray ascontiguousarray(ndarray a, dtype=None):
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = numpy.dtype(dtype)

    if dtype == a.dtype and a.flags.c_contiguous:
        return a
    else:
        newarray = ndarray(a.shape, dtype)
        elementwise_copy(a, newarray)
        return newarray

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------

cpdef ndarray rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=0):
    cdef Py_ssize_t ndim=a.ndim
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise ValueError('Axis out of range')
    if axis < start:
        start -= 1
    if axis == start:
        return a
    if ndim == 2:
        return a.transpose()

    axes = list(range(ndim))
    del axes[axis]
    axes.insert(start, axis)
    return a.transpose(axes)


cdef class broadcast:
    """Object that performs broadcasting.

    CuPy actually uses this class to support broadcasting in various
    operations. Note that this class does not provide an iterator.

    Args:
        arrays (tuple of arrays): Arrays to be broadcasted.

    Attributes:
        shape (tuple of ints): The broadcasted shape.
        nd (int): Number of dimensions of the broadcasted shape.
        size (int): Total size of the broadcasted shape.
        values (list of arrays): The broadcasted arrays.

    .. seealso:: :class:`numpy.broadcast`

    """

    cdef:
        readonly tuple values
        readonly tuple shape
        readonly object size
        readonly object nd

    def __init__(self, *arrays):
        rev = slice(None, None, -1)
        shape_arr = [a.shape[rev] for a in arrays
                     if isinstance(a, ndarray)]
        r_shape = [max(ss) for ss
                   in six.moves.zip_longest(*shape_arr, fillvalue=0)]

        self.shape = shape = tuple(r_shape[rev])
        self.size = size = internal_prod(shape)
        self.nd = ndim = len(shape)

        broadcasted = list(arrays)
        for i, a in enumerate(broadcasted):
            if not isinstance(a, ndarray):
                continue

            a_shape = a.shape
            if a_shape == shape:
                continue

            r_strides = [
                a_st if sh == a_sh else (0 if a_sh == 1 else None)
                for sh, a_sh, a_st
                in six_zip(r_shape, a.shape[rev], a.strides[rev])]

            if None in r_strides:
                raise ValueError('Broadcasting failed')

            offset = (0,) * (ndim - len(r_strides))

            broadcasted[i] = view = a.view()
            view._shape = shape
            view._strides = offset + tuple(r_strides[rev])
            view._size = size
            view._update_contiguity()

        self.values = tuple(broadcasted)


# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------

#elementeise
def _create_bit_op(name, op, no_bool, doc=''):
    types = () if no_bool else ('??->?',)
    return create_ufunc(
        'cupy_' + name,
        types + ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
                 'LL->L', 'qq->q', 'QQ->Q'),
        'out0 = in0 %s in1' % op,
        doc=doc)


bitwise_and = _create_bit_op(
    'bitwise_and', '&', False,
    '''Computes the bitwise AND of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_and`

    ''')


bitwise_or = _create_bit_op(
    'bitwise_or', '|', False,
    '''Computes the bitwise OR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_or`

    ''')


bitwise_xor = _create_bit_op(
    'bitwise_xor', '^', False,
    '''Computes the bitwise XOR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_xor`

    ''')


invert = create_ufunc(
    'cupy_invert',
    (('?->?', 'out0 = !in0'), 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'),
    'out0 = ~in0',
    doc='''Computes the bitwise NOT of an array elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.invert`

    ''')


left_shift = _create_bit_op(
    'left_shift', '<<', True,
    '''Shifts the bits of each integer element to the left.

    Only integer arrays are handled.

    .. seealso:: :data:`numpy.left_shift`

    ''')


right_shift = _create_bit_op(
    'right_shift', '>>', True,
    '''Shifts the bits of each integer element to the right.

    Only integer arrays are handled

    .. seealso:: :data:`numpy.right_shift`

    ''')

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------

_take_kernel = ElementwiseKernel(
    'raw T a, S indices, int64 cdim, int64 rdim',
    'T out',
    '''
      long long li = i / (rdim * cdim);
      long long ri = i % rdim;
      out = a[(li * cdim + indices) * rdim + ri];
    ''',
    'cupy_take')


def take(a, indices, axis=None, out=None):
    if axis is None:
        a = a.ravel()
        lshape = ()
        rshape = ()
    else:
        if axis >= a.ndim:
            raise ValueError('Axis overrun')
        lshape = a.shape[:axis]
        rshape = a.shape[axis + 1:]

    if numpy.isscalar(indices):
        a = rollaxis(a, axis)
        if out is None:
            return a[indices].copy()
        else:
            out[:] = a[indices]
            return out
    elif not isinstance(indices, ndarray):
        indices = array(indices, dtype=int)

    out_shape = lshape + indices.shape + rshape
    if out is None:
        out = ndarray(out_shape, dtype=a.dtype)
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')

    cdim = indices.size
    rdim = internal_prod(rshape)
    indices = indices.reshape(
        (1,) * len(lshape) + indices.shape + (1,) * len(rshape))
    return _take_kernel(a, indices, cdim, rdim, out)


def diagonal(a, offset=0, axis1=0, axis2=1):
    if axis1 < axis2:
        min_axis, max_axis = axis1, axis2
    else:
        min_axis, max_axis = axis2, axis1

    tr = list(six.moves.range(a.ndim))
    del tr[max_axis]
    del tr[min_axis]
    if offset >= 0:
        a = a.transpose(tr + [axis1, axis2])
    else:
        a = a.transpose(tr + [axis2, axis1])
        offset = -offset

    diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
    ret_shape = a.shape[:-2] + (diag_size,)
    if diag_size == 0:
        return ndarray(ret_shape, dtype=a.dtype)

    a = a[..., :diag_size, offset:offset + diag_size]

    ret = a.view()
    ret._shape = a.shape[:-2] + (diag_size,)
    ret._strides = a.strides[:-2] + (a.strides[-1] + a.strides[-2],)
    ret._size = internal_prod(ret._shape)
    ret._update_contiguity()
    return ret


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

def dot(a, b, out=None):
    a_ndim = a.ndim
    b_ndim = b.ndim
    assert a_ndim > 0 and b_ndim > 0
    a_is_vec = a_ndim == 1
    b_is_vec = b_ndim == 1

    if a_is_vec:
        a = a.reshape((1, a.size))
        a_ndim = 2
    if b_is_vec:
        b = b.reshape((b.size, 1))
        b_ndim = 2

    a_axis = a_ndim - 1
    b_axis = b_ndim - 2

    if a.shape[a_axis] != b.shape[b_axis]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = rollaxis(a, a_axis, 0)
    if b_axis:
        b = rollaxis(b, b_axis, 0)

    k = a.shape[0]
    m = b.size // k
    n = a.size // k

    ret_shape = a.shape[1:] + b.shape[1:]
    if out is None:
        if a_is_vec:
            ret_shape = () if b_is_vec else ret_shape[1:]
        elif b_is_vec:
            ret_shape = ret_shape[:-1]
    else:
        if out.size != n * m:
            raise ValueError('Output array has an invalid size')
        if not out.flags.c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    return _tensordot_core(a, b, out, n, m, k, ret_shape)


cpdef ndarray _tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t k, ret_shape):
    ret_dtype = a.dtype.char
    if ret_dtype != b.dtype.char:
        ret_dtype = numpy.find_common_type((ret_dtype, b.dtype), ()).char

    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    if not a.size or not b.size:
        if a.size or b.size:
            raise ValueError('cannot dot zero-sized and non-zero-sized arrays')
        if out is None:
            out = ndarray(ret_shape, dtype=ret_dtype)
        out.fill(0)
        return out

    if out is None:
        out = ndarray(ret_shape, dtype)
        if dtype == ret_dtype:
            ret = out
        else:
            ret = ndarray(ret_shape, ret_dtype)
    else:
        ret = out
        if out.dtype != dtype:
            out = ndarray(ret_shape, dtype)

    # It copies the operands if needed
    if a.shape != (k, n):
        a = a.reshape((k, n))
    if b.shape != (k, m):
        b = b.reshape((k, m))
    c = out
    if c.shape != (n, m):
        c = c.view()
        c.shape = (n, m)

    # Be careful that cuBLAS uses the FORTRAN-order matrix representation.
    if k == 1:
        if n == 1:
            # Scalar-vector product
            multiply(a, b, c)
        elif m == 1:
            # Scalar-vector product
            multiply(a.T, b, c)
        else:
            # Outer product A^T * B
            # c is C-contiguous while cuBLAS requires F-contiguous arrays, so
            # we compute C^T = B^T * A here.
            handle = device.get_cublas_handle()
            c.fill(0)
            a, inca = _to_cublas_vector(a, 1)
            b, incb = _to_cublas_vector(b, 1)
            if dtype == 'f':
                cublas.sger(handle, m, n, 1, b.data.ptr, incb, a.data.ptr,
                            inca, c.data.ptr, m)
            elif dtype == 'd':
                cublas.dger(handle, m, n, 1, b.data.ptr, incb, a.data.ptr,
                            inca, c.data.ptr, m)
        if dtype != ret_dtype:
            elementwise_copy(out, ret)
        return ret

    handle = device.get_cublas_handle()
    if n == 1:
        if m == 1:
            # Inner product
            a, inca = _to_cublas_vector(a, 0)
            b, incb = _to_cublas_vector(b, 0)
            mode = cublas.getPointerMode(handle)
            cublas.setPointerMode(handle,
                                  cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                if dtype == 'f':
                    cublas.sdot(handle, k, a.data.ptr, inca, b.data.ptr, incb,
                                c.data.ptr)
                elif dtype == 'd':
                    cublas.ddot(handle, k, a.data.ptr, inca, b.data.ptr, incb,
                                c.data.ptr)
            finally:
                cublas.setPointerMode(handle, mode)
        else:
            # Matrix-vector product B^T * A
            a, inca = _to_cublas_vector(a, 0)
            b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
            if transb:
                # gemv requires (m, k) as the original matrix dimensions
                # rather than the transposed dimensions.
                m, k = k, m
            if dtype == 'f':
                cublas.sgemv(handle, transb, m, k, 1, b.data.ptr, ldb,
                             a.data.ptr, inca, 0, c.data.ptr, 1)
            elif dtype == 'd':
                cublas.dgemv(handle, transb, m, k, 1, b.data.ptr, ldb,
                             a.data.ptr, inca, 0, c.data.ptr, 1)
    elif m == 1:
        # Matrix-vector product A^T * B
        a, transa, lda = _mat_to_cublas_contiguous(a, 1)
        b, incb = _to_cublas_vector(b, 0)
        if transa:
            # gemv requires (n, k) as the original matrix dimensions rather
            # than the transposed dimensions.
            n, k = k, n
        if dtype == 'f':
            cublas.sgemv(handle, transa, n, k, 1, a.data.ptr, lda, b.data.ptr,
                         incb, 0, c.data.ptr, 1)
        elif dtype == 'd':
            cublas.dgemv(handle, transa, n, k, 1, a.data.ptr, lda, b.data.ptr,
                         incb, 0, c.data.ptr, 1)
    else:
        # Matrix-Matrix product A^T * B
        # c is C-contiguous while cuBLAS assumes F-contiguous inputs, so we
        # compute C^T = B^T * A here.
        a, transa, lda = _mat_to_cublas_contiguous(a, 0)
        b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
        if dtype == 'f':
            cublas.sgemm(handle, transb, transa, m, n, k, 1, b.data.ptr, ldb,
                         a.data.ptr, lda, 0, c.data.ptr, m)
        elif dtype == 'd':
            cublas.dgemm(handle, transb, transa, m, n, k, 1, b.data.ptr, ldb,
                         a.data.ptr, lda, 0, c.data.ptr, m)

    if dtype != ret_dtype:
        elementwise_copy(out, ret)
    return ret


cpdef tuple _mat_to_cublas_contiguous(ndarray a, int trans):
    assert a.ndim == 2
    f = a.flags
    if f.f_contiguous:
        return a, trans, a.strides[1] // a.itemsize
    if not f.c_contiguous:
        a = a.copy()
    return a, 1 - trans, a.strides[0] // a.itemsize


cpdef tuple _to_cublas_vector(ndarray a, int rundim):
    if a.strides[rundim] < 0:
        return a.copy(), 1
    else:
        return a, a.strides[rundim] // a.itemsize

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------

def create_comparison(name, op, doc=''):
    return create_ufunc(
        'cupy_' + name,
        ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
         'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?'),
        'out0 = in0 %s in1' % op,
        doc=doc)


greater = create_comparison(
    'greater', '>',
    '''Tests elementwise if ``x1 > x2``.

    .. seealso:: :data:`numpy.greater`

    ''')


greater_equal = create_comparison(
    'greater_equal', '>=',
    '''Tests elementwise if ``x1 >= x2``.

    .. seealso:: :data:`numpy.greater_equal`

    ''')


less = create_comparison(
    'less', '<',
    '''Tests elementwise if ``x1 < x2``.

    .. seealso:: :data:`numpy.less`

    ''')


less_equal = create_comparison(
    'less_equal', '<=',
    '''Tests elementwise if ``x1 <= x2``.

    .. seealso:: :data:`numpy.less_equal`

    ''')


equal = create_comparison(
    'equal', '==',
    '''Tests elementwise if ``x1 == x2``.

    .. seealso:: :data:`numpy.equal`

    ''')


not_equal = create_comparison(
    'not_equal', '!=',
    '''Tests elementwise if ``x1 != x2``.

    .. seealso:: :data:`numpy.equal`

    ''')


_all = create_reduction_func(
    'cupy_all',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a & b', 'out0 = a', 'bool'),
    'true', '')


_any = create_reduction_func(
    'cupy_any',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a | b', 'out0 = a', 'bool'),
    'false', '')


# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------

_sum = create_reduction_func(
    'cupy_sum',
    ('?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a', None), 0)


_prod = create_reduction_func(
    'cupy_prod',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'],
    ('in0', 'a * b', 'out0 = a', None), 1)


def create_arithmetic(name, op, boolop, doc):
    return create_ufunc(
        'cupy_' + name,
        (('??->?', 'out0 = in0 %s in1' % boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
        'out0 = in0 %s in1' % op,
        doc=doc)


add = create_arithmetic(
    'add', '+', '|',
    '''Adds two arrays elementwise.

    .. seealso:: :data:`numpy.add`

    ''')


negative = create_ufunc(
    'cupy_negative',
    (('?->?', 'out0 = !in0'),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    'out0 = -in0',
    doc='''Takes numerical negative elementwise.

    .. seealso:: :data:`numpy.negative`

    ''')


multiply = create_arithmetic(
    'multiply', '*', '&',
    '''Multiplies two arrays elementwise.

    .. seealso:: :data:`numpy.multiply`

    ''')


divide = create_ufunc(
    'cupy_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 / in1'),
     ('ff->f', 'out0 = in0 / in1'),
     ('dd->d', 'out0 = in0 / in1')),
    'out0 = in1 == 0 ? 0 : floor((double)in0 / (double)in1)',
    doc='''Divides arguments elementwise.

    .. seealso:: :data:`numpy.divide`

    ''')


power = create_ufunc(
    'cupy_power',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = powf(in0, in1)'),
     ('ff->f', 'out0 = powf(in0, in1)'),
     ('dd->d', 'out0 = pow(in0, in1)')),
    'out0 = rint(pow((double)in0, (double)in1))',
    doc='''Computes ``x1 ** x2`` elementwise.

    .. seealso:: :data:`numpy.power`

    ''')


subtract = create_arithmetic(
    'subtract', '-', '^',
    '''Subtracts arguments elementwise.

    .. seealso:: :data:`numpy.subtract`

    ''')


true_divide = create_ufunc(
    'cupy_true_divide',
    ('bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d', 'LL->d',
     'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = (out0_type)in0 / (out0_type)in1',
    doc='''Elementwise true division (i.e. division as floating values).

    .. seealso:: :data:`numpy.true_divide`

    ''')


if six.PY3:
    divide = true_divide


floor_divide = create_ufunc(
    'cupy_floor_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = _floor_divide(in0, in1)',
    doc='''Elementwise floor division (i.e. integer quotient).

    .. seealso:: :data:`numpy.floor_divide`

    ''')


remainder = create_ufunc(
    'cupy_remainder',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('ff->f', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('dd->d', 'out0 = in0 - _floor_divide(in0, in1) * in1')),
    'out0 = (in0 - _floor_divide(in0, in1) * in1) * (in1 != 0)',
    doc='''Computes the remainder of Python division elementwise.

    .. seealso:: :data:`numpy.remainder`

    ''')


absolute = create_ufunc(
    'cupy_absolute',
    (('?->?', 'out0 = in0'),
     'b->b', ('B->B', 'out0 = in0'), 'h->h', ('H->H', 'out0 = in0'),
     'i->i', ('I->I', 'out0 = in0'), 'l->l', ('L->L', 'out0 = in0'),
     'q->q', ('Q->Q', 'out0 = in0'),
     ('e->e', 'out0 = fabsf(in0)'),
     ('f->f', 'out0 = fabsf(in0)'),
     ('d->d', 'out0 = fabs(in0)')),
    'out0 = in0 > 0 ? in0 : -in0',
    doc='''Elementwise absolute value function.

    .. seealso:: :data:`numpy.absolute`

    ''')


# Fixed version of sqrt
sqrt_fixed = create_ufunc(
    'cupy_sqrt',
    ('e->e', 'f->f', 'd->d'),
    'out0 = sqrt(in0)')


_clip = create_ufunc(
    'cupy_clip',
    ('???->?', 'bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I',
     'lll->l', 'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'),
    'out0 = min(in2, max(in1, in0))')

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if axis is None:
        axis = tuple(range(a.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)

    if dtype is None and issubclass(a.dtype.type,
                                    (numpy.integer, numpy.bool_)):
        dtype = 'd'

    shape = a.shape
    items = 1
    for ax in axis:
        items *= shape[ax]
    alpha = 1. / max(items - ddof, 0)
    arrmean = a.mean(axis=axis, dtype=dtype, keepdims=True)
    if out is None:
        return _var_core(a, arrmean, alpha, axis=axis, keepdims=keepdims)
    else:
        return _var_core_out(
            a, arrmean, alpha, out, axis=axis, keepdims=keepdims)


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return sqrt_fixed(ret, dtype=dtype, out=out)


_var_core = ReductionKernel(
    'S x, T mean, T alpha', 'T out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')
_var_core_out = ReductionKernel(
    'S x, T mean, T alpha', 'U out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')

# TODO(okuta) needs cast
_mean = create_reduction_func(
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a / (_in_ind.size() / _out_ind.size())', None))
