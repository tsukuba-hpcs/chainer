import numpy
import six


class DatasetMixin(object):

    """Default implementation of dataset indexing.

    DatasetMixin provides the :meth:`__getitem__` operator. The default
    implementation uses :meth:`get_example` to extract each example, and
    combines the results into a list. This mixin makes it easy to implement a
    new dataset that does not support efficient slicing.

    Dataset implementation using DatasetMixin still has to provide the
    :meth:`__len__` operator explicitly.

    """

    def __getitem__(self, index):
        """Returns an example or a sequence of examples.

        It implements the standard Python indexing and numpy like advanced
        indexing. It uses the :meth:`get_example` method by default, but it may
        be overridden by the implementation to, for example, improve the
        slicing performance.

        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError
