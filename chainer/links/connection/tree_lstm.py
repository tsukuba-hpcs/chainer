from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.activation import tree_lstm
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.links.connection import linear

from chainer import cuda
from chainer import link
from chainer import utils
from chainer import variable


class ChildSumTreeLSTM(link.Chain):
    """Child-Sum TreeLSTM unit.

    This is a Child-Sum TreeLSTM unit as a chain.
    This link is a variable arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, c,
    and the output, h, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    For example, this link is called such as
    ``func(c1, c2, h1, h2, x)`` if the number of children nodes is 2,
    while ``func(c1, c2, c3, h1, h2, h3, x)`` if that is 3.
    This function is independent from an order of children nodes.
    Thus, the returns of ``func(c1, c2, h1, h2, x)`` equal to
    those of ``func(c2, c1, h2, h1, x)``.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h_aio (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`) and summation of children's
            output vectors. :math:`a`, :math:`i` and :math:`o` denotes
            input compound (:math:`u` in the paper),
            input gate and output gate, respectively.
        W_h_f (chainer.links.Linear): Linear layer of connections between
            :math:`f` and the output of each child.

    See the paper for details: `Improved Semantic Representations From \
    Tree-Structured Long Short-Term Memory Networks \
    <http://www.aclweb.org/anthology/P15-1150>`_.

    """

    def __init__(self, in_size, out_size):
        super(ChildSumTreeLSTM, self).__init__(
            W_x=linear.Linear(in_size, 4 * out_size),
            W_h_aio=linear.Linear(out_size, 3 * out_size, nobias=True),
            W_h_f=linear.Linear(out_size, out_size, nobias=True),
        )
        self.in_size = in_size
        self.state_size = out_size
        utils.experimental('chainer.links.tree_lstm.py')

    def __call__(self, *cshsx):
        """Returns new cell state and output of Child-Sum TreeLSTM.

        Args:
            cshsx (list of ~chainer.Variable): Variable arguments which include
                all cell vectors and all output vectors of variable children,
                and an input vector.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state vector, and ``h_new`` is
                new output vector.

        """

        cs = cshsx[:len(cshsx) // 2]
        hs = cshsx[len(cshsx) // 2:-1]
        x = cshsx[-1]

        if x is None:
            x = self.xp.zeros(
                (cs[0].shape[0], self.in_size), dtype=cs[0].dtype)

        W_x_in = self.W_x(x)
        W_x_aio_in, W_x_f_in = split_axis.split_axis(
            W_x_in, [3 * self.state_size], axis=1)

        if any(h is None for h in hs):
            hs = list(hs)
            for i, h in enumerate(hs, start=1):
                if h is not None:
                    hs[i] = self.xp.zeros(
                        (cs[0].shape[0], self.in_size), dtype=cs[0].dtype)

        aio_in = self.W_h_aio(sum(hs)) + W_x_aio_in
        W_h_fs_in = concat.concat(split_axis.split_axis(
            self.W_h_f(concat.concat(hs, axis=0)), len(hs), axis=0), axis=1)
        f_in = W_h_fs_in + \
            concat.concat([W_x_f_in] * len(hs), axis=1)
        tree_lstm_in = concat.concat([aio_in, f_in], axis=1)

        if any(c is None for c in cs):
            cs = list(cs)
            for i, c in enumerate(cs):
                if c is None:
                    xp = self.xp
                    with cuda.get_device(self._device_id):
                        cs[i] = variable.Variable(
                            xp.zeros((x.shape[0], self.state_size),
                                     dtype=x.dtype))
            cs = tuple(cs)

        return tree_lstm.tree_lstm(*(cs + (tree_lstm_in, )))


class NaryTreeLSTM(link.Chain):
    """N-ary TreeLSTM unit.

    This is a N-ary TreeLSTM unit as a chain.
    This link is a fixed-length arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, c,
    and the output, h, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    For example, this link is called such as
    ``func(c1, c2, h1, h2, x)`` if the number of children nodes
    was set 2 (``n_ary = 2``), while
    ``func(c1, c2, c3, h1, h2, h3, x)`` if that was 3
    (``n_ary = 3``).
    This function is dependent from an order of children nodes
    unlike Child-Sum TreeLSTM.
    Thus, the returns of ``func(c1, c2, h1, h2, x)`` are
    different from those of ``func(c2, c1, h2, h1, x)``.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.
        n_ary (int): The number of children nodes in a tree structure.

    Attributes:
        W_x (chainer.links.Linear): Linear layer of
            connections from input vectors.
        W_h (chainer.links.Linear): Linear layer of connections between
            (:math:`a`, :math:`i`, :math:`o`, all :math:`f`s)
            and the output of each child
            :math:`a`, :math:`i`, :math:`o` and :math:`f` denotes input
            compound, input gate, output gate and forget gate, respectively.
            :math:`a`, input compound, equals to :math:`u` in
            the paper by Tai et al.

    See the papers for details: `Improved Semantic Representations From \
    Tree-Structured Long Short-Term Memory Networks \
    <http://www.aclweb.org/anthology/P15-1150>`_, and
    `A Fast Unified Model for Parsing and Sentence Understanding \
    <https://arxiv.org/pdf/1603.06021.pdf>`_.

    Tai et al.'s N-Ary TreeLSTM is little extended in
    Bowman et al., and this link is based on
    the variant by Bowman et al.
    Specifically, eq. 10 in Tai et al. only has one W matrix
    to be applied to x, consistently for all children.
    On the other hand, Bowman et al.'s model has multiple matrices,
    each of which affects the forget gate for each child's cell individually.

    """

    def __init__(self, in_size, out_size, n_ary=2):
        assert(n_ary >= 2)
        super(NaryTreeLSTM, self).__init__(
            W_x=linear.Linear(in_size, (3 + n_ary) * out_size),
        )
        for i in range(1, n_ary + 1):
            self.add_link(
                'W_h{}'.format(i),
                linear.Linear(out_size, (3 + n_ary) * out_size, nobias=True))
        self.in_size = in_size
        self.state_size = out_size
        self.n_ary = n_ary
        utils.experimental('chainer.links.tree_lstm.py')

    def __call__(self, *cshsx):
        """Returns new cell state and output of N-ary TreeLSTM.

        Args:
            cshsx (list of ~chainer.Variable): Arguments which include all cell
                vectors and all output vectors of fixed-length children,
                and an input vector. The number of arguments must be same
                as ``n_ary * 2 + 1``.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state vector, and ``h_new`` is
                new output vector.

        """

        assert(len(cshsx) == self.n_ary * 2 + 1)
        cs = cshsx[:self.n_ary]
        hs = cshsx[self.n_ary:-1]
        x = cshsx[-1]

        if x is None:
            x = self.xp.zeros(
                (cs[0].shape[0], self.in_size), dtype=cs[0].dtype)

        tree_lstm_in = self.W_x(x)

        for i, h in enumerate(hs, start=1):
            if h is not None:
                tree_lstm_in += getattr(self, 'W_h{}'.format(i))(h)

        if any(c is None for c in cs):
            cs = list(cs)
            for i, c in enumerate(cs):
                if c is None:
                    xp = self.xp
                    with cuda.get_device(self._device_id):
                        cs[i] = variable.Variable(
                            xp.zeros((x.shape[0], self.state_size),
                                     dtype=x.dtype))
            cs = tuple(cs)
        return tree_lstm.tree_lstm(*(cs + (tree_lstm_in, )))
