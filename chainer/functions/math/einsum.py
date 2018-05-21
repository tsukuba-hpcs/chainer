import warnings

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


def _enumerate_axes(subscripts):
    if '@' in subscripts:
        left_sub, right_sub = subscripts.split('@')
        for i, s in enumerate(left_sub):
            yield i, s
        yield slice(len(left_sub), -len(right_sub) or None), '@'
        for i, s in enumerate(right_sub):
            yield i - len(right_sub), s
    else:
        for i, s in enumerate(subscripts):
            yield i, s


def _einsum(xp, dtype, in_subscripts, out_subscript, *inputs,
            check_controversial_sum=False):
    if '@' in in_subscripts and '@' not in out_subscript:
        # einsum does not usually allow summing over '...'
        subscripts = '{}->...{}'.format(
            in_subscripts.replace('@', '...'),
            out_subscript
        )
        y = xp.einsum(subscripts, *inputs)
        sum_ndim = y.ndim - len(out_subscript)
        if check_controversial_sum and sum_ndim > 0:
            warnings.warn(UserWarning(
                "einsum should not support summing over Ellipsis, "
                "while NumPy 1.14 sometimes accidentally supports it. "
                "This feature will be deleted in a future version "
                "of Chainer. See also NumPy issues #10926, #9984."
            ))
        y = xp.sum(y, axis=tuple(range(sum_ndim)))
    elif in_subscripts == out_subscript:
        # Avoid cupy issue #1179
        y, = inputs
    else:
        subscripts = '{}->{}'.format(
            in_subscripts,
            out_subscript
        ).replace('@', '...')
        y = xp.einsum(subscripts, *inputs)
    return utils.force_array(y, dtype)


class EinSum(function_node.FunctionNode):

    def __init__(self, in_subs, out_sub):
        self.in_subs = in_subs
        self.out_sub = out_sub

    def check_type_forward(self, in_types):
        for in_type in in_types:
            type_check.expect(in_type.dtype.kind == 'f')

        in_subs = self.in_subs.split(',')
        type_check.expect(in_types.size() == len(in_subs))

        shape_dict = {}
        for in_sub, in_type in zip(in_subs, in_types):
            for axis, char in _enumerate_axes(in_sub):
                shape = in_type.shape[axis]
                if char in shape_dict:
                    type_check.expect(shape_dict[char] == shape)
                else:
                    shape_dict[char] = shape

    def forward(self, inputs):
        n_args = len(inputs)
        # TODO(kataoka): Do not retain inputs if n_args == 1
        self.retain_inputs(tuple(range(n_args)))

        xp = cuda.get_array_module(inputs[0])
        dtype = xp.result_type(*[x.dtype for x in inputs])
        y = _einsum(xp, dtype, self.in_subs, self.out_sub, *inputs,
                    check_controversial_sum=True)
        return y,

    def backward(self, indices, grad_outputs):
        inputs = self.get_retained_inputs()
        g, = grad_outputs

        fwd_in_subs = self.in_subs.split(',')
        fwd_out_sub = self.out_sub
        return tuple(
            DiagEinSum(
                in_subs=','.join([
                    (fwd_out_sub if j == i else s)
                    for j, s in enumerate(fwd_in_subs)
                ]),
                out_sub=fwd_in_subs[i],
                out_shape=inputs[i].shape,
            ).apply(tuple(
                (g if j == i else x)
                for j, x in enumerate(inputs)
            ))[0]
            for i in indices
        )


class DiagEinSum(EinSum):

    def __init__(self, in_subs, out_sub, out_shape):
        self.in_subs = in_subs
        self.out_sub = out_sub
        self.out_shape = out_shape

    def forward(self, inputs):
        n_args = len(inputs)
        # TODO(kataoka): Do not retain inputs if n_args == 1
        self.retain_inputs(tuple(range(n_args)))

        xp = cuda.get_array_module(inputs[0])
        dtype = xp.result_type(*[x.dtype for x in inputs])

        out_set = set(self.out_sub)

        # '@' is a single char, ',' is excluded.
        io_set = out_set.intersection(set(self.in_subs))

        if len(io_set) == len(self.out_sub):
            y = _einsum(xp, dtype, self.in_subs, self.out_sub, *inputs)
        else:
            direct_sub = []
            inverse_sub = []
            expander = []
            for c in sorted(out_set):
                if c in io_set:
                    direct_sub.append(c)
                    expander.append(slice(None))
                else:
                    expander.append(None)
                inverse_sub.append(c)

            y = xp.zeros(self.out_shape, dtype)
            diag_y = _einsum(
                xp, dtype, self.out_sub, ''.join(inverse_sub), y)
            # Make the view writeable as numpy PR #5410 for numpy<1.10.
            diag_y.setflags(write=True)
            diag_y[...] = _einsum(
                xp, dtype, self.in_subs, ''.join(direct_sub), *inputs
            )[expander]
        return y,


def einsum(*operands):
    input_subscripts, output_subscript, ioperands = \
        _parse_einsum_input(operands)
    return EinSum(
        in_subs=input_subscripts,
        out_sub=output_subscript,
    ).apply(ioperands)[0]


# #################### cupy.linalg.einsum ####################
# From cupy PR #873

einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
einsum_symbols_set = set(einsum_symbols)


def _parse_einsum_input(operands):
    """Parses einsum operands.

    This function is based on `numpy.core.einsumfunc._parse_einsum_input`
    function in NumPy 1.14.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('@a,@a', '@', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('@a,@a', '@', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

        # Check for proper "->"
        if ("-" in subscripts) or (">" in subscripts):
            if any((
                    subscripts.count("-") > 1,
                    subscripts.count(">") > 1,
                    subscripts.count("->") != 1,
            )):
                raise ValueError("Subscripts can only contain one '->'.")

        # Parse "..."
        subscripts = subscripts.replace("...", "@")
        if "." in subscripts:
            raise ValueError("Invalid Ellipses.")

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "@"
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "@"
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")

        # Make sure output subscripts are in the input
        for char in output_subscript:
            if char not in input_subscripts:
                raise ValueError(
                    "Output character %s did not appear in the input"
                    % ('...' if char == '@' else char))

    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s == '@' or tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)
