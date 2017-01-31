import heapq

import six

import chainer
from chainer import function
from chainer import variable

_var_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
_func_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}


class DotNode(object):
    """Node of the computational graph, with utilities for dot language.

    This class represents a node of computational graph,
    with some utilities for dot language.

    """

    def __init__(self, node, attribute=None):
        """Initializes DotNode.

        Args:
            node: :class: `Variable` object or :class: `Function` object.
            attribute (dict): Attributes for the node.

        """
        assert isinstance(node, (variable.Variable, function.Function))
        self.node = node
        self.id_ = id(node)
        self.attribute = {'label': node.label}
        if isinstance(node, variable.Variable):
            self.attribute.update({'shape': 'oval'})
        else:
            self.attribute.update({'shape': 'box'})
        if attribute is not None:
            self.attribute.update(attribute)

    @property
    def label(self):
        """The text that represents properties of the node.

        Returns:
            string: The text that represents the id and attributes of this
                node.
        """

        attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                      in self.attribute.items()]
        return "%s [%s];" % (self.id_, ",".join(attributes))


class ComputationalGraph(object):

    """Class that represents computational graph.

    .. note::

      We assume that the computational graph is directed and acyclic.

    """

    def __init__(self, nodes, edges, variable_style=None, function_style=None,
                 rankdir='TB'):
        """Initializes computational graph.

        Args:
            nodes (list): List of nodes. Each node is either
                 :class:`Variable` object or :class:`Function` object.
            edges (list): List of edges. Each edge consists of pair of nodes.
            variable_style (dict): Dot node style for variable.
            function_style (dict): Dot node style for function.
            rankdir (str): Direction of the graph that must be
                TB (top to bottom), BT (bottom to top), LR (left to right)
                or RL (right to left).

        """
        self.nodes = nodes
        self.edges = edges
        self.variable_style = variable_style
        self.function_style = function_style
        if rankdir not in ('TB', 'BT', 'LR', 'RL'):
            raise ValueError('rankdir must be in TB, BT, LR or RL.')
        self.rankdir = rankdir

    def _to_dot(self):
        """Converts graph in dot format.

        `label` property of is used as short description of each node.
        Returns:
            str: The graph in dot format.

        """
        ret = 'digraph graphname{rankdir=%s;' % self.rankdir
        for node in self.nodes:
            assert isinstance(node, (variable.Variable, function.Function))
            if isinstance(node, variable.Variable):
                ret += DotNode(node, self.variable_style).label
            else:
                ret += DotNode(node, self.function_style).label
        for edge in self.edges:
            head, tail = edge
            if (isinstance(head, variable.Variable) and
                    isinstance(tail, function.Function)):
                head_attr = self.variable_style
                tail_attr = self.function_style
            elif (isinstance(head, function.Function) and
                  isinstance(tail, variable.Variable)):
                head_attr = self.function_style
                tail_attr = self.variable_style
            else:
                raise TypeError(
                    'head and tail should be the set of Variable and Function')
            head_node = DotNode(head, head_attr)
            tail_node = DotNode(tail, tail_attr)
            ret += "%s -> %s;" % (head_node.id_, tail_node.id_)
        ret += "}"
        return ret

    def dump(self, format='dot'):
        """Dumps graph as a text.

        Args:
            format(str): The graph language name of the output.
            Currently, it must be 'dot'.

        Returns:
            str: The graph in specified format.

        """
        if format == 'dot':
            return self._to_dot()
        else:
            NotImplementedError('Currently, only dot format is supported.')


def build_computational_graph(
        outputs, remove_split=True, variable_style=_var_style,
        function_style=_func_style, rankdir='TB'):
    """Builds a graph of functions and variables backward-reachable from outputs.

    Args:
        outputs(list): nodes from which the graph is constructed.
            Each element of outputs must be either :class:`Variable`
            object or :class:`Function` object.
        remove_split(bool): It must be ``True``. This argument is left for
            backward compatibility.
        variable_style(dict): Dot node style for variable.
            Possible keys are 'shape', 'color', 'fillcolor', 'style', and etc.
        function_style(dict): Dot node style for function.
        rankdir (str): Direction of the graph that must be
            TB (top to bottom), BT (bottom to top), LR (left to right)
            or RL (right to left).

    Returns:
        ComputationalGraph: A graph consisting of nodes and edges that
        are backward-reachable from at least one of ``outputs``.

        If ``unchain_backward`` was called in some variable in the
        computational graph before this function, backward step is
        stopped at this variable.

        For example, suppose that computational graph is as follows::

                |--> f ---> y
            x --+
                |--> g ---> z

        Let ``outputs = [y, z]``.
        Then the full graph is emitted.

        Next, let ``outputs = [y]``. Note that ``z`` and ``g``
        are not backward-reachable from ``y``.
        The resulting graph would be following::

            x ---> f ---> y

        See :class:`TestGraphBuilder` for details.

    """
    if not remove_split:
        raise ValueError('remove_split=False is not supported anymore')

    cands = []
    seen_edges = set()
    nodes = set()
    push_count = [0]

    # This class is for object that has not been implemented __eq__
    class HashableObject(object):

        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return self.v.__hash__()

        def __eq__(self, r):
            return self.v is r.v

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        add_cand(o)
        nodes.add(HashableObject(o))

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, variable.Variable):
            creator = cand.creator
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                nodes.add(HashableObject(creator))
                nodes.add(HashableObject(cand))
        elif isinstance(cand, function.Function):
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    nodes.add(HashableObject(input_))
                    nodes.add(HashableObject(cand))
    return ComputationalGraph(list(i.v for i in nodes), list(seen_edges),
                              variable_style, function_style, rankdir)


def build_hierarchical_computational_graph(
        outputs, model, variable_style=_var_style, function_style=_func_style,
        rankdir='TB', draw_variable=True):
    assert isinstance(model, chainer.Chain)

    def get_parent(name):
        return '/'.join(name.split('/')[:-1])

    nodenames = dict((p, n) for n, p in model.namedparams())
    nodegroup = {}
    cg = build_computational_graph(outputs)
    for node in cg.nodes:
        # Parameters
        if isinstance(node, variable.Variable) and node in nodenames:
            nodegroup[node] = get_parent(nodenames[node])
        # Determine parametric Function's group from parameter variables
        elif isinstance(node, function.Function):
            for input_var in node.inputs:
                if input_var in nodenames:
                    nodegroup[node] = get_parent(nodenames[input_var])
                    # Set output variables' group same as the function's one
                    for output_var in node.outputs:
                        nodegroup[output_var()] = nodegroup[node]
                    break

    for node in cg.nodes:
        # Non-parametric Function
        if node not in nodegroup and isinstance(node, function.Function):
            input_var_name = None
            for input_var in node.inputs:
                if input_var in nodegroup:
                    input_var_name = nodegroup[input_var]
            if input_var_name is not None:
                nodegroup[node] = get_parent(input_var_name)
                # Output variables of non-parametric Function
                for output_var in node.outputs:
                    nodegroup[output_var()] = nodegroup[node]
            else:
                print(node, 'None!!')

    subgraphs = {}
    for var_or_func, subgraph in six.iteritems(nodegroup):
        parts = [n for n in subgraph.split('/') if n]
        if parts:
            leaf = subgraphs
            for p in parts[:-1]:
                if p not in leaf:
                    leaf[p] = {}
                leaf = leaf[p]
            if parts[-1] not in leaf:
                leaf[parts[-1]] = {}
            leaf[parts[-1]][var_or_func] = {}
        else:
            subgraphs[var_or_func] = {}

    def dot_subgraph(subgraph_name, subgraph_dict):
        ret = 'subgraph cluster_%s{label=%s;\n' % (
            id(subgraph_dict), subgraph_name)
        for node in subgraph_dict.keys():
            if isinstance(node, str):
                ret += dot_subgraph(node, subgraph_dict[node])
            else:
                if isinstance(node, variable.Variable):
                    if draw_variable:
                        ret += DotNode(node, variable_style).label + '\n'
                elif isinstance(node, function.Function):
                    ret += DotNode(node, function_style).label + '\n'
                else:
                    raise ValueError('{}'.format(node))
        ret += '}\n'
        return ret

    ret = 'digraph graphname{rank=same;rankdir=%s;\n' % rankdir
    for subgraph_name, subgraph_dict in six.iteritems(subgraphs):
        ret += dot_subgraph(subgraph_name, subgraph_dict)
        if isinstance(subgraph_name, variable.Variable):
            ret += DotNode(subgraph_name, variable_style).label + '\n'
        elif isinstance(subgraph_name, function.Function):
            ret += DotNode(subgraph_name, function_style).label + '\n'

    if not draw_variable:
        for edge_i, edge in enumerate(cg.edges):
            head, tail = edge
            if isinstance(head, variable.Variable):
                if head.creator is not None:
                    head = head.creator
            if isinstance(tail, variable.Variable):
                for node in cg.nodes:
                    if isinstance(node, function.Function):
                        for input_var in node.inputs:
                            if id(input_var) == id(tail):
                                tail = node
                                break
                        if isinstance(tail, function.Function):
                            break
            cg.edges[edge_i] = head, tail

    drawn_edges = []
    for edge in cg.edges:
        head, tail = edge
        if (id(head), id(tail)) in drawn_edges:
            continue
        drawn_edges.append((id(head), id(tail)))

        if (isinstance(head, variable.Variable) and
                isinstance(tail, function.Function)):
            head_attr = variable_style
            tail_attr = function_style
        elif (isinstance(head, function.Function) and
              isinstance(tail, variable.Variable)):
            head_attr = function_style
            tail_attr = variable_style
        else:
            if draw_variable:
                raise TypeError(
                    'head and tail should be the set of Variable and Function')
            else:
                head_attr = function_style
                tail_attr = function_style
        if not draw_variable:
            if isinstance(head, variable.Variable) \
                    or isinstance(tail, variable.Variable):
                continue
        head_node = DotNode(head, head_attr)
        tail_node = DotNode(tail, tail_attr)
        ret += "%s -> %s;\n" % (head_node.id_, tail_node.id_)
    ret += "}\n"
    return ret
