# %%
from pydantic.dataclasses import dataclass

from dsl import substr
from common import str_to_id
from input_data_graph import Node as GraphNode, InputDataGraph, gen_data_input_graph


@dataclass(frozen=True)
class ConstantPosDagExpr:
    k: int

    def __repr__(self):
        return f"ConstantPos({self.k})"


@dataclass(frozen=True)
class ConstantStrDagExpr:
    s: str

    def __repr__(self):
        return f'ConstantStr("{self.s}")'


@dataclass(frozen=True)
class SubStrDagExpr:
    s: str
    lexprs: frozenset[ConstantPosDagExpr | GraphNode]
    rexprs: frozenset[ConstantPosDagExpr | GraphNode]

    def __repr__(self):
        i = str_to_id(self.s)
        ls = ", ".join(sorted([str(e) for e in self.lexprs]))
        rs = ", ".join(sorted([str(e) for e in self.rexprs]))
        return f"SubStr(v{i}, {{{ls}}}, {{{rs}}})"


@dataclass(frozen=True)
class DagNode:
    id: int

    def __repr__(self):
        return f"Node({self.id})"


@dataclass(frozen=True)
class DagEdge:
    n1: DagNode
    n2: DagNode

    def __repr__(self):
        return f"Edge({self.n1.id}, {self.n2.id})"


@dataclass(frozen=True)
class Dag:
    nodes: set[DagNode]
    start_node: DagNode
    final_node: DagNode
    edges: set[DagEdge]
    W: dict[DagEdge, set[SubStrDagExpr | ConstantStrDagExpr]]


def gen_substr_expr(s: str, lpos: int, rpos: int, G: InputDataGraph) -> SubStrDagExpr:
    str_id = str_to_id(s)
    lexprs = {ConstantPosDagExpr(lpos)}
    rexprs = {ConstantPosDagExpr(rpos)}
    for v in G.V:
        for label in G.I[v]:
            if label.str_id == str_id:
                if label.index == lpos:
                    lexprs.add(v)
                if label.index == rpos:
                    rexprs.add(v)
    return SubStrDagExpr(s, frozenset(lexprs), frozenset(rexprs))


def gen_dag(in_str: str, out_str: str, G: InputDataGraph) -> Dag:
    nodes: set[DagNode] = set()
    edges: set[DagEdge] = set()
    W: dict[DagEdge, set] = dict()

    # Add nodes
    start_node = DagNode(0)
    final_node = DagNode(len(out_str))
    nodes.add(start_node)
    nodes.add(final_node)
    for i in range(1, len(out_str) + 1):
        nodes.add(DagNode(i))

    # Iterate over substrings
    for i in range(0, len(out_str)):
        for j in range(i + 1, len(out_str) + 1):
            edge = DagEdge(DagNode(i), DagNode(j))
            edges.add(edge)
            out_ss = substr(out_str, i + 1, j + 1)
            W[edge] = {ConstantStrDagExpr(out_ss)}

            for l in range(1, len(in_str) + 1):
                for r in range(l + 1, len(in_str) + 2):
                    in_ss = substr(in_str, l, r)
                    if in_ss == out_ss:
                        ss_expr = gen_substr_expr(in_str, l, r, G)
                        W[edge].add(ss_expr)

    return Dag(nodes, start_node, final_node, edges, W)


inputs = ["Mumbai, India"]
outputs = ["India"]
G = gen_data_input_graph(inputs)
dag = gen_dag(inputs[0], outputs[0], G)
for edge, exprs in dag.W.items():
    exprs = sorted(list(exprs), key=lambda e: str(e))
    exprs = ", ".join([e.__repr__() for e in exprs])
    print(f"W({edge.n1.id}, {edge.n2.id}) = {{{exprs}}}")
