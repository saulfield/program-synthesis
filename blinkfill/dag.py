# %%
from itertools import product
from typing import TypeAlias

import dsl
from common import str_to_id
from dsl import substr
from input_data_graph import InputDataGraph
from input_data_graph import Node as GraphNode
from input_data_graph import Edge as GraphEdge
from input_data_graph import gen_data_input_graph
from pydantic.dataclasses import dataclass


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


PosExprSet: TypeAlias = frozenset[ConstantPosDagExpr | GraphNode]


@dataclass(frozen=True)
class SubStrDagExpr:
    i: int
    lexprs: PosExprSet
    rexprs: PosExprSet

    def __repr__(self):
        ls = ", ".join(sorted([str(e) for e in self.lexprs]))
        rs = ", ".join(sorted([str(e) for e in self.rexprs]))
        return f"SubStr(v{self.i}, {{{ls}}}, {{{rs}}})"


SubStrExprSet: TypeAlias = set[SubStrDagExpr | ConstantStrDagExpr]


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
    W: dict[DagEdge, SubStrExprSet]


def gen_substr_expr(s: str, lpos: int, rpos: int, G: InputDataGraph) -> SubStrDagExpr:
    str_id = str_to_id(s)
    lexprs: set[ConstantPosDagExpr | GraphNode] = {ConstantPosDagExpr(lpos)}
    rexprs: set[ConstantPosDagExpr | GraphNode] = {ConstantPosDagExpr(rpos)}
    for v in G.V:
        for label in G.I[v]:
            if label.str_id == str_id:
                if label.index == lpos:
                    lexprs.add(v)
                if label.index == rpos:
                    rexprs.add(v)
    return SubStrDagExpr(str_id, frozenset(lexprs), frozenset(rexprs))


def gen_dag(in_str: str, out_str: str, G: InputDataGraph) -> Dag:
    nodes: set[DagNode] = set()
    edges: set[DagEdge] = set()
    W: dict[DagEdge, SubStrExprSet] = dict()

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


def intersect_pos_expr_sets(pos_set1: PosExprSet, pos_set2: PosExprSet) -> PosExprSet:
    merged_pos_exprs: set[ConstantPosDagExpr | GraphNode] = set()
    for p1, p2 in product(pos_set1, pos_set2):
        match p1, p2:
            case ConstantPosDagExpr(k1), ConstantPosDagExpr(k2):
                if k1 == k2:
                    merged_pos_exprs.add(ConstantPosDagExpr(k1))
            case GraphNode(id1), GraphNode(id2):
                if id1 == id2:
                    merged_pos_exprs.add(GraphNode(id1))
            case _:
                pass
    return frozenset(merged_pos_exprs)


def intersect_expr_sets(expr_set1: SubStrExprSet, expr_set2: SubStrExprSet) -> SubStrExprSet:
    merged_exprs: SubStrExprSet = set()
    for expr1, expr2 in product(expr_set1, expr_set2):
        match expr1, expr2:
            case ConstantStrDagExpr(s1), ConstantStrDagExpr(s2):
                if s1 == s2:
                    merged_exprs.add(ConstantStrDagExpr(s1))
            case SubStrDagExpr(v1, ls1, rs1), SubStrDagExpr(v2, ls2, rs2):
                if v1 == v2:
                    lpos_exprs = intersect_pos_expr_sets(ls1, ls2)
                    rpos_exprs = intersect_pos_expr_sets(rs1, rs2)
                    if len(lpos_exprs) > 0 and len(rpos_exprs) > 0:
                        merged_exprs.add(SubStrDagExpr(v1, lpos_exprs, rpos_exprs))
            case _:
                pass
    return merged_exprs


def intersect_dag(dag1: Dag, dag2: Dag) -> Dag:
    def node_rewriter():
        i = 0
        cache: dict[tuple[DagNode, DagNode], int] = dict()

        def get_id(n1: DagNode, n2: DagNode) -> int:
            nonlocal i, cache
            key = (n1, n2)
            id = cache.get(key)
            if id is None:
                id = i
                i += 1
                cache[key] = id
            return id

        return get_id

    nodes: set[DagNode] = set()
    edges: set[DagEdge] = set()
    W: dict[DagEdge, SubStrExprSet] = dict()
    node_id = node_rewriter()

    # Edges and middle nodes
    dag1_edges = sorted(list(dag1.edges), key=lambda e: (e.n1.id, e.n2.id))
    dag2_edges = sorted(list(dag2.edges), key=lambda e: (e.n1.id, e.n2.id))
    for edge1, edge2 in product(dag1_edges, dag2_edges):
        merged_exprs = intersect_expr_sets(dag1.W[edge1], dag2.W[edge2])

        # Only add nodes/edges if these edges have tokens in common
        if merged_exprs:
            # New nodes and labels
            n1 = DagNode(node_id(edge1.n1, edge2.n1))
            n2 = DagNode(node_id(edge1.n2, edge2.n2))
            nodes.add(n1)
            nodes.add(n2)

            # New edge and label
            edge = DagEdge(n1, n2)
            edges.add(edge)
            W[edge] = merged_exprs

    # Start/end nodes
    start_node = DagNode(node_id(dag1.start_node, dag2.start_node))
    final_node = DagNode(node_id(dag1.final_node, dag2.final_node))
    return Dag(nodes, start_node, final_node, edges, W)


def edges_from(dag: Dag, i: int):
    return [(e.n1.id, e.n2.id) for e in dag.edges if e.n1.id == i]


def edge_exprs(dag: Dag, i: int, j: int):
    edge = DagEdge(DagNode(i), DagNode(j))
    return dag.W[edge]


def gen_dsl_pos_exprs(G: InputDataGraph, pos_expr: ConstantPosDagExpr | GraphNode) -> set[dsl.PositionExpr]:
    match pos_expr:
        case ConstantPosDagExpr(k):
            return {dsl.ConstantPos(k)}
        case GraphNode(id):
            result: set[dsl.PositionExpr] = set()
            for edge, tokens in G.L.items():
                dir = None
                if edge.n1.id == id:
                    dir = dsl.Dir.Start
                elif edge.n2.id == id:
                    dir = dsl.Dir.End
                if dir is not None:
                    match_pos_set = {dsl.MatchPos(tok.t, tok.k, dir) for tok in tokens}
                    result.update(match_pos_set)
            return result


def gen_dsl_exprs(G: InputDataGraph, dag_expr: SubStrDagExpr | ConstantStrDagExpr) -> set[dsl.SubstringExpr]:
    match dag_expr:
        case ConstantStrDagExpr(s):
            return {dsl.ConstantStr(s)}
        case SubStrDagExpr(i, lexprs, rexprs):
            lpos_set: set[dsl.PositionExpr] = set()
            rpos_set: set[dsl.PositionExpr] = set()
            for lexpr in lexprs:
                lpos_set.update(gen_dsl_pos_exprs(G, lexpr))
            for rexpr in rexprs:
                rpos_set.update(gen_dsl_pos_exprs(G, rexpr))
            return {dsl.SubStr(dsl.Var(i), lpos, rpos) for lpos, rpos in product(lpos_set, rpos_set)}


# Example
inputs = [
    "Mumbai, India",
    "Newark, United States",
]
outputs = [
    "India",
    "United States",
]

G = gen_data_input_graph(inputs)
dag1 = gen_dag(inputs[0], outputs[0], G)
dag2 = gen_dag(inputs[1], outputs[1], G)
dag = intersect_dag(dag1, dag2)

# for edge, exprs in dag.W.items():
#     exprs = sorted(list(exprs), key=lambda e: str(e))
#     exprs = ", ".join([e.__repr__() for e in exprs])
#     print(f"W({edge.n1.id}, {edge.n2.id}) = {{{exprs}}}")

expr = edge_exprs(dag, 0, 18).pop()
# print(G.L[GraphEdge(GraphNode(34), GraphNode(35))])

dsl_exprs = gen_dsl_exprs(G, expr)
env = {1: "Newark, United States"}
dsl.eval_expr(env, dsl_exprs.pop())
