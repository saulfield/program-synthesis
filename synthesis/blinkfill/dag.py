import heapq
from itertools import product
from typing import TypeAlias

from pydantic.dataclasses import dataclass

from synthesis.blinkfill import dsl
from synthesis.blinkfill.common import Graph, make_node_rewriter, str_to_id
from synthesis.blinkfill.dsl import substr
from synthesis.blinkfill.input_data_graph import InputDataGraph, rank_nodes


@dataclass(frozen=True)
class ConstantPosDagExpr:
    k: int

    def __repr__(self):  # pragma: no cover
        return f"ConstantPos({self.k})"


@dataclass(frozen=True)
class ConstantStrDagExpr:
    s: str

    def __repr__(self):  # pragma: no cover
        return f'ConstantStr("{self.s}")'


PosExpr: TypeAlias = ConstantPosDagExpr | int
PosExprSet: TypeAlias = frozenset[PosExpr]


@dataclass(frozen=True)
class SubStrDagExpr:
    i: int
    lexprs: PosExprSet
    rexprs: PosExprSet

    def __repr__(self):  # pragma: no cover
        ls = ", ".join(sorted([str(e) for e in self.lexprs]))
        rs = ", ".join(sorted([str(e) for e in self.rexprs]))
        return f"SubStr(v{self.i}, {{{ls}}}, {{{rs}}})"


SubStrExpr: TypeAlias = SubStrDagExpr | ConstantStrDagExpr
SubStrExprSet: TypeAlias = frozenset[SubStrExpr]


@dataclass(frozen=True)
class Dag:
    g: Graph
    start_node: int
    final_node: int
    edge_data: dict[tuple[int, int], SubStrExprSet]


def gen_substr_expr(s: str, lpos: int, rpos: int, idg: InputDataGraph) -> SubStrDagExpr:
    str_id = str_to_id(s)
    lexprs: set[PosExpr] = {ConstantPosDagExpr(lpos)}
    rexprs: set[PosExpr] = {ConstantPosDagExpr(rpos)}
    for v in idg.g.nodes:
        for label in idg.node_data[v]:
            if label.str_id == str_id:
                if label.index == lpos:
                    lexprs.add(v)
                if label.index == rpos:
                    rexprs.add(v)
    col = 1  # NOTE: column number here, not string ID
    return SubStrDagExpr(col, frozenset(lexprs), frozenset(rexprs))


def gen_dag_single(idg: InputDataGraph, in_str: str, out_str: str) -> Dag:
    g = Graph()
    edge_data: dict[tuple[int, int], SubStrExprSet] = dict()

    # Add nodes
    start_node = 0
    final_node = len(out_str)
    g.add_node(start_node)
    g.add_node(final_node)
    for i in range(1, len(out_str) + 1):
        g.add_node(i)

    # Iterate over substrings
    for i in range(0, len(out_str)):
        for j in range(i + 1, len(out_str) + 1):
            edge = (i, j)
            g.add_edge(edge)
            out_ss = substr(out_str, i + 1, j + 1)
            edge_data[edge] = frozenset({ConstantStrDagExpr(out_ss)})

            for left in range(1, len(in_str) + 1):
                for right in range(left + 1, len(in_str) + 2):
                    in_ss = substr(in_str, left, right)
                    if in_ss == out_ss:
                        ss_expr = gen_substr_expr(in_str, left, right, idg)
                        edge_data[edge] |= {ss_expr}
    return Dag(g, start_node, final_node, edge_data)


def intersect_pos_expr_sets(pos_set1: PosExprSet, pos_set2: PosExprSet) -> PosExprSet:
    merged_pos_exprs: set[PosExpr] = set()
    for p1, p2 in product(pos_set1, pos_set2):
        match p1, p2:
            case ConstantPosDagExpr(k1), ConstantPosDagExpr(k2):
                if k1 == k2:
                    merged_pos_exprs.add(ConstantPosDagExpr(k1))
            case id1, id2 if isinstance(id1, int) and isinstance(id2, int):
                if id1 == id2:
                    merged_pos_exprs.add(id1)
            case _:
                pass
    return frozenset(merged_pos_exprs)


def intersect_expr_sets(expr_set1: SubStrExprSet, expr_set2: SubStrExprSet) -> SubStrExprSet:
    merged_exprs: set[SubStrExpr] = set()
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
    return frozenset(merged_exprs)


def intersect_dag(dag1: Dag, dag2: Dag) -> Dag:
    g = Graph()
    edge_data: dict[tuple[int, int], SubStrExprSet] = dict()
    node_id = make_node_rewriter()

    dag1_edges = sorted(list(dag1.g.edges))
    dag2_edges = sorted(list(dag2.g.edges))
    for e1, e2 in product(dag1_edges, dag2_edges):
        # Only add nodes/edges if these edges have tokens in common
        merged_exprs = intersect_expr_sets(dag1.edge_data[e1], dag2.edge_data[e2])
        if merged_exprs:
            # New nodes and labels
            src1, dst1 = e1
            src2, dst2 = e2
            src = node_id(src1, src2)
            dst = node_id(dst1, dst2)
            g.add_node(src)
            g.add_node(dst)

            # New edge and label
            edge = (src, dst)
            g.add_edge(edge)
            edge_data[edge] = merged_exprs
    # Start/end nodes
    start_node = node_id(dag1.start_node, dag2.start_node)
    final_node = node_id(dag1.final_node, dag2.final_node)
    return Dag(g, start_node, final_node, edge_data)


def gen_dag(idg: InputDataGraph, inputs: list[str], outputs: list[str]) -> Dag:
    dag = gen_dag_single(idg, inputs[0], outputs[0])
    for i in range(1, len(outputs)):
        dag = intersect_dag(dag, gen_dag_single(idg, inputs[i], outputs[i]))
    return dag


def best_pos_expr(node_scores: dict[int, int], exprs: PosExprSet) -> PosExpr:
    best_score = -1
    best_expr = None
    for expr in exprs:
        match expr:
            case ConstantPosDagExpr(_):
                score = 0
            case i:
                assert isinstance(i, int)
                score = node_scores[i]
        if score > best_score:
            best_score = score
            best_expr = expr
    assert best_expr is not None
    return best_expr


def pos_index(idg: InputDataGraph, pos: PosExpr) -> int:
    match pos:
        case ConstantPosDagExpr(k):
            return k
        case _:
            # NOTE: the paper is unclear on how to measure these lengths,
            # so we just take the average across all the labels.
            labels = idg.node_data[pos]
            return int(sum([label.index for label in labels]) / len(labels))


def expr_score(idg: InputDataGraph, node_scores: dict[int, int], expr: SubStrExpr):
    match expr:
        case ConstantStrDagExpr(s):
            return 0.1 * len(s) ** 2
        case SubStrDagExpr(_, lexprs, rexprs):
            lexpr = best_pos_expr(node_scores, lexprs)
            rexpr = best_pos_expr(node_scores, rexprs)
            left = pos_index(idg, lexpr)
            right = pos_index(idg, rexpr)
            str_len = abs(right - left)
            return 1.5 * str_len**2


def best_path(idg: InputDataGraph, dag: Dag) -> list[SubStrExpr]:
    edge_scores: dict[tuple[int, int], tuple[float, SubStrExpr]] = dict()
    node_scores = rank_nodes(idg)
    for edge, exprs in dag.edge_data.items():
        results = [(expr_score(idg, node_scores, expr), expr) for expr in exprs]
        results = sorted(results, key=lambda x: x[0], reverse=True)
        best_score, best_expr = results[0]
        edge_scores[edge] = (best_score, best_expr)

    # Dijkstra’s Algorithm
    start = dag.start_node
    end = dag.final_node
    queue = [(0, start)]
    came_from: dict[int, int] = dict()
    cost_so_far = dict()
    cost_so_far[start] = 0
    v = start

    while queue:
        _, v = heapq.heappop(queue)
        if v == end:
            break
        for v_out in dag.g.outgoing[v]:
            score, _ = edge_scores[(v, v_out)]
            new_cost = cost_so_far[v] - score
            if v_out not in cost_so_far or new_cost < cost_so_far[v_out]:
                cost_so_far[v_out] = new_cost
                came_from[v_out] = v
                heapq.heappush(queue, (new_cost, v_out))

    assert v == end, "No path found"
    path: list[SubStrExpr] = []
    while v != start:
        v_prev = came_from[v]
        expr = edge_scores[(v_prev, v)][1]
        path.append(expr)
        v = v_prev
    path.reverse()
    return path


def gen_dsl_pos_exprs(idg: InputDataGraph, pos_expr: PosExpr) -> set[dsl.PositionExpr]:
    match pos_expr:
        case ConstantPosDagExpr(k):
            return {dsl.ConstantPos(k)}
        case id if isinstance(id, int):
            result: set[dsl.PositionExpr] = set()
            for (src, dst), tokens in idg.edge_data.items():
                dir = None
                if src == id:
                    dir = dsl.Dir.Start
                elif dst == id:
                    dir = dsl.Dir.End
                if dir is not None:
                    match_pos_set = {dsl.MatchPos(tok.t, tok.k, dir) for tok in tokens}
                    result.update(match_pos_set)
            return result
        case expr:
            raise ValueError("Unexpected type for pos_expr", expr)


def gen_dsl_exprs(idg: InputDataGraph, dag_expr: SubStrExpr) -> set[dsl.SubstringExpr]:
    match dag_expr:
        case ConstantStrDagExpr(s):
            return {dsl.ConstantStr(s)}
        case SubStrDagExpr(i, lexprs, rexprs):
            lpos_set: set[dsl.PositionExpr] = set()
            rpos_set: set[dsl.PositionExpr] = set()
            for lexpr in lexprs:
                lpos_set.update(gen_dsl_pos_exprs(idg, lexpr))
            for rexpr in rexprs:
                rpos_set.update(gen_dsl_pos_exprs(idg, rexpr))
            return {dsl.SubStr(dsl.Var(i), lpos, rpos) for lpos, rpos in product(lpos_set, rpos_set)}


def gen_program(idg: InputDataGraph, dag_exprs: list[SubStrExpr]) -> dsl.Concat:
    substr_exprs = [gen_dsl_exprs(idg, dag_expr).pop() for dag_expr in dag_exprs]
    return dsl.Concat(tuple(substr_exprs))
