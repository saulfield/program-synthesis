from itertools import product

from pydantic.dataclasses import dataclass

from synthesis.blinkfill.common import Graph, make_node_rewriter, str_to_id
from synthesis.blinkfill.dsl import Regex, find_matches, substr


@dataclass(frozen=True)
class Tok:
    t: Regex | str
    k: int

    def __repr__(self):  # pragma: no cover
        t_str = repr(self.t) if isinstance(self.t, Regex) else f'"{self.t}"'
        return f"({t_str}, {self.k})"


@dataclass(frozen=True)
class Label:
    str_id: int  # ID corresponding to the input string
    index: int  # index (position) in the string

    def __repr__(self):  # pragma: no cover
        return f"Label({self.str_id}, {self.index})"


@dataclass
class InputDataGraph:
    g: Graph
    node_data: dict[int, set[Label]]
    edge_data: dict[tuple[int, int], set[Tok]]


def get_match_ids(t: str | Regex, s: str, i: int, j: int) -> list[int]:
    matches = find_matches(s, t)
    ids: list[int] = []
    for m in matches:
        if m.start == i and m.end == j:
            ids.append(m.k)
            ids.append(m.k - len(matches) - 1)
    return ids


def gen_input_data_graph_single(s: str) -> InputDataGraph:
    len_s = len(s)
    str_id = str_to_id(s)
    g = Graph()
    node_data: dict[int, set[Label]] = dict()
    edge_data: dict[tuple[int, int], set[Tok]] = dict()

    # Create nodes
    for node_id in range(0, len_s + 3):
        g.add_node(node_id)
        node_data[node_id] = {Label(str_id, node_id)}

    # Create special edge labels
    edge_start = (0, 1)
    edge_end = (len_s + 1, len_s + 2)
    g.add_edge(edge_start)
    g.add_edge(edge_end)
    edge_data[edge_start] = {Tok(Regex.StartT, 1)}
    edge_data[edge_end] = {Tok(Regex.EndT, 1)}

    # Create edges and labels
    for i in range(1, len(s) + 1):
        for j in range(i + 1, len(s) + 2):
            edge = (i, j)
            g.add_edge(edge)
            cs = substr(s, i, j)
            edge_data[edge] = {Tok(cs, m_id) for m_id in get_match_ids(cs, s, i, j)}
            for t in Regex:
                edge_data[edge].update({Tok(t, m_id) for m_id in get_match_ids(t, s, i, j)})
    return InputDataGraph(g, node_data, edge_data)


def intersect(g1: InputDataGraph, g2: InputDataGraph) -> InputDataGraph:
    g = Graph()
    node_data: dict[int, set[Label]] = dict()
    edge_data: dict[tuple[int, int], set[Tok]] = dict()
    node_id = make_node_rewriter()

    g1_edges = sorted(list(g1.g.edges))
    g2_edges = sorted(list(g2.g.edges))
    for e1, e2 in product(g1_edges, g2_edges):
        # Only add nodes/edges if these edges have tokens in common
        merged_set = g1.edge_data[e1] & g2.edge_data[e2]
        if merged_set:
            # New nodes and labels
            src1, dst1 = e1
            src2, dst2 = e2
            src = node_id(src1, src2)
            dst = node_id(dst1, dst2)
            g.add_node(src)
            g.add_node(dst)
            node_data[src] = g1.node_data[src1] | g2.node_data[src2]
            node_data[dst] = g1.node_data[dst1] | g2.node_data[dst2]

            # New edge and label
            edge = (src, dst)
            g.add_edge(edge)
            edge_data[edge] = merged_set
    return InputDataGraph(g, node_data, edge_data)


def gen_input_data_graph(strings: list[str]) -> InputDataGraph:
    graph = gen_input_data_graph_single(strings[0])
    for s in strings[1:]:
        graph = intersect(graph, gen_input_data_graph_single(s))
    return graph


def node_dist(idg: InputDataGraph, v1: int, v2: int) -> int:
    dist = 0
    for l1, l2 in product(idg.node_data[v1], idg.node_data[v2]):
        if l1.str_id == l2.str_id:
            dist += abs(l2.index - l1.index)
    return dist


def rank_nodes(idg: InputDataGraph) -> dict[int, int]:
    in_scores: dict[int, int] = {v: 0 for v in idg.g.nodes}
    out_scores: dict[int, int] = {v: 0 for v in idg.g.nodes}
    nodes_sorted = idg.g.topological_sort()

    for v in nodes_sorted:
        for v_in in idg.g.incoming[v]:
            dist = node_dist(idg, v, v_in)
            in_scores[v] = max(in_scores[v], in_scores[v_in] + dist)

    for v in reversed(nodes_sorted):
        for v_out in idg.g.outgoing[v]:
            dist = node_dist(idg, v, v_out)
            out_scores[v] = max(out_scores[v], out_scores[v_out] + dist)

    return {v: in_scores[v] + out_scores[v] for v in idg.g.nodes}
