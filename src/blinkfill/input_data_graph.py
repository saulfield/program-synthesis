# %%
from itertools import product

from blinkfill.common import str_to_id
from blinkfill.dsl import Regex, find_matches, substr
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Tok:
    t: Regex | str
    k: int

    def __repr__(self):
        t_str = self.t.__repr__() if isinstance(self.t, Regex) else f'"{self.t}"'
        return f"({t_str}, {self.k})"


def get_match_ids(t: str | Regex, s: str, i: int, j: int) -> list[int]:
    matches = find_matches(s, t)
    ids: list[int] = []
    for m in matches:
        if m.start == i and m.end == j:
            ids.append(m.k)
            ids.append(m.k - len(matches) - 1)
    return ids


@dataclass(frozen=True)
class Node:
    id: int

    def __repr__(self):
        return f"Node({self.id})"


@dataclass(frozen=True)
class Edge:
    n1: Node
    n2: Node

    def __repr__(self):
        return f"Edge({self.n1}, {self.n2})"


@dataclass(frozen=True)
class NodeLabel:
    str_id: int  # ID corresponding to the input string
    index: int  # index (position) in the string

    def __repr__(self):
        return f"NodeLabel({self.str_id}, {self.index})"


@dataclass
class InputDataGraph:
    V: set[Node]
    E: set[Edge]
    I: dict[Node, set[NodeLabel]]
    L: dict[Edge, set[Tok]]


def gen_input_graph(s: str):
    len_s = len(s)
    V: set[Node] = set()
    E: set[Edge] = set()
    I: dict[Node, set[NodeLabel]] = dict()
    L: dict[Edge, set[Tok]] = dict()
    str_id = str_to_id(s)
    # str_id = 1

    # Create nodes
    for node_id in range(0, len_s + 3):
        node = Node(node_id)
        V.add(node)
        I[node] = {NodeLabel(str_id, node_id)}

    # Create special edge labels
    edge_start = Edge(Node(0), Node(1))
    edge_end = Edge(Node(len_s + 1), Node(len_s + 2))
    E.add(edge_start)
    E.add(edge_end)
    L[edge_start] = {Tok(Regex.StartT, 1)}
    L[edge_end] = {Tok(Regex.EndT, 1)}

    # Create edges and labels
    for i in range(1, len(s) + 1):
        for j in range(i + 1, len(s) + 2):
            edge = Edge(Node(i), Node(j))
            E.add(edge)
            cs = substr(s, i, j)
            L[edge] = {Tok(cs, m_id) for m_id in get_match_ids(cs, s, i, j)}
            for t in Regex:
                L[edge].update({Tok(t, m_id) for m_id in get_match_ids(t, s, i, j)})

    return InputDataGraph(V, E, I, L)


def intersect(G1: InputDataGraph, G2: InputDataGraph) -> InputDataGraph:
    def node_rewriter():
        i = 0
        cache: dict[tuple[Node, Node], int] = dict()

        def get_id(n1: Node, n2: Node) -> int:
            nonlocal i, cache
            key = (n1, n2)
            id = cache.get(key)
            if id is None:
                id = i
                i += 1
                cache[key] = id
            return id

        return get_id

    V: set[Node] = set()
    E: set[Edge] = set()
    I: dict[Node, set[NodeLabel]] = dict()
    L: dict[Edge, set[Tok]] = dict()

    node_id = node_rewriter()

    g1_edges = sorted(list(G1.E), key=lambda e: (e.n1.id, e.n2.id))
    g2_edges = sorted(list(G2.E), key=lambda e: (e.n1.id, e.n2.id))
    for e1, e2 in product(g1_edges, g2_edges):
        merged_set = G1.L[e1] & G2.L[e2]

        # Only add nodes/edges if these edges have tokens in common
        if merged_set:

            # New nodes and labels
            n1 = Node(node_id(e1.n1, e2.n1))
            n2 = Node(node_id(e1.n2, e2.n2))
            V.add(n1)
            V.add(n2)
            I[n1] = G1.I[e1.n1] | G2.I[e2.n1]
            I[n2] = G1.I[e1.n2] | G2.I[e2.n2]

            # New edge and label
            edge = Edge(n1, n2)
            E.add(edge)
            L[edge] = merged_set

    return InputDataGraph(V, E, I, L)


def gen_input_data_graph(strings: list[str]) -> InputDataGraph:
    graph = gen_input_graph(strings[0])
    for s in strings[1:]:
        graph = intersect(graph, gen_input_graph(s))
    return graph


@dataclass
class NodeScore:
    in_score: int = 0
    out_score: int = 0
    score: int = 0


def node_dist(G: InputDataGraph, v1: int, v2: int) -> int:
    dist = 0
    for l1, l2 in product(G.I[Node(v1)], G.I[Node(v2)]):
        if l1.str_id == l2.str_id:
            dist += abs(l2.index - l1.index)
    return dist


def topological_sort(nodes: set[int], adj: dict[int, set[int]]) -> list[int]:
    visited: dict[int, bool] = {v: False for v in nodes}
    result = []

    def dfs(v: int):
        if visited[v]:
            return
        visited[v] = True

        for v_out in adj[v]:
            dfs(v_out)

        result.insert(0, v)

    for v in nodes:
        dfs(v)

    # Verify
    for v, v_adjs in adj.items():
        for va in v_adjs:
            assert result.index(v) < result.index(va)

    return result


def rank_nodes(G: InputDataGraph) -> dict[int, int]:
    nodes = {v.id for v in G.V}
    adj: dict[int, set[int]] = {v: set() for v in nodes}
    adj_inv: dict[int, set[int]] = {v: set() for v in nodes}
    in_scores: dict[int, int] = {v: 0 for v in nodes}
    out_scores: dict[int, int] = {v: 0 for v in nodes}

    for e in G.E:
        adj[e.n1.id].add(e.n2.id)
        adj_inv[e.n2.id].add(e.n1.id)

    nodes_sorted = topological_sort(nodes, adj)
    for v in nodes_sorted:
        for vi in adj_inv[v]:
            dist = node_dist(G, v, vi)
            in_scores[v] = max(in_scores[v], in_scores[vi] + dist)

    for v in reversed(nodes_sorted):
        for vo in adj[v]:
            dist = node_dist(G, v, vo)
            out_scores[v] = max(out_scores[v], out_scores[vo] + dist)

    result: dict[int, int] = {v: in_scores[v] + out_scores[v] for v in nodes}
    return result


if __name__ == "__main__":
    # Examples from 5.2
    # ----------------

    # G1 = gen_input_graph("1 lb")
    # G2 = gen_input_graph("23 g")
    # G = intersect(G1, G2)

    strings = [
        "1 lb",
        "23 g",
        "4 tons",
        "102 grams",
        "75 kg",
    ]
    G = gen_input_data_graph(strings)

    print("V:", G.V)
    print("E:", G.E)
    print("I:", G.I)
    print("L:", G.L)
    for edge, tokens in sorted(G.L.items(), key=lambda e: (e[0].n1.id, e[0].n2.id)):
        print(f"L({edge}) = {tokens}")

    # Prints something akin to the following:
    #
    # L(Edge(0, 1)) = {(∧, 1)},
    # L(Edge(1, 2)) = {(αn, -2), (d, 1), (αn, 1), (d, -1)},
    # L(Edge(2, 3)) = {(ws, 1), (" ", -1), (ws, -1), (" ", 1)},
    # L(Edge(3, 4)) = {(αs, -1), (αs, 1), (ls, -1), (α, -1), (αn, -1), (α, 1), (ls, 1), (αn, 2), (l, -1), (l, 1)},
    # L(Edge(4, 5)) = {($, 1)}
    #
    # NOTE: this output is identical to the paper, with the exception of the 'ws' tokens.
    # It's unclear why they didn't include it in their output, since this should be matched.
