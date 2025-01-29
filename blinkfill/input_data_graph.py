# %%
from itertools import product
from dsl import Regex, substr, find_matches
from pydantic.dataclasses import dataclass


def gen_id():
    i = 1
    while True:
        yield i
        i += 1


id_gen = gen_id()


def string2id(s: str) -> int:
    return next(id_gen)


@dataclass(frozen=True)
class Tok:
    t: Regex | str
    k: int

    def __repr__(self):
        t_str = self.t.__repr__() if isinstance(self.t, Regex) else f'"{self.t}"'
        return f"({t_str}, {self.k})"


def get_match_ids(t: str | Regex, s: str, i: int, j: int) -> list[int]:
    matches = find_matches(s, t)
    ids = []
    for m in matches:
        if m.start == i and m.end == j:
            ids.append(m.k)
            ids.append(m.k - len(matches) - 1)
    return ids


def regex_tokens(cs: str) -> set[Regex]:
    return set()


@dataclass(frozen=True)
class Node:
    ids: tuple[int, ...]

    def __repr__(self):
        args = ", ".join([str(id) for id in self.ids])
        return f"Node({args})"


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
    str_id = string2id(s)

    # Create nodes
    for node_id in range(0, len_s + 3):
        node = Node((node_id,))
        V.add(node)
        I[node] = {NodeLabel(str_id, node_id)}

    # Create special edge labels
    edge_start = Edge(Node((0,)), Node((1,)))
    edge_end = Edge(Node((len_s + 1,)), Node((len_s + 2,)))
    E.add(edge_start)
    E.add(edge_end)
    L[edge_start] = {Tok(Regex.StartT, 1)}
    L[edge_end] = {Tok(Regex.EndT, 1)}

    # Create edges and labels
    for i in range(1, len(s) + 1):
        for j in range(i + 1, len(s) + 2):
            edge = Edge(Node((i,)), Node((j,)))
            E.add(edge)
            cs = substr(s, i, j)
            L[edge] = {Tok(cs, m_id) for m_id in get_match_ids(cs, s, i, j)}
            for t in Regex:
                L[edge].update({Tok(t, m_id) for m_id in get_match_ids(t, s, i, j)})

    return InputDataGraph(V, E, I, L)


def intersect(G1: InputDataGraph, G2: InputDataGraph) -> InputDataGraph:
    V: set[Node] = set()
    E: set[Edge] = set()
    I: dict[Node, set[NodeLabel]] = dict()
    L: dict[Edge, set[Tok]] = dict()

    for e1, e2 in list(product(G1.E, G2.E)):
        merged_set = G1.L[e1] & G2.L[e2]

        # Only add nodes/edges if these edges have tokens in common
        if merged_set:

            # New nodes and labels
            n1 = Node(e1.n1.ids + e2.n1.ids)
            n2 = Node(e1.n2.ids + e2.n2.ids)
            V.add(n1)
            V.add(n2)
            I[n1] = G1.I[e1.n1] | G2.I[e2.n1]
            I[n2] = G1.I[e1.n2] | G2.I[e2.n2]

            # New edge and label
            edge = Edge(n1, n2)
            E.add(edge)
            L[edge] = merged_set

    return InputDataGraph(V, E, I, L)


def gen_data_input_graph(strings: list[str]) -> InputDataGraph:
    G = gen_input_graph(strings[0])
    for s in strings[1:]:
        G = intersect(G, gen_input_graph(s))
    return G


# Examples from 5.2
# ----------------

# G1 = gen_input_graph("1 lb")
# G2 = gen_input_graph("23 g")
# G = intersect(G1, G2)

strings = ["1 lb", "23 g", "4 tons", "102 grams", "75 kg"]
G = gen_data_input_graph(strings)

print("V:", G.V)
print("E:", G.E)
print("I:", G.I)
print("L:", G.L)
for edge, tokens in sorted(G.L.items(), key=lambda e: e[0].n1.ids + e[0].n2.ids):
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
