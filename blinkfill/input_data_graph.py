# %%
from dsl import Regex, substr, find_matches
from pydantic.dataclasses import dataclass


def gen_id():
    i = 1
    while True:
        yield i
        i += 1


def string2id(s: str) -> int:
    return next(gen_id())


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


@dataclass
class InputDataGraph:
    V: set[int]
    E: set[tuple[int, int]]
    I: dict[int, tuple[int, int]]
    L: dict[tuple[int, int], set[Tok]]


def gen_input_graph(s: str):
    len_s = len(s)
    V: set[int] = set()
    E: set[tuple[int, int]] = set()
    I: dict[int, tuple[int, int]] = dict()
    L: dict[tuple[int, int], set[Tok]] = dict()
    str_id = string2id(s)

    # Create nodes
    for node_id in range(0, len_s + 3):
        V.add(node_id)
        I[node_id] = (str_id, node_id)

    # Create special edge labels
    L[(0, 1)] = {Tok(Regex.StartT, 1)}
    L[(len_s + 1, len_s + 2)] = {Tok(Regex.EndT, 1)}

    # Create edges and labels
    for i in range(1, len(s) + 1):
        for j in range(i + 1, len(s) + 2):
            E.add((i, j))
            cs = substr(s, i, j)
            L[(i, j)] = {Tok(cs, m_id) for m_id in get_match_ids(cs, s, i, j)}
            for t in Regex:
                L[(i, j)].update({Tok(t, m_id) for m_id in get_match_ids(t, s, i, j)})

    return InputDataGraph(V, E, I, L)


G1 = gen_input_graph("1 lb")
G2 = gen_input_graph("23 g")
# print("V:", V)
# print("E:", E)
# print("I:", I)
for edge, tokens in sorted(G1.L.items()):
    print(f"L{edge} = {tokens}")
