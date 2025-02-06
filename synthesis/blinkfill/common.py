from pydantic.dataclasses import dataclass
from pydantic import Field


@dataclass
class Graph:
    nodes: set[int] = Field(default_factory=set)
    edges: set[tuple[int, int]] = Field(default_factory=set)
    outgoing: dict[int, set[int]] = Field(default_factory=dict)
    incoming: dict[int, set[int]] = Field(default_factory=dict)

    def add_node(self, i: int):
        self.nodes.add(i)
        self.outgoing.setdefault(i, set())
        self.incoming.setdefault(i, set())

    def add_edge(self, edge: tuple[int, int]):
        src, dst = edge
        assert src in self.nodes, "Node does not exist"
        assert dst in self.nodes, "Node does not exist"
        self.edges.add((src, dst))
        self.outgoing[src].add(dst)
        self.incoming[dst].add(src)

    def topological_sort(self) -> list[int]:
        visited: dict[int, bool] = {v: False for v in self.nodes}
        result = []

        def dfs(v: int):
            if visited[v]:
                return
            visited[v] = True

            for v_out in self.outgoing[v]:
                dfs(v_out)

            result.insert(0, v)

        for v in self.nodes:
            dfs(v)

        # Verify
        for v, vs_outgoing in self.outgoing.items():
            for v_out in vs_outgoing:
                assert result.index(v) < result.index(v_out)
        return result


def make_node_rewriter():
    i = 0
    cache: dict[tuple[int, int], int] = dict()

    def get_id(n1: int, n2: int) -> int:
        nonlocal i, cache
        key = (n1, n2)
        id = cache.get(key)
        if id is None:
            id = i
            i += 1
            cache[key] = id
        return id

    return get_id


# TODO: this should be encapsulated in the main interface
def make_counter():
    i = 0

    def f():
        nonlocal i
        r = i
        i += 1
        return r

    return f


_gen_str_id = make_counter()
_ = _gen_str_id()
_str_id_map: dict[str, int] = dict()


def str_to_id(s: str) -> int:
    id = _str_id_map.get(s)
    if id is None:
        id = _gen_str_id()
        _str_id_map[s] = id
    return id
