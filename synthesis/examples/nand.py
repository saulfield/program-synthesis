# %%
import time
from itertools import product
from pathlib import Path

import graphviz
from pydantic.dataclasses import dataclass

from synthesis.blinkfill.visualize import graphviz_render


@dataclass(frozen=True)
class Spec:
    name: str
    truth_table: list[tuple[list[int], list[int]]]


@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Input(Node):
    i: int

    def __repr__(self) -> str:
        return f"Input({self.i})"


@dataclass(frozen=True)
class Nand(Node):
    a: Node
    b: Node

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Nand):
            case_1 = self.a == value.a and self.b == value.b
            case_2 = self.a == value.b and self.b == value.a
            return case_1 or case_2
        return super().__eq__(value)

    def __hash__(self) -> int:
        return hash(self.a) + hash(self.b)

    def __repr__(self) -> str:
        return f"Nand({self.a}, {self.b})"


def hash_test():
    a = Nand(Input(0), Input(1))
    b = Nand(Input(0), Input(1))
    c = Nand(Input(1), Input(0))
    d = Nand(Input(0), Input(0))
    assert hash(a) == hash(b)
    assert hash(b) == hash(c)
    assert hash(c) != hash(d)


@dataclass(frozen=True)
class Output(Node):
    n: Node

    def __repr__(self) -> str:
        return f"Output({self.n})"


@dataclass(frozen=True)
class Program(Node):
    outputs: tuple[Output]

    def __repr__(self) -> str:
        return f"Program({','.join([repr(o) for o in self.outputs])})"


def draw_circuit(program: Program, name: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(name=name)
    dot.attr(rankdir="LR")

    edges: set[tuple[str, str]] = set()

    def visit(node: Node) -> str:
        match node:
            case Input(i):
                with dot.subgraph(name="cluster_input") as s:  # type: ignore
                    s.attr(rank="same")
                    node_id = str(i)
                    s.node(node_id, label=chr(ord("a") + i), shape="plain")
                    return node_id
            case Nand(a, b):
                node_id = str(hash(node))
                dot.node(node_id, label=r"{{<left>|<right>}| NAND}", shape="record")
                e1 = (visit(a), f"{node_id}:left")
                e2 = (visit(b), f"{node_id}:right")
                if e1 not in edges:
                    edges.add(e1)
                    dot.edge(*e1)
                if e2 not in edges:
                    edges.add(e2)
                    dot.edge(*e2)
                return node_id
            case _:
                raise ValueError("Unexpected node type.")

    for output in program.outputs:
        node_id = str(hash(output))
        dot.node(node_id, label="out", shape="circle")
        dot.edge(visit(output.n), node_id)
    return dot


def count_gates(program: Program) -> int:
    gates = set()

    def visit(node: Node):
        match node:
            case Nand(a, b):
                gates.add(Nand(a, b))
                visit(a)
                visit(b)
            case Output(node):
                return visit(node)
            case Program(outputs):
                for output in outputs:
                    visit(output)
            case _:
                pass

    visit(program)
    return len(gates)


LEVELS = [
    Spec(
        "NOT",
        [
            ([0], [1]),
            ([1], [0]),
        ],
    ),
    Spec(
        "AND",
        [
            ([0, 0], [0]),
            ([0, 1], [0]),
            ([1, 0], [0]),
            ([1, 1], [1]),
        ],
    ),
    Spec(
        "OR",
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [1]),
        ],
    ),
    Spec(
        "XOR",
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ],
    ),
    # Spec(
    #     "HalfAdder",
    #     [
    #         ([0, 0], [0, 0]),
    #         ([0, 1], [0, 1]),
    #         ([1, 0], [0, 1]),
    #         ([1, 1], [1, 0]),
    #     ],
    # ),
]


def eval_node(env: dict[int, int], node: Node) -> list[int]:
    match node:
        case Input(i):
            return [env[i]]
        case Nand(a, b):
            return [int(not (eval_node(env, a)[0] and eval_node(env, b)[0]))]
        case Output(node):
            return eval_node(env, node)
        case Program(outputs):
            result = []
            for output in outputs:
                result += eval_node(env, output)
            return result
        case _:
            raise ValueError("Unknown node type.")


def eval_program(program: Program, inputs: list[int]) -> list[int]:
    env = {i: val for i, val in enumerate(inputs)}
    return eval_node(env, program)


def check(spec: Spec, program: Program) -> bool:
    for inputs, outputs in spec.truth_table:
        if eval_program(program, inputs) != outputs:
            return False
    return True


def gen_programs(n_inputs: int, max_depth: int) -> dict[int, set[Node]]:
    prog_sets: dict[int, set[Node]] = {0: {Input(i) for i in range(n_inputs)}}
    for depth in range(1, max_depth + 1):
        prev = set()
        for progs in prog_sets.values():
            prev.update(progs)
        prog_sets[depth] = {Nand(a, b) for a, b in product(prev, prev)}
    return prog_sets


@dataclass(frozen=True)
class EvalResult:
    rows: tuple[tuple[int, ...], ...]


def eval_result(spec: Spec, node: Node) -> EvalResult:
    rows = []
    for inputs, outputs in spec.truth_table:
        env = {i: val for i, val in enumerate(inputs)}
        rows.append(tuple(eval_node(env, node)))
    return EvalResult(tuple(rows))


def grow(spec: Spec, cache: dict[EvalResult, Node]):
    nodes: set[Node] = set(cache.values())
    for a, b in product(nodes, nodes):
        node = Nand(a, b)
        result = eval_result(spec, node)
        if result not in cache:
            cache[result] = node


def main():
    base_path = Path(__file__).parent / "diagrams"
    base_path.mkdir(exist_ok=True)
    for spec in LEVELS[:4]:
        print(f"Component: {spec.name}")
        start = time.perf_counter()

        max_depth = 3
        n_inputs = len(spec.truth_table[0][0])
        input_nodes = {Input(i) for i in range(n_inputs)}
        cache: dict[EvalResult, Node] = {eval_result(spec, n): n for n in input_nodes}

        found = False
        for depth in range(1, max_depth + 1):
            if found:
                break
            grow(spec, cache)
            nodes = cache.values()
            print(f"Programs at depth {depth}: {len(nodes)}")
            for node in sorted(nodes, key=lambda p: repr(p)):
                full_prog = Program((Output(node),))
                if check(spec, full_prog):
                    found = True
                    n_gates = count_gates(full_prog)
                    print("Program found!")
                    print(f"Gates: {n_gates}")
                    # print(full_prog)

                    dot = draw_circuit(full_prog, f"{spec.name}")
                    graphviz_render(dot, base_path / f"{spec.name}")

                    # for inputs, outputs in spec.truth_table:
                    #     result = eval_program(full_prog, inputs)
                    #     print(f"{inputs} -> {result}")
                    break
        if not found:
            print("No solution found.")
        eval_elapsed = time.perf_counter() - start

        print(f"Eval took: {eval_elapsed:0.2f}s")
        print()


if __name__ == "__main__":
    main()
