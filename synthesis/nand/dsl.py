from pydantic.dataclasses import dataclass
import graphviz


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


@dataclass(frozen=True)
class Output(Node):
    n: Node

    def __repr__(self) -> str:
        return f"Output({self.n})"


@dataclass(frozen=True)
class Program(Node):
    outputs: tuple[Output, ...]

    def __repr__(self) -> str:
        return f"Program({','.join([repr(o) for o in self.outputs])})"


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


def hash_test():
    a = Nand(Input(0), Input(1))
    b = Nand(Input(0), Input(1))
    c = Nand(Input(1), Input(0))
    d = Nand(Input(0), Input(0))
    assert hash(a) == hash(b)
    assert hash(b) == hash(c)
    assert hash(c) != hash(d)


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

    with dot.subgraph(name="cluster_output") as s:  # type: ignore
        s.attr(rank="same")
        for i, output in enumerate(program.outputs):
            node_id = str(hash(output))
            s.node(node_id, label=f"out_{i}", shape="plain")
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
