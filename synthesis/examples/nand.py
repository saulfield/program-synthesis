# %%
from pydantic.dataclasses import dataclass


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

    def __repr__(self) -> str:
        return f"Nand({self.a}, {self.b})"


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


LEVELS = [
    Spec(
        "Inverter",
        [
            ([0], [1]),
            ([1], [0]),
        ],
    ),
    Spec(
        "And",
        [
            ([0, 0], [0]),
            ([0, 1], [0]),
            ([1, 0], [0]),
            ([1, 1], [1]),
        ],
    ),
    Spec(
        "Or",
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [1]),
        ],
    ),
    Spec(
        "Xor",
        [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ],
    ),
    Spec(
        "HalfAdder",
        [
            ([0, 0], [0, 0]),
            ([0, 1], [0, 1]),
            ([1, 0], [0, 1]),
            ([1, 1], [1, 0]),
        ],
    ),
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


def main():
    spec = LEVELS[0]
    tt = spec.truth_table
    n_inputs = len(tt[0][0])
    # n_outputs = len(tt[0][1])
    inputs = [Input(i) for i in range(n_inputs)]
    program = Program((Output(Nand(inputs[0], inputs[0])),))

    print(f"Component: {spec.name}")
    for inputs, outputs in spec.truth_table:
        print(f"{inputs} -> {outputs}")
    print(f"Program: {program}")
    for inputs, outputs in tt:
        result = eval_program(program, inputs)
        print(f"{inputs} -> {result}")
    print(f"Spec passed: {check(spec, program)}")


if __name__ == "__main__":
    main()
