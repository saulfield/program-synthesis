# %%
from itertools import product
import time
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

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Nand):
            case_1 = self.a == value.a and self.b == value.b
            case_2 = self.a == value.b and self.b == value.a
            return case_1 or case_2
        return super().__eq__(value)

    def __hash__(self) -> int:
        return super().__hash__()

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


def main():
    for spec in LEVELS:
        # spec = LEVELS[1]
        print(f"Component: {spec.name}")
        # for inputs, outputs in spec.truth_table:
        #     print(f"{inputs} -> {outputs}")
        # print()

        start = time.perf_counter()
        max_depth = 3
        n_inputs = len(spec.truth_table[0][0])
        prog_sets = gen_programs(n_inputs, max_depth)
        gen_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        found = False
        for depth, programs in prog_sets.items():
            if found:
                break
            print(f"Checking {len(programs)} programs at depth {depth}...")
            for program in programs:
                full_prog = Program((Output(program),))
                if check(spec, full_prog):
                    found = True
                    print("Program found!")
                    print(program)
                    # for inputs, outputs in spec.truth_table:
                    #     result = eval_program(full_prog, inputs)
                    #     print(f"{inputs} -> {result}")
                    break
        if not found:
            print("No solution found.")
        eval_elapsed = time.perf_counter() - start

        print(f"Gen (depth={max_depth}) took: {gen_elapsed:0.2f}s")
        print(f"Eval took: {eval_elapsed:0.2f}s")
        print()


if __name__ == "__main__":
    main()
