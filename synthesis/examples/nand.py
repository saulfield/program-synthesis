# %%
import time
from itertools import product
from pathlib import Path


from pydantic.dataclasses import dataclass

from synthesis.blinkfill.visualize import graphviz_render
from synthesis.nand.dsl import Input, Nand, Node, Output, Program, count_gates, draw_circuit, eval_node, eval_program
from synthesis.nand.spec import SPECS, Spec


def check(spec: Spec, program: Program, col: int | None = None) -> bool:
    for inputs, outputs in spec.truth_table:
        expected = [outputs[col]] if col is not None else outputs
        if eval_program(program, inputs) != expected:
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


def simplify(node: Node) -> Node:
    match node:
        case Nand(Nand(a, b), Nand(c, d)) if a == b == c == d:
            # print(f"Simplifying: {node} -> {a}")
            return a
        case _:
            return node


def grow(spec: Spec, cache: dict[EvalResult, Node]):
    nodes: set[Node] = set(cache.values())
    for a, b in product(nodes, nodes):
        node = simplify(Nand(a, b))
        result = eval_result(spec, node)
        if result not in cache:
            cache[result] = node


def grow_elim(spec: Spec, nodes: set[Node]):
    cache: dict[EvalResult, Node] = dict()
    for a, b in product(nodes, nodes):
        node = simplify(Nand(a, b))
        result = eval_result(spec, node)
        if result not in cache:
            cache[result] = node
    new_nodes = set(cache.values())
    nodes |= new_nodes


def grow_naive(nodes: set[Node]):
    new_nodes = {simplify(Nand(a, b)) for a, b in product(nodes, nodes)}
    nodes |= new_nodes


def main():
    base_path = Path(__file__).parent / "diagrams"
    base_path.mkdir(exist_ok=True)
    for spec in SPECS[-2:-1]:
        print(f"Component: {spec.name}")
        start = time.perf_counter()

        max_depth = 5
        n_inputs = len(spec.truth_table[0][0])
        n_outputs = len(spec.truth_table[0][1])
        input_nodes: set[Node] = {Input(i) for i in range(n_inputs)}
        # cache: dict[EvalResult, Node] = {eval_result(spec, n): n for n in input_nodes}
        nodes = input_nodes

        found = False
        solutions: dict[int, tuple[Output, int]] = dict()
        for depth in range(1, max_depth + 1):
            # if found:
            #     break

            # grow(spec, cache)
            # nodes = list(cache.values())

            grow_elim(spec, nodes)

            # grow_naive(nodes)

            # output_nodes = [nodes] * n_outputs
            # output_progs = list(product(*output_nodes))
            print(f"Programs at depth {depth}: {len(nodes)} nodes")
            for i in range(n_outputs):
                for node in nodes:
                    program = Program((Output(node),))
                    if check(spec, program, col=i):
                        gates = count_gates(program)
                        sol = solutions.get(i)
                        if sol is None or gates < sol[1]:
                            print(f"Found solution for column {i} ({gates} gates)")
                            solutions[i] = (program.outputs[0], gates)
                        break
            if len(solutions) == n_outputs:
                outputs = [solutions[i][0] for i in sorted(solutions.keys())]
                full_program = Program(tuple(outputs))
                gates = count_gates(full_program)
                print("Solution found!")
                print(f"Gates: {gates}")
                # print(full_program)
                # dot = draw_circuit(full_program, f"{spec.name}")
                # graphviz_render(dot, base_path / f"{spec.name}")
                # found = True
                # break
        if not found:
            print("No solution found.")
        eval_elapsed = time.perf_counter() - start

        print(f"Synthesis took: {eval_elapsed:0.2f}s")
        print()


# Program(Output(Nand(Nand(Input(0), Input(1)), Nand(Nand(Input(0), Input(1)), Nand(Nand(Input(1), Input(2)), Nand(Input(0), Input(2)))))))


if __name__ == "__main__":
    main()
