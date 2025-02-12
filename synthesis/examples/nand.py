# %%
import time
from itertools import product
from pathlib import Path


from pydantic.dataclasses import dataclass

from synthesis.blinkfill.visualize import graphviz_render
from synthesis.nand.dsl import Input, Nand, Node, Output, Program, count_gates, draw_circuit, eval_node, eval_program
from synthesis.nand.spec import SPECS, Spec


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


def grow_naive(nodes: set[Node]):
    new_nodes = {Nand(a, b) for a, b in product(nodes, nodes)}
    nodes |= new_nodes


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
    for spec in SPECS[:5]:
        print(f"Component: {spec.name}")
        start = time.perf_counter()

        max_depth = 3
        n_inputs = len(spec.truth_table[0][0])
        n_outputs = len(spec.truth_table[0][1])
        input_nodes: set[Node] = {Input(i) for i in range(n_inputs)}
        nodes = input_nodes
        cache: dict[EvalResult, Node] = {eval_result(spec, n): n for n in input_nodes}

        found = False
        for depth in range(1, max_depth + 1):
            if found:
                break

            grow(spec, cache)
            nodes = list(cache.values())

            # grow_naive(nodes)

            output_nodes = [nodes] * n_outputs
            output_progs = list(product(*output_nodes))
            print(f"Programs at depth {depth}: {len(nodes)} nodes, {len(output_progs)} permutations")
            for node_permutations in output_progs:
                # for node in sorted(nodes, key=lambda p: repr(p)):
                full_prog = Program(tuple([Output(node) for node in node_permutations]))
                if check(spec, full_prog):
                    found = True
                    n_gates = count_gates(full_prog)
                    print("Program found!")
                    print(f"Gates: {n_gates}")
                    # print(full_prog)
                    # dot = draw_circuit(full_prog, f"{spec.name}")
                    # graphviz_render(dot, base_path / f"{spec.name}")
                    break
        if not found:
            print("No solution found.")
        eval_elapsed = time.perf_counter() - start

        print(f"Synthesis took: {eval_elapsed:0.2f}s")
        print()


# Program(Output(Nand(Nand(Input(0), Input(1)), Nand(Nand(Input(0), Input(1)), Nand(Nand(Input(1), Input(2)), Nand(Input(0), Input(2)))))))


if __name__ == "__main__":
    main()
