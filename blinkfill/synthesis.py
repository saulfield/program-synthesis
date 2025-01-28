# %%
import time
from itertools import product

from dsl import *

grammar = {
    "e": [["Concat", "f", "f", "f", "f"]],
    "f": [["ConstantStr", "s"], ["SubStr", "p", "p"]],
    "p": [["ConstantPos", "i"]],
    "s": ["."],
    "i": [i for i in range(1, 10)],
}


def construct(ast_name: str, args: list):
    match ast_name:
        case "Concat":
            return Concat(tuple(args))
        case "ConstantStr":
            return ConstantStr(args[0])
        case "SubStr":
            lpos: ConstantPos = args[0]
            rpos: ConstantPos = args[1]
            if lpos.k > rpos.k:
                return None
            return SubStr(Var(1), lpos, rpos)
        case "ConstantPos":
            return ConstantPos(args[0])
        case _:
            raise ValueError(f"Unknown AST node: {ast_name}")


def synthesize(inputs: list[str], outputs: list[str], max_depth: int = 1):
    """Bottom-up enumerative synthesis"""

    cache = {non_term: set() for non_term in grammar.keys()}

    # Add all terminals
    for terminal in grammar["s"]:
        cache["s"].add(terminal)
    for terminal in grammar["i"]:
        cache["i"].add(terminal)

    for _ in range(max_depth):
        # Grow
        for non_term in ["p", "f", "e"]:
            for rule in grammar[non_term]:
                cached_terms = [cache.get(expand_term, set()) for expand_term in rule[1:]]
                for combination in product(*cached_terms):
                    new_term = construct(rule[0], list(combination))
                    if new_term and new_term not in cache[non_term]:
                        # print(new_term)
                        cache[non_term].add(new_term)

        # Validate
        # print(len(cache["e"]))
        env = {1: inputs[0]}
        for program in cache["e"]:
            # print(program)
            # print(eval_expr(env, program))
            if eval_expr(env, program) == outputs[0]:
                return program
    return None


start = time.perf_counter()

inputs = ["Clark Kent"]
outputs = ["C.K."]
program = synthesize(inputs, outputs, max_depth=1)
if program:
    print("Program found!")
    print(program)
else:
    print("No program found.")

end = time.perf_counter()

print(f"Took: {end - start:0.2f}s")
