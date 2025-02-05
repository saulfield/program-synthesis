# %%
# Attempt to find a simple BlinkFill DSL program using bottom-up enumerative search.

# type: ignore
import time
from itertools import product
from typing import Any

from synthesis.blinkfill.dsl import *

GRAMMAR = {
    "e": [["Concat", "f", "f", "f", "f"]],
    "f": [["ConstantStr", "s"], ["SubStr", "p", "p"]],
    "p": [["ConstantPos", "i"]],
    "s": [["str"]],
    "i": [["int"]],
}


def construct(ast_name: str, args: list[Expr]) -> Expr:
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


def eval_any(env: dict[int, str], expr: Expr):
    match expr:
        case ConstantPos(_) | MatchPos(_):
            return eval_pos(env[1], expr)
        case Concat(_) | ConstantStr(_) | SubStr(_):
            return eval_expr(env, expr)
        case _:
            raise ValueError(f"Unsupported expression: {expr}")


def synthesize(examples: list[tuple[str, str]]):
    """Bottom-up enumerative synthesis."""

    inputs = [e[0] for e in examples]
    outputs = [e[1] for e in examples]
    env = {1: inputs[0]}
    term_cache = {non_term: set() for non_term in GRAMMAR.keys()}

    # Add all terminals
    novel_chars = set(outputs[0]) - set(inputs[0])
    term_cache["s"].update(novel_chars)
    term_cache["i"].update(list(range(1, len(inputs[0]) + 1)))

    start = time.perf_counter()
    for nonterminal in ["p", "f", "e"]:
        # Grow
        val_cache: dict[Any, set] = dict()
        for rule in GRAMMAR[nonterminal]:
            cached_terms = [term_cache.get(expand_term, set()) for expand_term in rule[1:]]
            for combination in product(*cached_terms):
                new_term = construct(rule[0], list(combination))
                if new_term:
                    val = eval_any(env, new_term)
                    val_cache.setdefault(val, set())
                    val_cache[val].add(new_term)
                    # val_str = f'"{val}"' if isinstance(val, str) else val
                    # print(f"{new_term} -> {val_str}")

        # Eliminate equivalent terms
        for val, term_set in val_cache.items():
            # TODO: get the "simplest" term from the set
            new_term = term_set.pop()
            term_cache[nonterminal].add(new_term)
    end = time.perf_counter()
    print(f"Grow + elim: {end - start:0.2f}s")

    # Validate
    result = None
    start = time.perf_counter()
    for program in term_cache["e"]:
        if eval_expr(env, program) == outputs[0]:
            result = program
            break
    end = time.perf_counter()
    print(f"Validate took: {end - start:0.2f}s")

    print("Production counts:")
    for nonterminal, exprs in term_cache.items():
        print(nonterminal, len(exprs))

    return result


def test_synthesis():
    examples = [("Clark Kent", "C.K.")]

    start = time.perf_counter()
    program = synthesize(examples)
    end = time.perf_counter()

    if program:
        print("Program found!")
        print(program)
    else:
        print("No program found.")

    print(f"Took: {end - start:0.2f}s")

    # Without equivalence reduction:
    # e 4477456
    # f 46
    # p 9
    # s 1
    # i 9
    # Program found!
    # Concat(SubStr(Var(1), ConstantPos(1), ConstantPos(2)), ConstantStr(.), SubStr(Var(1), ConstantPos(7), ConstantPos(8)), ConstantStr(.))
    # Took: 62.63s

    # With equivalence reduction:
    # e 1591313
    # f 38
    # p 9
    # s 1
    # i 9
    # Program found!
    # Concat(SubStr(Var(1), ConstantPos(1), ConstantPos(2)), ConstantStr(.), SubStr(Var(1), ConstantPos(7), ConstantPos(8)), ConstantStr(.))
    # Took: 46.80s


# Test
examples = [("Clark Kent", "C.K.")]

start = time.perf_counter()
program = synthesize(examples)
end = time.perf_counter()

if program:
    print("Program found!")
    print(program)
else:
    print("No program found.")

print(f"Took: {end - start:0.2f}s")
