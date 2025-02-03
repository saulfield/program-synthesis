# %%
import time
from synthesis import synthesize
from dsl import *


def test_dsl():
    # Example from 6.1
    # ----------------
    # e2 ≡ Concat(f1, ConstantStr(”.”), f2, ConstantStr(”.”))
    # f1 ≡ SubStr(v1, (C, 1, Start), (C, 1, End))
    # f2 ≡ SubStr(v1, (C, −1, Start), (l, −1, Start))

    f1 = SubStr(
        Var(1),
        MatchPos(Regex.CAPS, 1, Dir.Start),
        MatchPos(Regex.CAPS, 1, Dir.End),
    )
    f2 = SubStr(
        Var(1),
        MatchPos(Regex.CAPS, -1, Dir.Start),
        MatchPos(Regex.Lowercase, -1, Dir.Start),
    )
    expr = Concat((f1, ConstantStr("."), f2, ConstantStr(".")))

    f1 = SubStr(Var(1), ConstantPos(1), ConstantPos(2))
    f2 = SubStr(Var(1), ConstantPos(15), ConstantPos(16))
    expr = Concat((f1, ConstantStr("."), f2, ConstantStr(".")))

    eval_program(expr, ["Brandon Henry Saunders"])  # prints "B.S."


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
