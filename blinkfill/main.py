# %%
from dsl import *


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
expr = Concat([f1, ConstantStr("."), f2, ConstantStr(".")])

env = {1: "Brandon Henry Saunders"}
eval_expr(env, expr)  # prints "B.S."
