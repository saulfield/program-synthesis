import pytest

from synthesis.blinkfill import learn, run
from synthesis.blinkfill.dsl import (
    Concat,
    ConstantPos,
    ConstantStr,
    Dir,
    MatchPos,
    Regex,
    SubStr,
    Var,
    eval_program,
)
from synthesis.blinkfill.input_data_graph import Edge, InputDataGraph, Node, Tok, gen_input_data_graph


# Example from 6.1
# ----------------
# e2 ≡ Concat(f1, ConstantStr(”.”), f2, ConstantStr(”.”))
# f1 ≡ SubStr(v1, (C, 1, Start), (C, 1, End))
# f2 ≡ SubStr(v1, (C, −1, Start), (l, −1, Start))
def test_dsl():
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

    eval_program(expr, "Brandon Henry Saunders")  # prints "B.S."


# Example from 5.2
def test_input_data_graph():
    strings = ["1 lb", "23 g", "4 tons", "102 grams", "75 kg"]
    g = gen_input_data_graph(strings)

    # Prints something akin to the following:
    # L(Edge(0, 1)) = {(∧, 1)},
    # L(Edge(1, 2)) = {(αn, -2), (d, 1), (αn, 1), (d, -1)},
    # L(Edge(2, 3)) = {(ws, 1), (" ", -1), (ws, -1), (" ", 1)},
    # L(Edge(3, 4)) = {(αs, -1), (αs, 1), (ls, -1), (α, -1), (αn, -1), (α, 1), (ls, 1), (αn, 2), (l, -1), (l, 1)},
    # L(Edge(4, 5)) = {($, 1)}
    # NOTE: this output is identical to the paper, with the exception of the 'ws' tokens.
    # It's unclear why they didn't include it in their output, since this should be matched.
    for edge, tokens in sorted(g.L.items(), key=lambda e: (e[0].n1.id, e[0].n2.id)):
        print(f"L({edge}) = {tokens}")

    def edge_at(g: InputDataGraph, i, j):
        return g.L[Edge(Node(i), Node(j))]

    assert edge_at(g, 0, 1) == {Tok(Regex.StartT, 1)}
    assert edge_at(g, 1, 2) == {
        Tok(Regex.Alphanumeric, -2),
        Tok(Regex.Digits, 1),
        Tok(Regex.Alphanumeric, 1),
        Tok(Regex.Digits, -1),
    }
    assert edge_at(g, 2, 3) == {Tok(Regex.Whitespace, 1), Tok(" ", -1), Tok(Regex.Whitespace, -1), Tok(" ", 1)}
    assert edge_at(g, 3, 4) == {
        Tok(Regex.AlphabetsWSpaces, -1),
        Tok(Regex.AlphabetsWSpaces, 1),
        Tok(Regex.LowercaseWSpaces, -1),
        Tok(Regex.Alphabets, -1),
        Tok(Regex.Alphanumeric, -1),
        Tok(Regex.Alphabets, 1),
        Tok(Regex.LowercaseWSpaces, 1),
        Tok(Regex.Alphanumeric, 2),
        Tok(Regex.Lowercase, -1),
        Tok(Regex.Lowercase, 1),
    }
    assert edge_at(g, 4, 5) == {Tok(Regex.EndT, 1)}


table1 = [
    ("Mumbai, India", "India"),
    ("Los Angeles, United States of America", None),
    ("Newark, United States", None),
    ("New York, United States of America", None),
    ("Wellington, New Zealand", None),
    ("New Delhi, India", None),
]

table2 = [
    ("Brandon Henry Saunders", "B.S."),
    ("William Lee", None),
    ("Dafna Q. Chen", None),
    ("Danelle D. Saunders", None),
    ("Emilio William Concepcion", None),
]

table3 = [
    ("[CPT-00350", "[CPT-00350]"),
    ("[CPT-00340", None),
    ("[CPT-11536]", None),
    ("[CPT-115]", None),
]

table4 = [
    ("nextData 12 Street moreInfo 35", "12 Street"),
    ("nextData Main moreInfo 36", "Main"),
    ("nextData Albany Street moreInfo 37", None),
    ("nextData 134 Green Street moreInfo 39", None),
]


@pytest.mark.parametrize("table", [table1, table2, table3, table4])
def test_soundness(table: list):
    inputs = [row[0] for row in table]
    outputs = [row[1] for row in table if row[1] is not None]
    program = learn(inputs, outputs)
    for input_str, output_str in zip(inputs, outputs):
        assert run(program, input_str) == output_str
