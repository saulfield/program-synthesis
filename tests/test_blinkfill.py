import pytest

from synthesis.blinkfill import learn, run

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
def test_soundness(table):
    inputs = [row[0] for row in table]
    outputs = [row[1] for row in table if row[1] is not None]
    program = learn(inputs, outputs)
    for input_str, output_str in zip(inputs, outputs):
        assert run(program, input_str) == output_str
