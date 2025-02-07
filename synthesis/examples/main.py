# %%
import time
from pathlib import Path

from synthesis.blinkfill import learn, run
from synthesis.blinkfill.dag import gen_dag
from synthesis.blinkfill.input_data_graph import gen_idg
from synthesis.blinkfill.visualize import draw_dag, draw_idg, graphviz_render


def blinkfill_example():
    inputs = [
        "Mumbai, India",
        "Los Angeles, United States of America",
        "Newark, United States",
        "New York, United States of America",
        "Wellington, New Zealand",
        "New Delhi, India",
    ]
    outputs = [
        "India",
        "United States of America",
    ]

    start = time.perf_counter()
    program = learn(inputs, outputs)
    learn_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    result = run(program, "Newark, United States")
    run_elapsed = time.perf_counter() - start

    print(program)
    print(f'Output: "{result}"')
    print(f"Learn took: {learn_elapsed:0.2f}s")
    print(f"Run took: {run_elapsed:0.2f}s")


def visualize():
    # inputs = [
    #     "1 lb",
    #     "23 g",
    #     "4 tons",
    #     "102 grams",
    #     "75 kg",
    # ]
    inputs = [
        "Mumbai, India",
        "Los Angeles, United States of America",
        "Newark, United States",
        "New York, United States of America",
        "Wellington, New Zealand",
        "New Delhi, India",
    ]
    outputs = [
        "India",
        # "United States of America",
    ]
    idg = gen_idg(inputs)
    dag = gen_dag(idg, inputs, outputs)
    idg_dot = draw_idg(idg)
    dag_dot = draw_dag(dag)

    base_path = Path(__file__).parent
    graphviz_render(idg_dot, base_path / "InputDataGraph")
    graphviz_render(dag_dot, base_path / "DAG")


if __name__ == "__main__":
    # blinkfill_example()
    visualize()
