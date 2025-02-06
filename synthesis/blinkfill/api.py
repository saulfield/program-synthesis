from synthesis.blinkfill.dag import best_path, gen_dag, gen_program
from synthesis.blinkfill.dsl import Concat, eval_program
from synthesis.blinkfill.input_data_graph import gen_input_data_graph
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Program:
    expr: Concat


def learn(inputs: list[str], outputs: list[str]) -> Program:
    idg = gen_input_data_graph(inputs)
    dag = gen_dag(idg, inputs, outputs)
    exprs = best_path(idg, dag)
    program = gen_program(idg, exprs)
    return Program(program)


def run(program: Program, s: str) -> str:
    return eval_program(program.expr, s)
