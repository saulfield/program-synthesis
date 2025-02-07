from pathlib import Path
import graphviz

from synthesis.blinkfill.dag import Dag
from synthesis.blinkfill.input_data_graph import InputDataGraph


def draw_idg(idg: InputDataGraph) -> graphviz.Digraph:  # pragma: no cover
    dot = graphviz.Digraph(name="InputDataGraph")
    with dot.subgraph() as sg:  # type: ignore
        sg.node_attr.update(shape="circle")
        sg.edge_attr.update(arrowsize="0.5")
        for v in idg.g.nodes:
            sg.node(str(v))
            for vi in idg.g.outgoing[v]:
                label = repr(idg.edge_data[(v, vi)])
                sg.edge(str(v), str(vi), label=f"{label}")
    return dot


def draw_dag(dag: Dag) -> graphviz.Digraph:  # pragma: no cover
    dot = graphviz.Digraph(name="DAG")
    with dot.subgraph() as sg:  # type: ignore
        sg.node_attr.update(shape="circle")
        sg.edge_attr.update(arrowsize="0.5")
        for v in dag.g.nodes:
            sg.node(str(v))
            for vi in dag.g.outgoing[v]:
                label = repr(set(dag.edge_data[(v, vi)]))
                sg.edge(str(v), str(vi), label=f"{label}")
    return dot


def graphviz_render(dot: graphviz.Digraph, file_path: Path | None = None):  # pragma: no cover
    from IPython.core.getipython import get_ipython
    from IPython.display import display_svg

    assert dot.name is not None, "Name required for graphviz object"
    filename = file_path if file_path else Path(__file__).parent / dot.name
    dot.render(filename=filename, cleanup=True, format="svg")
    if get_ipython():
        display_svg(dot._repr_image_svg_xml(), raw=True)
    print(f"Diagram: file://{filename}.svg")
