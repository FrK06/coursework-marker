"""
Conditional edge routing — directs flow from retriever to the correct
module specialist based on state["module_code"].
"""
from ..state import GraphState


def route_by_module(state: GraphState) -> str:
    """
    Conditional edge: route to the correct specialist node.

    Returns one of "DSP", "MLCC", or "AIDI" which maps to the
    corresponding specialist node name in the graph.
    """
    return state["module_code"]
