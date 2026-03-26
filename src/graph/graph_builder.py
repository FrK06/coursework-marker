"""
Graph Builder — Constructs and compiles the LangGraph StateGraph.

Graph flow:
    START -> retriever -> route_by_module -> {DSP|MLCC|AIDI}_specialist -> feedback -> END
"""
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes.retriever_node import retriever_node
from .nodes.mlcc_specialist import mlcc_specialist_node
from .nodes.dsp_specialist import dsp_specialist_node
from .nodes.aidi_specialist import aidi_specialist_node
from .nodes.feedback_node import feedback_node
from .edges.routing import route_by_module

logger = logging.getLogger(__name__)


def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph for KSB assessment.

    Returns:
        Compiled graph ready for .invoke() or .stream()
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("retriever", retriever_node)
    graph.add_node("DSP", dsp_specialist_node)
    graph.add_node("MLCC", mlcc_specialist_node)
    graph.add_node("AIDI", aidi_specialist_node)
    graph.add_node("feedback", feedback_node)

    # ── Wire edges ──────────────────────────────────────────────────
    graph.add_edge(START, "retriever")

    graph.add_conditional_edges(
        "retriever",
        route_by_module,
        {
            "DSP": "DSP",
            "MLCC": "MLCC",
            "AIDI": "AIDI",
        },
    )

    graph.add_edge("DSP", "feedback")
    graph.add_edge("MLCC", "feedback")
    graph.add_edge("AIDI", "feedback")
    graph.add_edge("feedback", END)

    # ── Compile ─────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("LangGraph assessment pipeline compiled successfully")

    return compiled
