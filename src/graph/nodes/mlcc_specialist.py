"""
MLCC Specialist Node - Evaluates all 11 MLCC KSBs against rubric criteria.

KSBs: K1, K2, K16, K18, K19, K25, S15, S16, S19, S23, B5

Uses the shared specialist pipeline with MLCC-specific prompts.
Auto-REFERRAL KSBs: {B5, S23}
"""
from ...prompts.mlcc_templates import (
    build_mlcc_evaluation_prompt,
    get_mlcc_system_prompt,
    MLCC_AUTO_REFERRAL_KSBS,
)
from ._specialist_common import run_specialist_pipeline
from ..state import GraphState


def mlcc_specialist_node(state: GraphState) -> dict:
    """
    LangGraph node: Evaluate all MLCC KSBs using MLCC-specific prompts.

    Reads:
        state["ksb_criteria"], state["evidence_map"], state["content_quality"],
        state["assignment_brief"], state["pages_are_accurate"]

    Writes:
        ksb_scores, overall_recommendation, content_warnings, phase, errors
    """
    return run_specialist_pipeline(
        state=state,
        module_code="MLCC",
        build_prompt=build_mlcc_evaluation_prompt,
        get_system_prompt=get_mlcc_system_prompt,
        auto_referral_ksbs=MLCC_AUTO_REFERRAL_KSBS,
    )
