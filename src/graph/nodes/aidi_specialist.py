"""
AIDI Specialist Node — Evaluates all 19 AIDI KSBs against rubric criteria.

KSBs: K1, K4, K5, K6, K8, K9, K11, K12, K21, K24, K29, S3, S5, S6, S25, S26, B3, B4, B8

Uses the shared specialist pipeline with AIDI-specific prompts.
Auto-REFERRAL KSBs: {B3, B4, B8}
"""
from ...prompts.aidi_templates import (
    build_aidi_evaluation_prompt,
    get_aidi_system_prompt,
    AIDI_AUTO_REFERRAL_KSBS,
)
from ._specialist_common import run_specialist_pipeline
from ..state import GraphState


def aidi_specialist_node(state: GraphState) -> dict:
    """
    LangGraph node: Evaluate all AIDI KSBs using AIDI-specific prompts.

    Reads:
        state["ksb_criteria"], state["evidence_map"], state["content_quality"],
        state["assignment_brief"], state["pages_are_accurate"]

    Writes:
        ksb_scores, overall_recommendation, content_warnings, phase, errors
    """
    return run_specialist_pipeline(
        state=state,
        module_code="AIDI",
        build_prompt=build_aidi_evaluation_prompt,
        get_system_prompt=get_aidi_system_prompt,
        auto_referral_ksbs=AIDI_AUTO_REFERRAL_KSBS,
    )
