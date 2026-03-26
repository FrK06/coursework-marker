"""
DSP Specialist Node — Evaluates all 19 DSP KSBs against rubric criteria.

KSBs: K2, K5, K15, K20, K22, K24, K26, K27, S1, S9, S10, S13, S17, S18, S21, S22, S26, B3, B7

Uses the shared specialist pipeline with DSP-specific prompts.
Auto-REFERRAL KSBs: {B7}
"""
from ...prompts.dsp_templates import (
    build_dsp_evaluation_prompt,
    get_dsp_system_prompt,
    DSP_AUTO_REFERRAL_KSBS,
)
from ._specialist_common import run_specialist_pipeline
from ..state import GraphState


def dsp_specialist_node(state: GraphState) -> dict:
    """
    LangGraph node: Evaluate all DSP KSBs using DSP-specific prompts.

    Reads:
        state["ksb_criteria"], state["evidence_map"], state["content_quality"],
        state["assignment_brief"], state["pages_are_accurate"]

    Writes:
        ksb_scores, overall_recommendation, content_warnings, phase, errors
    """
    return run_specialist_pipeline(
        state=state,
        module_code="DSP",
        build_prompt=build_dsp_evaluation_prompt,
        get_system_prompt=get_dsp_system_prompt,
        auto_referral_ksbs=DSP_AUTO_REFERRAL_KSBS,
    )
