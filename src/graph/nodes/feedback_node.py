"""
Feedback Node — Generates structured, actionable feedback per KSB
and an overall summary.

Per-KSB process:
1. Read ksb_scores[ksb_code] grade and rationale
2. If REFERRAL → target PASS criteria in feedback
3. If PASS → target MERIT criteria in feedback
4. If MERIT → articulate what made it excellent
5. Call Ollama to generate strengths + improvements
6. Format as structured markdown
7. Write to state["ksb_feedback"][ksb_code]
"""
import logging
from typing import Dict, List, Any

from ...llm.ollama_client import OllamaClient
from ..state import GraphState, KSBScore, KSBFeedback

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, LLMConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback prompt templates
# ═══════════════════════════════════════════════════════════════════════════════

_FEEDBACK_SYSTEM_PROMPT = """You are an academic feedback writer for apprenticeship coursework.
Your feedback must be constructive, specific, and actionable.
ONLY reference evidence and findings from the evaluation provided — do NOT invent content."""

_PER_KSB_FEEDBACK_PROMPT = """Generate structured feedback for this KSB evaluation.

## KSB: {ksb_code} — {ksb_title}
## Grade: {grade} (Confidence: {confidence})

## Evaluation Rationale
{rationale}

## Gaps Identified
{gaps}

## Target for Feedback
{feedback_target}

## Instructions

Generate feedback with:

1. **Strengths** (2-3 bullet points):
   - Reference specific evidence from the evaluation
   - Include section references where available
   - Be specific about what was done well

2. **Areas for Development** (2-3 bullet points):
   - Prioritise improvements (high priority first)
   - Include concrete, actionable examples
   - Reference the assignment brief requirements where relevant

3. **Closing sentence**:
   - Encouraging and appropriate to the grade level

Format your response as:

STRENGTHS:
- [strength 1 with section reference]
- [strength 2 with section reference]

IMPROVEMENTS:
- HIGH: [high priority improvement] (Example: ...)
- MEDIUM: [medium priority improvement] (Example: ...)

CLOSING:
[one encouraging sentence]"""

_OVERALL_SUMMARY_PROMPT = """Generate an overall assessment summary from these KSB results.

## Module: {module_code}
## Overall Recommendation: {overall_recommendation}

## Grade Distribution
{grade_distribution}

## Per-KSB Results
{per_ksb_summary}

## Content Warnings
{content_warnings}

## Instructions

Generate a structured overall summary with:

1. **Grade Distribution Table** — Merit / Pass / Referral counts
2. **Overall Recommendation** — with justification
3. **Top 5 Strengths** — across all KSBs
4. **Top 5 Priority Improvements** — across all KSBs
5. **Next Steps** — conditional on referral count:
   - If referrals > 0: specific remediation guidance
   - If all pass/merit: enhancement suggestions

Keep the summary concise and actionable."""


def feedback_node(state: GraphState) -> dict:
    """
    LangGraph node: Generate structured feedback per KSB and overall summary.

    Reads:
        state["ksb_scores"], state["ksb_criteria"], state["module_code"],
        state["overall_recommendation"], state["content_warnings"]

    Writes:
        ksb_feedback, overall_feedback, phase, errors
    """
    ksb_scores = state.get("ksb_scores", {})
    ksb_criteria = state.get("ksb_criteria", [])
    module_code = state.get("module_code", "UNKNOWN")
    overall_recommendation = state.get("overall_recommendation", "UNKNOWN")
    content_warnings = state.get("content_warnings", [])
    errors = list(state.get("errors", []))

    logger.info(f"Feedback node: generating feedback for {len(ksb_scores)} KSBs")

    # Build criteria lookup
    criteria_lookup = {c["code"]: c for c in ksb_criteria}

    # Initialise LLM
    try:
        llm = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
    except Exception as e:
        logger.error(f"Failed to initialise Ollama for feedback: {e}")
        errors.append(f"Ollama init failed for feedback: {e}")
        # Generate fallback feedback without LLM
        ksb_feedback = _generate_fallback_feedback(ksb_scores, criteria_lookup)
        overall_feedback = _generate_fallback_summary(
            ksb_scores, module_code, overall_recommendation
        )
        return {
            "ksb_feedback": ksb_feedback,
            "overall_feedback": overall_feedback,
            "phase": "feedback",
            "errors": errors,
        }

    ksb_feedback: Dict[str, KSBFeedback] = {}

    # Generate per-KSB feedback
    for ksb_code, score in ksb_scores.items():
        criterion = criteria_lookup.get(ksb_code, {})
        ksb_title = criterion.get("title", ksb_code)

        try:
            feedback = _generate_ksb_feedback(llm, ksb_code, ksb_title, score, criterion)
            ksb_feedback[ksb_code] = feedback
            logger.info(f"  {ksb_code}: feedback generated")
        except Exception as e:
            logger.error(f"  {ksb_code}: feedback generation failed: {e}")
            errors.append(f"Feedback error for {ksb_code}: {e}")
            ksb_feedback[ksb_code] = _fallback_ksb_feedback(ksb_code, score)

    # Generate overall summary
    try:
        overall_feedback = _generate_overall_summary(
            llm, ksb_scores, ksb_feedback, module_code,
            overall_recommendation, content_warnings
        )
    except Exception as e:
        logger.error(f"Overall summary generation failed: {e}")
        errors.append(f"Overall summary error: {e}")
        overall_feedback = _generate_fallback_summary(
            ksb_scores, module_code, overall_recommendation
        )

    logger.info("Feedback node complete")

    return {
        "ksb_feedback": ksb_feedback,
        "overall_feedback": overall_feedback,
        "phase": "done",
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-KSB feedback generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_ksb_feedback(
    llm: OllamaClient,
    ksb_code: str,
    ksb_title: str,
    score: KSBScore,
    criterion: dict,
) -> KSBFeedback:
    """Generate LLM-powered feedback for a single KSB."""
    grade = score["grade"]

    # Determine feedback target based on grade
    if grade == "REFERRAL":
        feedback_target = (
            f"The student received REFERRAL. Target PASS criteria in your feedback.\n"
            f"PASS criteria: {criterion.get('pass_criteria', 'N/A')}\n"
            f"Help the student understand what they need to do to achieve PASS."
        )
    elif grade == "PASS":
        feedback_target = (
            f"The student received PASS. Target MERIT criteria in your feedback.\n"
            f"MERIT criteria: {criterion.get('merit_criteria', 'N/A')}\n"
            f"Help the student understand what they need to do to achieve MERIT."
        )
    else:  # MERIT
        feedback_target = (
            f"The student received MERIT. Articulate what made their work excellent.\n"
            f"MERIT criteria: {criterion.get('merit_criteria', 'N/A')}\n"
            f"Highlight specific strengths that demonstrate merit-level work."
        )

    gaps_text = "\n".join(f"- {g}" for g in score.get("gaps", [])) or "None identified"

    prompt = _PER_KSB_FEEDBACK_PROMPT.format(
        ksb_code=ksb_code,
        ksb_title=ksb_title,
        grade=grade,
        confidence=score.get("confidence", "MEDIUM"),
        rationale=score.get("rationale", "No rationale available."),
        gaps=gaps_text,
        feedback_target=feedback_target,
    )

    raw_response = llm.generate(
        prompt=prompt,
        system_prompt=_FEEDBACK_SYSTEM_PROMPT,
        temperature=LLMConfig.SUMMARY_TEMPERATURE,
        max_tokens=800,
    )

    # Parse the response into structured feedback
    strengths, improvements = _parse_feedback_response(raw_response)

    # Format as markdown
    formatted = _format_ksb_markdown(ksb_code, ksb_title, grade, strengths, improvements)

    return KSBFeedback(
        ksb_code=ksb_code,
        strengths=strengths,
        improvements=improvements,
        formatted_markdown=formatted,
    )


def _parse_feedback_response(response: str) -> tuple[List[str], List[str]]:
    """Parse LLM feedback response into strengths and improvements lists."""
    strengths = []
    improvements = []

    current_section = None
    for line in response.split('\n'):
        line = line.strip()

        if 'STRENGTHS' in line.upper() or 'WHAT YOU DID WELL' in line.upper():
            current_section = 'strengths'
            continue
        elif 'IMPROVEMENTS' in line.upper() or 'AREAS FOR DEVELOPMENT' in line.upper():
            current_section = 'improvements'
            continue
        elif 'CLOSING' in line.upper():
            current_section = None
            continue

        if line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            if text and current_section == 'strengths':
                strengths.append(text)
            elif text and current_section == 'improvements':
                improvements.append(text)

    # Fallback: if parsing failed, split response roughly
    if not strengths and not improvements:
        lines = [l.strip() for l in response.split('\n') if l.strip().startswith('-')]
        mid = len(lines) // 2
        strengths = [l[2:].strip() for l in lines[:max(mid, 1)]]
        improvements = [l[2:].strip() for l in lines[max(mid, 1):]]

    return strengths[:3], improvements[:3]


def _format_ksb_markdown(
    ksb_code: str,
    ksb_title: str,
    grade: str,
    strengths: List[str],
    improvements: List[str],
) -> str:
    """Format KSB feedback as structured markdown."""
    lines = [f"### {ksb_code}: {ksb_title} — {grade}", ""]

    lines.append("**What You Did Well**")
    for s in strengths:
        lines.append(f"- {s}")
    if not strengths:
        lines.append("- No specific strengths identified in the evidence.")
    lines.append("")

    lines.append("**Areas for Development**")
    for imp in improvements:
        # Add priority emoji based on content
        if imp.startswith("HIGH:"):
            lines.append(f"- \U0001f534 {imp[5:].strip()}")
        elif imp.startswith("MEDIUM:"):
            lines.append(f"- \U0001f7e1 {imp[7:].strip()}")
        else:
            lines.append(f"- {imp}")
    if not improvements:
        lines.append("- No specific improvements needed — excellent work.")
    lines.append("")

    # Closing sentence
    if grade == "MERIT":
        lines.append("Excellent work demonstrating strong understanding and application.")
    elif grade == "PASS":
        lines.append("Solid work meeting the requirements. Review the suggestions above to aim for Merit.")
    else:
        lines.append("Please review the areas above carefully and resubmit with the suggested improvements.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Overall summary generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_overall_summary(
    llm: OllamaClient,
    ksb_scores: Dict[str, KSBScore],
    ksb_feedback: Dict[str, KSBFeedback],
    module_code: str,
    overall_recommendation: str,
    content_warnings: List[str],
) -> str:
    """Generate LLM-powered overall assessment summary."""
    # Build grade distribution
    grades = [s["grade"] for s in ksb_scores.values()]
    merit_count = grades.count("MERIT")
    pass_count = grades.count("PASS")
    referral_count = grades.count("REFERRAL")

    grade_distribution = (
        f"- Merit: {merit_count}\n"
        f"- Pass: {pass_count}\n"
        f"- Referral: {referral_count}\n"
        f"- Total: {len(grades)}"
    )

    # Build per-KSB summary
    per_ksb_lines = []
    for code, score in ksb_scores.items():
        per_ksb_lines.append(
            f"- {code}: {score['grade']} (confidence: {score['confidence']}) — "
            f"{score.get('rationale', 'N/A')[:100]}"
        )
    per_ksb_summary = "\n".join(per_ksb_lines)

    warnings_text = "\n".join(f"- {w}" for w in content_warnings) or "None"

    prompt = _OVERALL_SUMMARY_PROMPT.format(
        module_code=module_code,
        overall_recommendation=overall_recommendation,
        grade_distribution=grade_distribution,
        per_ksb_summary=per_ksb_summary,
        content_warnings=warnings_text,
    )

    raw_response = llm.generate(
        prompt=prompt,
        system_prompt=_FEEDBACK_SYSTEM_PROMPT,
        temperature=LLMConfig.SUMMARY_TEMPERATURE,
        max_tokens=1200,
    )

    # Prepend the grade distribution table
    header = (
        f"## Assessment Summary — {module_code}\n\n"
        f"| Grade | Count |\n"
        f"|-------|-------|\n"
        f"| Merit | {merit_count} |\n"
        f"| Pass | {pass_count} |\n"
        f"| Referral | {referral_count} |\n"
        f"| **Total** | **{len(grades)}** |\n\n"
        f"**Overall Recommendation: {overall_recommendation}**\n\n"
        f"---\n\n"
    )

    return header + raw_response


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback feedback (when LLM is unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

def _fallback_ksb_feedback(ksb_code: str, score: KSBScore) -> KSBFeedback:
    """Generate basic feedback without LLM."""
    grade = score["grade"]
    gaps = score.get("gaps", [])

    if grade == "MERIT":
        strengths = ["Strong evidence demonstrating merit-level understanding."]
        improvements = ["Continue to maintain this high standard."]
    elif grade == "PASS":
        strengths = ["Basic requirements met with adequate evidence."]
        improvements = [f"Address: {g}" for g in gaps[:2]] or ["Review merit criteria for enhancement."]
    else:
        strengths = ["Some relevant content identified."]
        improvements = [f"Address: {g}" for g in gaps[:2]] or ["Significant gaps need to be addressed."]

    formatted = _format_ksb_markdown(ksb_code, ksb_code, grade, strengths, improvements)

    return KSBFeedback(
        ksb_code=ksb_code,
        strengths=strengths,
        improvements=improvements,
        formatted_markdown=formatted,
    )


def _generate_fallback_feedback(
    ksb_scores: Dict[str, KSBScore],
    criteria_lookup: dict,
) -> Dict[str, KSBFeedback]:
    """Generate fallback feedback for all KSBs without LLM."""
    feedback = {}
    for ksb_code, score in ksb_scores.items():
        feedback[ksb_code] = _fallback_ksb_feedback(ksb_code, score)
    return feedback


def _generate_fallback_summary(
    ksb_scores: Dict[str, KSBScore],
    module_code: str,
    overall_recommendation: str,
) -> str:
    """Generate basic summary without LLM."""
    grades = [s["grade"] for s in ksb_scores.values()]
    return (
        f"## Assessment Summary — {module_code}\n\n"
        f"- Merit: {grades.count('MERIT')}\n"
        f"- Pass: {grades.count('PASS')}\n"
        f"- Referral: {grades.count('REFERRAL')}\n\n"
        f"**Overall Recommendation: {overall_recommendation}**\n\n"
        f"*Detailed feedback unavailable — LLM was not accessible.*"
    )
