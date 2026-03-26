"""
Shared helper functions used by all specialist nodes (MLCC, DSP, AIDI).

Extracted to avoid duplicating ~150 lines across three specialist files.
Each specialist imports these and calls run_specialist_pipeline() with
its own module-specific prompt builder and auto-REFERRAL set.
"""
import re
import logging
from typing import Dict, Any, List, Callable, Set, Optional

from ...llm.ollama_client import OllamaClient
from ...validation.output_validator import OutputValidator
from ...prompts.base_templates import extract_grade, detect_placeholders
from ...brief.brief_parser import AssignmentBrief, TaskRequirement, get_default_brief
from ..state import GraphState, KSBScore

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, ModelConfig, LLMConfig

logger = logging.getLogger(__name__)


# Type alias for the module-specific prompt builder function
PromptBuilder = Callable[
    [str, str, str, str, str, str, str, str],  # ksb_code, title, pass, merit, referral, evidence, brief, image
    str,
]


def run_specialist_pipeline(
    state: GraphState,
    module_code: str,
    build_prompt: PromptBuilder,
    get_system_prompt: Callable[[], str],
    auto_referral_ksbs: Set[str],
) -> dict:
    """
    Generic specialist pipeline shared by DSP, MLCC, and AIDI nodes.

    Args:
        state: Current graph state.
        module_code: "DSP" | "MLCC" | "AIDI".
        build_prompt: Module-specific prompt builder function.
        get_system_prompt: Returns the system prompt string.
        auto_referral_ksbs: Set of KSB codes that should auto-REFERRAL on adversarial content.

    Returns:
        Partial state dict with ksb_scores, overall_recommendation, content_warnings, phase, errors.
    """
    ksb_criteria = state["ksb_criteria"]
    evidence_map = state.get("evidence_map", {})
    content_quality = state.get("content_quality", {})
    brief_dict = state.get("assignment_brief", {})
    pages_are_accurate = state.get("pages_are_accurate", False)
    image_analyses = state.get("image_analyses", [])
    errors = list(state.get("errors", []))

    logger.info(f"{module_code} specialist: evaluating {len(ksb_criteria)} KSBs")

    # Initialise LLM client
    try:
        model_config = ModelConfig.get_model_config(OLLAMA_MODEL)
        llm = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
    except Exception as e:
        logger.error(f"Failed to initialise Ollama: {e}")
        errors.append(f"Ollama init failed: {e}")
        return {
            "ksb_scores": {},
            "overall_recommendation": "ERROR",
            "content_warnings": [f"LLM unavailable: {e}"],
            "phase": "scoring",
            "errors": errors,
        }

    validator = OutputValidator(module_code=module_code)
    system_prompt = get_system_prompt()
    brief = reconstruct_brief(brief_dict)

    ksb_scores: Dict[str, KSBScore] = {}
    content_warnings: List[str] = []
    adversarial_ksbs = set(content_quality.get("adversarial_ksbs", []))

    for criterion in ksb_criteria:
        ksb_code = criterion["code"]
        ksb_title = criterion.get("title", "")

        logger.info(f"  Evaluating {ksb_code}: {ksb_title}")

        # Step 1: adversarial check
        if ksb_code in adversarial_ksbs and ksb_code in auto_referral_ksbs:
            logger.warning(f"  {ksb_code}: adversarial content detected -> auto-REFERRAL")
            content_warnings.append(
                f"{ksb_code}: Auto-REFERRAL due to adversarial reflection table"
            )
            ksb_scores[ksb_code] = auto_referral_score(ksb_code, "adversarial_table")
            continue

        # Step 2: gather evidence
        evidence = evidence_map.get(ksb_code, {})
        chunks = evidence.get("chunks", [])

        if not chunks:
            logger.warning(f"  {ksb_code}: no evidence found -> REFERRAL")
            ksb_scores[ksb_code] = auto_referral_score(ksb_code, "no_evidence")
            continue

        # Step 3: placeholders
        weighted_placeholder_count, placeholder_detected = detect_placeholders(chunks)

        # Step 4: build prompt
        evidence_text = format_evidence(chunks, pages_are_accurate)
        brief_context = get_brief_context(brief, ksb_code)

        # Build image context from OCR results
        image_context = _build_image_context(image_analyses)

        prompt = build_prompt(
            ksb_code=ksb_code,
            ksb_title=ksb_title,
            pass_criteria=criterion.get("pass_criteria", ""),
            merit_criteria=criterion.get("merit_criteria", ""),
            referral_criteria=criterion.get("referral_criteria", ""),
            evidence_text=evidence_text,
            brief_context=brief_context,
            image_context=image_context,
        )

        # Step 5: call LLM
        try:
            temperature = model_config.get("temperature", LLMConfig.EVALUATION_TEMPERATURE)
            max_tokens = model_config.get("max_tokens", LLMConfig.MAX_OUTPUT_TOKENS)
            raw_response = llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"  {ksb_code}: LLM call failed: {e}")
            errors.append(f"LLM error for {ksb_code}: {e}")
            ksb_scores[ksb_code] = auto_referral_score(ksb_code, "llm_error")
            continue

        # Step 6: extract grade
        grade_result = extract_grade(raw_response)

        # Step 7: validate
        rubric_text = (
            f"PASS: {criterion.get('pass_criteria', '')}\n"
            f"MERIT: {criterion.get('merit_criteria', '')}"
        )
        validation = validator.validate_evaluation(
            evaluation_text=raw_response,
            evidence_text=evidence_text,
            ksb_code=ksb_code,
            rubric_text=rubric_text,
            template_text=prompt,
        )
        if not validation.is_valid:
            logger.warning(
                f"  {ksb_code}: validation failed -- "
                f"errors={validation.errors}, action={validation.suggested_action}"
            )

        # Step 8: grade caps
        grade = grade_result["grade"]
        confidence = grade_result["confidence"]

        if placeholder_detected and grade == "MERIT":
            logger.info(
                f"  {ksb_code}: placeholder count {weighted_placeholder_count:.1f} >= 5 "
                f"-> capping MERIT to PASS"
            )
            grade = "PASS"

        if (grade == "REFERRAL" and confidence == "LOW" and
                0 < weighted_placeholder_count < 5):
            logger.info(f"  {ksb_code}: LOW confidence REFERRAL with some placeholders -> upgrading to PASS")
            grade = "PASS"

        # Step 9: write score
        ksb_scores[ksb_code] = KSBScore(
            ksb_code=ksb_code,
            grade=grade,
            confidence=confidence,
            pass_criteria_met=grade_result.get("pass_criteria_met", grade != "REFERRAL"),
            merit_criteria_met=grade_result.get("merit_criteria_met", grade == "MERIT"),
            rationale=extract_rationale(raw_response),
            gaps=grade_result.get("gaps", []),
            evidence_strength=grade_result.get("evidence_strength", "unknown"),
            placeholder_detected=placeholder_detected,
            adversarial_detected=ksb_code in adversarial_ksbs,
            extraction_method=grade_result["method"],
            raw_llm_response=raw_response,
            audit_trail={
                "validation_confidence": validation.confidence_score,
                "validation_action": validation.suggested_action,
                "validation_warnings": validation.warnings,
                "validation_errors": validation.errors,
                "placeholder_count": weighted_placeholder_count,
                "evidence_chunks": len(chunks),
                "query_variations": len(evidence.get("query_variations", [])),
            },
        )

        logger.info(
            f"  {ksb_code}: {grade} (confidence={confidence}, "
            f"method={grade_result['method']}, "
            f"validation={validation.suggested_action})"
        )

    overall = compute_overall_recommendation(ksb_scores)

    logger.info(
        f"{module_code} specialist complete: {len(ksb_scores)} KSBs scored, "
        f"recommendation={overall}"
    )

    return {
        "ksb_scores": ksb_scores,
        "overall_recommendation": overall,
        "content_warnings": content_warnings,
        "phase": "scoring",
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def auto_referral_score(ksb_code: str, reason: str) -> KSBScore:
    """Create an auto-REFERRAL KSBScore."""
    return KSBScore(
        ksb_code=ksb_code,
        grade="REFERRAL",
        confidence="HIGH",
        pass_criteria_met=False,
        merit_criteria_met=False,
        rationale=f"Auto-REFERRAL: {reason}",
        gaps=[reason],
        evidence_strength="none",
        placeholder_detected=False,
        adversarial_detected=(reason == "adversarial_table"),
        extraction_method="auto",
        raw_llm_response="",
        audit_trail={"auto_referral_reason": reason},
    )


def format_evidence(chunks: list, pages_are_accurate: bool) -> str:
    """Format retrieved chunks into evidence text for the LLM prompt."""
    if not chunks:
        return "No relevant evidence found in the report for this criterion."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        location_parts = []
        section_number = metadata.get("section_number", "")
        section_title = metadata.get("section_title", "")

        if section_number:
            location_parts.append(f"Section {section_number}")
        elif section_title:
            num_match = re.match(r'^(\d+(?:\.\d+)*)', section_title)
            if num_match:
                location_parts.append(f"Section {num_match.group(1)}")
            elif section_title.strip():
                location_parts.append(f"Section: {section_title[:40].strip()}")

        if pages_are_accurate:
            page_start = metadata.get("page_start", 0)
            page_end = metadata.get("page_end", page_start)
            if page_start and page_start > 0:
                if page_start == page_end:
                    location_parts.append(f"page {page_start}")
                else:
                    location_parts.append(f"pages {page_start}-{page_end}")

        if not location_parts:
            chunk_index = metadata.get("chunk_index", i - 1)
            location_parts.append(f"Chunk {chunk_index + 1}")

        location_str = " / ".join(location_parts)
        similarity = chunk.get("similarity", 0)
        relevance = "HIGH" if similarity > 0.5 else "MEDIUM" if similarity > 0.3 else "LOW"

        parts.append(
            f"Evidence {i} ({location_str}) [Relevance: {relevance}]:\n{content}"
        )

    if pages_are_accurate:
        header = (
            "CITATION RULES:\n"
            "- Cite using Section AND page numbers shown below\n"
            "- Only cite locations shown in the evidence headers\n"
            "- Do NOT invent section or page numbers\n\n"
        )
    else:
        header = (
            "CITATION RULES:\n"
            "- Cite using SECTION numbers only (e.g., 'Section 3')\n"
            "- Do NOT cite page numbers - they are not available for this document\n"
            "- Only cite sections shown in the evidence headers below\n\n"
        )

    return header + "\n\n---\n\n".join(parts)


def _build_image_context(image_analyses: List[Dict]) -> str:
    """Build image context string from OCR analyses for inclusion in prompts."""
    if not image_analyses:
        return ""

    lines = ["\n## IMAGE EVIDENCE (extracted via OCR)\n"]
    for img in image_analyses:
        caption = img.get("caption", "no caption")
        lines.append(f"**{img['image_id']}** ({caption}):")
        lines.append(img["ocr_text"])
        lines.append("")

    return "\n".join(lines)


def get_brief_context(brief: Any, ksb_code: str) -> str:
    """Get assignment brief context for a specific KSB."""
    if brief is None:
        return f"This KSB ({ksb_code}) should be demonstrated across the submission."
    if hasattr(brief, 'get_context_for_ksb'):
        return brief.get_context_for_ksb(ksb_code)
    return f"This KSB ({ksb_code}) should be demonstrated across the submission."


def reconstruct_brief(brief_dict: dict) -> Optional[AssignmentBrief]:
    """Reconstruct AssignmentBrief from dict, or return None."""
    if not brief_dict:
        return None
    try:
        module_code = brief_dict.get("module_code", "")
        if module_code:
            default = get_default_brief(module_code)
            if default:
                return default

        tasks = []
        for t in brief_dict.get("tasks", []):
            tasks.append(TaskRequirement(
                task_number=t.get("task_number", 0),
                task_title=t.get("task_title", ""),
                description=t.get("description", ""),
                deliverables=t.get("deliverables", []),
                mapped_ksbs=t.get("mapped_ksbs", []),
                word_count=t.get("word_count"),
                weighting=t.get("weighting"),
            ))

        return AssignmentBrief(
            module_code=module_code,
            module_title=brief_dict.get("module_title", ""),
            tasks=tasks,
            overall_requirements=brief_dict.get("overall_requirements", ""),
            submission_guidelines=brief_dict.get("submission_guidelines", ""),
            ksb_task_mapping=brief_dict.get("ksb_task_mapping", {}),
            raw_text=brief_dict.get("raw_text", ""),
        )
    except Exception as e:
        logger.warning(f"Failed to reconstruct brief: {e}")
        return None


def extract_rationale(raw_response: str) -> str:
    """Extract the BRIEF JUSTIFICATION section from the LLM response."""
    patterns = [
        r'(?:STEP 6|BRIEF JUSTIFICATION)[:\s]*\n(.*?)(?=\n###|\n## |STEP 7|SPECIFIC IMPROVEMENTS|\Z)',
        r'(?:Justification)[:\s]*\n(.*?)(?=\n###|\n## |\Z)',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            rationale = match.group(1).strip()
            if len(rationale) > 20:
                return rationale[:1000]

    lines = raw_response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if len(line) > 50 and not line.startswith('#') and not line.startswith('|'):
            return line[:500]

    return "No rationale extracted."


def compute_overall_recommendation(ksb_scores: Dict[str, KSBScore]) -> str:
    """
    Compute overall recommendation.
    - REFERRAL if any KSB is REFERRAL
    - MERIT if > 50% of KSBs are MERIT
    - PASS otherwise
    """
    if not ksb_scores:
        return "UNKNOWN"

    grades = [score["grade"] for score in ksb_scores.values()]
    referral_count = grades.count("REFERRAL")
    merit_count = grades.count("MERIT")
    total = len(grades)

    if referral_count > 0:
        return "REFERRAL"
    elif merit_count > total / 2:
        return "MERIT"
    else:
        return "PASS"
