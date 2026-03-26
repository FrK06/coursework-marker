"""
Tests for the LangGraph assessment pipeline.

Run with: pytest tests/test_graph.py -v

Coverage:
- GraphState construction and defaults
- Routing logic for all 3 modules
- Placeholder detection
- Auto-REFERRAL trigger
- Grade extraction (4-method cascade)
- Feedback markdown structure
- Full end-to-end integration with mocked LLM
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# GraphState construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphState:
    """Tests for GraphState TypedDict construction."""

    def test_state_construction_minimal(self):
        """GraphState can be constructed with minimal required fields."""
        from src.graph.state import GraphState

        state: GraphState = {
            "module_code": "MLCC",
            "report_chunks": [],
            "report_images": [],
            "ksb_criteria": [],
            "assignment_brief": {},
            "pages_are_accurate": False,
            "evidence_map": {},
            "content_quality": {},
            "ksb_scores": {},
            "overall_recommendation": "",
            "content_warnings": [],
            "ksb_feedback": {},
            "overall_feedback": "",
            "current_ksb_index": 0,
            "errors": [],
            "phase": "retrieval",
        }
        assert state["module_code"] == "MLCC"
        assert state["phase"] == "retrieval"

    def test_state_with_ksb_evidence(self):
        """KSBEvidence can be embedded in evidence_map."""
        from src.graph.state import GraphState, KSBEvidence

        evidence = KSBEvidence(
            ksb_code="K1",
            chunks=[{"content": "test chunk", "metadata": {}, "similarity": 0.5}],
            query_variations=["query 1", "query 2"],
            search_strategy="hybrid",
            total_retrieved=1,
            similarity_scores=[0.5],
        )
        state: GraphState = {
            "module_code": "MLCC",
            "report_chunks": [],
            "report_images": [],
            "ksb_criteria": [],
            "assignment_brief": {},
            "pages_are_accurate": False,
            "evidence_map": {"K1": evidence},
            "content_quality": {},
            "ksb_scores": {},
            "overall_recommendation": "",
            "content_warnings": [],
            "ksb_feedback": {},
            "overall_feedback": "",
            "current_ksb_index": 0,
            "errors": [],
            "phase": "retrieval",
        }
        assert state["evidence_map"]["K1"]["ksb_code"] == "K1"
        assert state["evidence_map"]["K1"]["total_retrieved"] == 1

    def test_state_with_ksb_score(self):
        """KSBScore can be embedded in ksb_scores."""
        from src.graph.state import KSBScore

        score = KSBScore(
            ksb_code="K1",
            grade="PASS",
            confidence="HIGH",
            pass_criteria_met=True,
            merit_criteria_met=False,
            rationale="Meets basic requirements.",
            gaps=["Missing alternative comparison"],
            evidence_strength="moderate",
            placeholder_detected=False,
            adversarial_detected=False,
            extraction_method="json_block",
            raw_llm_response="...",
            audit_trail={},
        )
        assert score["grade"] == "PASS"
        assert score["pass_criteria_met"] is True

    def test_state_with_ksb_feedback(self):
        """KSBFeedback can be embedded in ksb_feedback."""
        from src.graph.state import KSBFeedback

        fb = KSBFeedback(
            ksb_code="K1",
            strengths=["Good problem framing"],
            improvements=["Add alternative comparison"],
            formatted_markdown="### K1: Test -- PASS\n\n**What You Did Well**\n- Good",
        )
        assert fb["ksb_code"] == "K1"
        assert len(fb["strengths"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouting:
    """Tests for conditional edge routing."""

    def test_route_mlcc(self):
        from src.graph.edges.routing import route_by_module

        state = {"module_code": "MLCC"}
        assert route_by_module(state) == "MLCC"

    def test_route_dsp(self):
        from src.graph.edges.routing import route_by_module

        state = {"module_code": "DSP"}
        assert route_by_module(state) == "DSP"

    def test_route_aidi(self):
        from src.graph.edges.routing import route_by_module

        state = {"module_code": "AIDI"}
        assert route_by_module(state) == "AIDI"


# ═══════════════════════════════════════════════════════════════════════════════
# Placeholder detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlaceholderDetection:
    """Tests for weighted placeholder detection."""

    def test_no_placeholders(self):
        from src.prompts.base_templates import detect_placeholders

        chunks = [
            {"content": "The model achieved 95% accuracy on the test set.", "similarity": 0.5},
            {"content": "We used AWS SageMaker for deployment.", "similarity": 0.4},
        ]
        count, detected = detect_placeholders(chunks)
        assert count == 0.0
        assert detected is False

    def test_placeholders_detected(self):
        from src.prompts.base_templates import detect_placeholders

        chunks = [
            {"content": "TBD TBD TBD TBD TBD fill with measured results", "similarity": 0.3},
            {"content": "TODO: add data here. TODO TODO TODO", "similarity": 0.2},
        ]
        count, detected = detect_placeholders(chunks)
        assert count >= 5.0
        assert detected is True

    def test_low_relevance_not_counted(self):
        """Placeholders in low-relevance chunks (similarity < 0.08) should not count."""
        from src.prompts.base_templates import detect_placeholders

        chunks = [
            {"content": "TBD TBD TBD TBD TBD TBD TBD TBD", "similarity": 0.01},
        ]
        count, detected = detect_placeholders(chunks)
        assert count == 0.0
        assert detected is False

    def test_medium_relevance_half_weight(self):
        """Placeholders at 0.08-0.15 similarity should count at 0.5 weight."""
        from src.prompts.base_templates import detect_placeholders

        chunks = [
            {"content": "tbd tbd tbd tbd", "similarity": 0.10},
        ]
        count, detected = detect_placeholders(chunks)
        assert count == 2.0  # 4 occurrences * 0.5 weight
        assert detected is False


# ═══════════════════════════════════════════════════════════════════════════════
# Boilerplate filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoilerplateFiltering:
    """Tests for boilerplate chunk detection."""

    def test_short_chunk_is_boilerplate(self):
        from src.prompts.base_templates import is_boilerplate

        assert is_boilerplate({"content": "Short text"}) is True

    def test_toc_is_boilerplate(self):
        from src.prompts.base_templates import is_boilerplate

        assert is_boilerplate({"content": "Table of Contents\n1. Introduction\n2. Methods\n3. Results\n4. Discussion\n5. Conclusion and more text here"}) is True

    def test_ksb_mapping_grid_is_boilerplate(self):
        from src.prompts.base_templates import is_boilerplate

        content = "| K1 | K2 | K16 | S15 |\n| Task 1 | Task 2 | Task 3 | Task 4 |\n| Evidence | Evidence | Evidence | Evidence |"
        assert is_boilerplate({"content": content}) is True

    def test_normal_content_not_boilerplate(self):
        from src.prompts.base_templates import is_boilerplate

        content = (
            "The model was trained using PyTorch on an AWS EC2 p3.2xlarge instance. "
            "We used a learning rate of 0.001 with Adam optimizer and trained for 50 epochs. "
            "The final accuracy on the held-out test set was 94.2%."
        )
        assert is_boilerplate({"content": content}) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Grade extraction (4-method cascade)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradeExtraction:
    """Tests for the 4-method grade extraction cascade."""

    def test_json_block_extraction(self):
        from src.prompts.base_templates import extract_grade

        evaluation = '''Some analysis here.

```json
{
  "ksb_code": "K1",
  "grade": "PASS",
  "confidence": "HIGH",
  "pass_criteria_met": true,
  "merit_criteria_met": false,
  "evidence_strength": "moderate",
  "gaps": ["Missing alternatives"],
  "key_evidence": ["E1", "E2"],
  "main_gap": "No comparison with baselines"
}
```

Some more text.'''

        result = extract_grade(evaluation)
        assert result["grade"] == "PASS"
        assert result["confidence"] == "HIGH"
        assert result["method"] == "json_block"
        assert result["pass_criteria_met"] is True

    def test_inline_json_extraction(self):
        from src.prompts.base_templates import extract_grade

        evaluation = 'The grade is {"ksb_code": "K2", "grade": "MERIT", "confidence": "MEDIUM"} based on the evidence.'

        result = extract_grade(evaluation)
        assert result["grade"] == "MERIT"
        assert result["method"] == "inline_json"

    def test_regex_extraction(self):
        from src.prompts.base_templates import extract_grade

        evaluation = "After review, the **Grade**: REFERRAL due to missing evidence."

        result = extract_grade(evaluation)
        assert result["grade"] == "REFERRAL"
        assert result["method"] == "regex"

    def test_regex_safety_net_referral_override(self):
        """If regex finds REFERRAL but pass_criteria_met: true is present, override to PASS."""
        from src.prompts.base_templates import extract_grade

        evaluation = '''Grade: REFERRAL
But actually "pass_criteria_met": true and "grade": "PASS" in the detailed analysis.'''

        result = extract_grade(evaluation)
        assert result["grade"] == "PASS"

    def test_heuristic_extraction(self):
        from src.prompts.base_templates import extract_grade

        evaluation = "Requirement 1: NOT MET\nRequirement 2: NOT MET\nRequirement 3: MET"

        result = extract_grade(evaluation)
        assert result["method"] == "heuristic"
        assert result["grade"] == "REFERRAL"
        assert result["confidence"] == "LOW"

    def test_hallucination_detection(self):
        from src.prompts.base_templates import extract_grade

        evaluation = '''The student works at Celestial Solutions, founded in 2019.
```json
{"ksb_code": "K1", "grade": "PASS", "confidence": "HIGH"}
```'''

        result = extract_grade(evaluation)
        assert result["possible_hallucination"] is True

    def test_placeholder_grade_cleaned(self):
        """If LLM outputs template placeholder in grade field, derive from criteria_met."""
        from src.prompts.base_templates import extract_grade

        evaluation = '''```json
{
  "ksb_code": "K1",
  "grade": "PASS|MERIT|REFERRAL",
  "confidence": "HIGH",
  "pass_criteria_met": true,
  "merit_criteria_met": true
}
```'''

        result = extract_grade(evaluation)
        assert result["grade"] == "MERIT"


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-REFERRAL trigger
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoReferral:
    """Tests for auto-REFERRAL logic in specialist nodes."""

    def test_mlcc_auto_referral_ksbs(self):
        from src.prompts.mlcc_templates import MLCC_AUTO_REFERRAL_KSBS

        assert "B5" in MLCC_AUTO_REFERRAL_KSBS
        assert "S23" in MLCC_AUTO_REFERRAL_KSBS
        assert len(MLCC_AUTO_REFERRAL_KSBS) == 2

    def test_dsp_auto_referral_ksbs(self):
        from src.prompts.dsp_templates import DSP_AUTO_REFERRAL_KSBS

        assert "B7" in DSP_AUTO_REFERRAL_KSBS
        assert len(DSP_AUTO_REFERRAL_KSBS) == 1

    def test_aidi_auto_referral_ksbs(self):
        from src.prompts.aidi_templates import AIDI_AUTO_REFERRAL_KSBS

        assert "B3" in AIDI_AUTO_REFERRAL_KSBS
        assert "B4" in AIDI_AUTO_REFERRAL_KSBS
        assert "B8" in AIDI_AUTO_REFERRAL_KSBS
        assert len(AIDI_AUTO_REFERRAL_KSBS) == 3

    def test_auto_referral_score_creation(self):
        from src.graph.nodes._specialist_common import auto_referral_score

        score = auto_referral_score("B5", "adversarial_table")
        assert score["grade"] == "REFERRAL"
        assert score["confidence"] == "HIGH"
        assert score["adversarial_detected"] is True
        assert score["pass_criteria_met"] is False

    def test_auto_referral_no_evidence(self):
        from src.graph.nodes._specialist_common import auto_referral_score

        score = auto_referral_score("K1", "no_evidence")
        assert score["grade"] == "REFERRAL"
        assert score["adversarial_detected"] is False
        assert "no_evidence" in score["gaps"]


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    """Tests for module-specific prompt builders."""

    def test_mlcc_prompt_contains_ksb(self):
        from src.prompts.mlcc_templates import build_mlcc_evaluation_prompt

        prompt = build_mlcc_evaluation_prompt(
            ksb_code="K1",
            ksb_title="ML methodologies",
            pass_criteria="States a clear business problem",
            merit_criteria="Strong problem framing",
            referral_criteria="ML approach unclear",
            evidence_text="Evidence 1: The student used supervised learning.",
            brief_context="Task 1: Technical feasibility study.",
        )
        assert "K1" in prompt
        assert "ML methodologies" in prompt
        assert "ANTI-HALLUCINATION" in prompt
        assert "STEP 1" in prompt

    def test_dsp_prompt_contains_statistical_guidance(self):
        from src.prompts.dsp_templates import build_dsp_evaluation_prompt

        prompt = build_dsp_evaluation_prompt(
            ksb_code="K22",
            ksb_title="Mathematical principles",
            pass_criteria="Correct use of statistics",
            merit_criteria="Strong statistical reasoning",
            referral_criteria="Misinterprets statistics",
            evidence_text="Evidence 1: p-value = 0.03.",
            brief_context="Task 3: Hypothesis testing.",
        )
        assert "K22" in prompt
        assert "effect size" in prompt.lower() or "confidence interval" in prompt.lower()

    def test_aidi_prompt_contains_ethics_guidance(self):
        from src.prompts.aidi_templates import build_aidi_evaluation_prompt

        prompt = build_aidi_evaluation_prompt(
            ksb_code="K9",
            ksb_title="Legal/ethical frameworks",
            pass_criteria="Identifies key legal issues",
            merit_criteria="Applies frameworks with specificity",
            referral_criteria="Ignores requirements",
            evidence_text="Evidence 1: GDPR compliance discussed.",
            brief_context="Task 3: Ethics and compliance.",
        )
        assert "K9" in prompt
        assert "responsible AI" in prompt.lower() or "governance" in prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback markdown structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackStructure:
    """Tests for feedback markdown formatting."""

    def test_feedback_markdown_merit(self):
        from src.graph.nodes.feedback_node import _format_ksb_markdown

        md = _format_ksb_markdown(
            ksb_code="K1",
            ksb_title="ML methodologies",
            grade="MERIT",
            strengths=["Strong problem framing", "Good alternative comparison"],
            improvements=["Could add more quantitative metrics"],
        )
        assert "### K1:" in md
        assert "MERIT" in md
        assert "**What You Did Well**" in md
        assert "**Areas for Development**" in md
        assert "Strong problem framing" in md
        assert "Excellent work" in md

    def test_feedback_markdown_referral(self):
        from src.graph.nodes.feedback_node import _format_ksb_markdown

        md = _format_ksb_markdown(
            ksb_code="B5",
            ksb_title="CPD",
            grade="REFERRAL",
            strengths=[],
            improvements=["HIGH: Add specific CPD evidence"],
        )
        assert "REFERRAL" in md
        assert "resubmit" in md.lower()

    def test_feedback_markdown_pass(self):
        from src.graph.nodes.feedback_node import _format_ksb_markdown

        md = _format_ksb_markdown(
            ksb_code="S15",
            ksb_title="Deployment",
            grade="PASS",
            strengths=["Working PoC demonstrated"],
            improvements=["Add monitoring"],
        )
        assert "PASS" in md
        assert "Merit" in md  # Should suggest aiming for Merit

    def test_fallback_feedback_generation(self):
        from src.graph.nodes.feedback_node import _fallback_ksb_feedback
        from src.graph.state import KSBScore

        score = KSBScore(
            ksb_code="K1",
            grade="PASS",
            confidence="MEDIUM",
            pass_criteria_met=True,
            merit_criteria_met=False,
            rationale="Basic requirements met.",
            gaps=["No alternative comparison"],
            evidence_strength="moderate",
            placeholder_detected=False,
            adversarial_detected=False,
            extraction_method="json_block",
            raw_llm_response="",
            audit_trail={},
        )
        fb = _fallback_ksb_feedback("K1", score)
        assert fb["ksb_code"] == "K1"
        assert len(fb["improvements"]) > 0
        assert "K1" in fb["formatted_markdown"]


# ═══════════════════════════════════════════════════════════════════════════════
# Overall recommendation
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverallRecommendation:
    """Tests for overall recommendation computation."""

    def test_referral_if_any_referral(self):
        from src.graph.nodes._specialist_common import compute_overall_recommendation

        scores = {
            "K1": {"grade": "MERIT"},
            "K2": {"grade": "PASS"},
            "K16": {"grade": "REFERRAL"},
        }
        assert compute_overall_recommendation(scores) == "REFERRAL"

    def test_merit_if_majority(self):
        from src.graph.nodes._specialist_common import compute_overall_recommendation

        scores = {
            "K1": {"grade": "MERIT"},
            "K2": {"grade": "MERIT"},
            "K16": {"grade": "PASS"},
        }
        assert compute_overall_recommendation(scores) == "MERIT"

    def test_pass_otherwise(self):
        from src.graph.nodes._specialist_common import compute_overall_recommendation

        scores = {
            "K1": {"grade": "PASS"},
            "K2": {"grade": "MERIT"},
            "K16": {"grade": "PASS"},
        }
        assert compute_overall_recommendation(scores) == "PASS"

    def test_unknown_if_empty(self):
        from src.graph.nodes._specialist_common import compute_overall_recommendation

        assert compute_overall_recommendation({}) == "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceFormatting:
    """Tests for evidence chunk formatting."""

    def test_format_with_section_numbers(self):
        from src.graph.nodes._specialist_common import format_evidence

        chunks = [
            {
                "content": "The model was deployed on AWS.",
                "metadata": {"section_number": "3.1", "section_title": "Deployment"},
                "similarity": 0.6,
            },
        ]
        result = format_evidence(chunks, pages_are_accurate=False)
        assert "Section 3.1" in result
        assert "SECTION numbers only" in result
        assert "HIGH" in result  # similarity > 0.5

    def test_format_with_page_numbers(self):
        from src.graph.nodes._specialist_common import format_evidence

        chunks = [
            {
                "content": "GDPR compliance was ensured.",
                "metadata": {"section_number": "4", "page_start": 12, "page_end": 12},
                "similarity": 0.35,
            },
        ]
        result = format_evidence(chunks, pages_are_accurate=True)
        assert "Section 4" in result
        assert "page 12" in result
        assert "MEDIUM" in result  # 0.3 < 0.35 <= 0.5

    def test_format_empty_chunks(self):
        from src.graph.nodes._specialist_common import format_evidence

        result = format_evidence([], pages_are_accurate=False)
        assert "No relevant evidence" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Full end-to-end integration test (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """
    Full end-to-end integration test with a 2-page mock report.
    Mocks the OllamaClient to avoid requiring a running Ollama instance.
    """

    MOCK_LLM_RESPONSE = '''### STEP 1: BRIEF REQUIREMENTS CHECK
- [x] Business problem identified
- [x] ML methodology selected

### STEP 2: LIST EVIDENCE FOUND
- [E1] "We selected supervised learning for customer churn prediction" (Section 2) - addresses methodology
- [E2] "The model achieves 92% accuracy on the test set" (Section 3) - addresses evaluation

### STEP 3: ASSESS PASS CRITERIA
| Pass Requirement | Status | Evidence |
|------------------|--------|----------|
| Clear business problem | MET | [E1] |
| Appropriate ML approach | MET | [E1] |

### STEP 4: ASSESS MERIT CRITERIA
| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| Justification vs alternatives | NOT MET | NOT FOUND |

### STEP 5: GRADE DECISION
```json
{
  "ksb_code": "K1",
  "grade": "PASS",
  "confidence": "HIGH",
  "pass_criteria_met": true,
  "merit_criteria_met": false,
  "evidence_strength": "moderate",
  "gaps": ["No alternative comparison"],
  "brief_requirements_met": ["Business problem", "ML approach"],
  "brief_requirements_missing": ["Alternative justification"],
  "key_evidence": ["E1", "E2"],
  "main_gap": "No comparison with baselines"
}
```

### STEP 6: BRIEF JUSTIFICATION
The student clearly identified a business problem and selected an appropriate supervised learning approach. The evidence shows basic understanding but lacks comparison with alternative methods.

### STEP 7: SPECIFIC IMPROVEMENTS
- Compare at least two ML approaches and justify your selection
- Include baseline model comparison results'''

    MOCK_FEEDBACK_RESPONSE = '''STRENGTHS:
- Clear problem framing with business context (Section 2)
- Working model with good accuracy metrics (Section 3)

IMPROVEMENTS:
- HIGH: Compare alternative ML approaches with justification
- MEDIUM: Add baseline model comparison results

CLOSING:
Solid work meeting the core requirements.'''

    MOCK_SUMMARY_RESPONSE = '''## Top 5 Strengths
1. Clear methodology selection
2. Good evaluation metrics

## Top 5 Priority Improvements
1. Add alternative comparisons
2. Improve documentation

## Next Steps
Continue refining the approach for Merit.'''

    def _build_mock_report_chunks(self):
        """Build mock 2-page report chunks."""
        return [
            {
                "content": (
                    "1. Introduction\n"
                    "This report presents a machine learning system for customer churn prediction. "
                    "The business objective is to reduce customer attrition by 15% through early "
                    "intervention. We selected supervised learning as the primary methodology "
                    "given the availability of labelled historical data."
                ),
                "metadata": {
                    "section_number": "1",
                    "section_title": "Introduction",
                    "chunk_index": 0,
                    "token_count": 50,
                    "has_figure_reference": False,
                },
                "chunk_id": "chunk_0",
            },
            {
                "content": (
                    "2. Methodology\n"
                    "We selected supervised learning for customer churn prediction using "
                    "a gradient boosting classifier. The dataset contains 10,000 customer records "
                    "with 25 features. We used an 80/10/10 train/validation/test split."
                ),
                "metadata": {
                    "section_number": "2",
                    "section_title": "Methodology",
                    "chunk_index": 1,
                    "token_count": 45,
                    "has_figure_reference": False,
                },
                "chunk_id": "chunk_1",
            },
            {
                "content": (
                    "3. Results\n"
                    "The model achieves 92% accuracy on the test set with F1 score of 0.89. "
                    "The ROC AUC is 0.94. We deployed the model on AWS SageMaker with "
                    "a REST API endpoint for real-time inference."
                ),
                "metadata": {
                    "section_number": "3",
                    "section_title": "Results",
                    "chunk_index": 2,
                    "token_count": 45,
                    "has_figure_reference": False,
                },
                "chunk_id": "chunk_2",
            },
            {
                "content": (
                    "4. Cloud Architecture\n"
                    "The system uses AWS S3 for data storage, SageMaker for training, "
                    "and ECR for container management. IAM roles restrict access. "
                    "CloudWatch monitors inference latency and error rates."
                ),
                "metadata": {
                    "section_number": "4",
                    "section_title": "Cloud Architecture",
                    "chunk_index": 3,
                    "token_count": 40,
                    "has_figure_reference": False,
                },
                "chunk_id": "chunk_3",
            },
            {
                "content": (
                    "5. Reflection\n"
                    "This project taught me the importance of proper experiment tracking. "
                    "I completed an AWS Cloud Practitioner certification during this project. "
                    "I plan to share my deployment pipeline template with my team."
                ),
                "metadata": {
                    "section_number": "5",
                    "section_title": "Reflection",
                    "chunk_index": 4,
                    "token_count": 40,
                    "has_figure_reference": False,
                },
                "chunk_id": "chunk_4",
            },
        ]

    @pytest.fixture
    def mock_state(self):
        """Build a complete initial state for MLCC with mock report."""
        from src.criteria import get_module_criteria
        from src.brief import get_default_brief

        criteria = get_module_criteria("MLCC")
        brief = get_default_brief("MLCC")
        chunks = self._build_mock_report_chunks()

        ksb_list = [
            {
                "code": c.code,
                "title": c.title,
                "full_description": c.full_description,
                "pass_criteria": c.pass_criteria,
                "merit_criteria": c.merit_criteria,
                "referral_criteria": c.referral_criteria,
                "category": c.category,
            }
            for c in criteria
        ]

        # Build mock evidence map (skip actual retrieval)
        evidence_map = {}
        for c in criteria:
            evidence_map[c.code] = {
                "ksb_code": c.code,
                "chunks": chunks[:3],  # Reuse first 3 chunks as evidence
                "query_variations": ["query1", "query2"],
                "search_strategy": "hybrid",
                "total_retrieved": 3,
                "similarity_scores": [0.6, 0.4, 0.3],
            }

        return {
            "module_code": "MLCC",
            "report_chunks": chunks,
            "report_images": [],
            "ksb_criteria": ksb_list,
            "assignment_brief": brief.to_dict() if brief else {},
            "pages_are_accurate": False,
            "evidence_map": evidence_map,
            "content_quality": {"status": "OK", "adversarial_ksbs": [], "off_topic_count": 0},
            "ksb_scores": {},
            "overall_recommendation": "",
            "content_warnings": [],
            "ksb_feedback": {},
            "overall_feedback": "",
            "current_ksb_index": 0,
            "errors": [],
            "phase": "retrieval",
        }

    @patch("src.graph.nodes._specialist_common.OllamaClient")
    def test_mlcc_specialist_all_ksbs_scored(self, MockOllama, mock_state):
        """MLCC specialist should produce a score for every KSB."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = self.MOCK_LLM_RESPONSE
        MockOllama.return_value = mock_llm

        from src.graph.nodes.mlcc_specialist import mlcc_specialist_node

        result = mlcc_specialist_node(mock_state)

        assert "ksb_scores" in result
        scores = result["ksb_scores"]
        assert len(scores) == 11  # MLCC has 11 KSBs

        # Every KSB should have a valid grade
        for code, score in scores.items():
            assert score["grade"] in ("MERIT", "PASS", "REFERRAL"), f"{code} has invalid grade"

    @patch("src.graph.nodes._specialist_common.OllamaClient")
    def test_dsp_specialist_all_ksbs_scored(self, MockOllama, mock_state):
        """DSP specialist should produce a score for every KSB."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = self.MOCK_LLM_RESPONSE
        MockOllama.return_value = mock_llm

        from src.graph.nodes.dsp_specialist import dsp_specialist_node
        from src.criteria import get_module_criteria
        from src.brief import get_default_brief

        # Rebuild state for DSP
        criteria = get_module_criteria("DSP")
        brief = get_default_brief("DSP")
        mock_state["module_code"] = "DSP"
        mock_state["ksb_criteria"] = [
            {
                "code": c.code, "title": c.title, "full_description": c.full_description,
                "pass_criteria": c.pass_criteria, "merit_criteria": c.merit_criteria,
                "referral_criteria": c.referral_criteria, "category": c.category,
            }
            for c in criteria
        ]
        mock_state["assignment_brief"] = brief.to_dict() if brief else {}

        # Build evidence for DSP KSBs
        chunks = self._build_mock_report_chunks()
        for c in criteria:
            mock_state["evidence_map"][c.code] = {
                "ksb_code": c.code,
                "chunks": chunks[:3],
                "query_variations": ["q1"],
                "search_strategy": "hybrid",
                "total_retrieved": 3,
                "similarity_scores": [0.5, 0.4, 0.3],
            }

        result = dsp_specialist_node(mock_state)
        assert len(result["ksb_scores"]) == 19  # DSP has 19 KSBs

    @patch("src.graph.nodes._specialist_common.OllamaClient")
    def test_aidi_specialist_all_ksbs_scored(self, MockOllama, mock_state):
        """AIDI specialist should produce a score for every KSB."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = self.MOCK_LLM_RESPONSE
        MockOllama.return_value = mock_llm

        from src.graph.nodes.aidi_specialist import aidi_specialist_node
        from src.criteria import get_module_criteria
        from src.brief import get_default_brief

        criteria = get_module_criteria("AIDI")
        brief = get_default_brief("AIDI")
        mock_state["module_code"] = "AIDI"
        mock_state["ksb_criteria"] = [
            {
                "code": c.code, "title": c.title, "full_description": c.full_description,
                "pass_criteria": c.pass_criteria, "merit_criteria": c.merit_criteria,
                "referral_criteria": c.referral_criteria, "category": c.category,
            }
            for c in criteria
        ]
        mock_state["assignment_brief"] = brief.to_dict() if brief else {}

        chunks = self._build_mock_report_chunks()
        for c in criteria:
            mock_state["evidence_map"][c.code] = {
                "ksb_code": c.code,
                "chunks": chunks[:3],
                "query_variations": ["q1"],
                "search_strategy": "hybrid",
                "total_retrieved": 3,
                "similarity_scores": [0.5, 0.4, 0.3],
            }

        result = aidi_specialist_node(mock_state)
        assert len(result["ksb_scores"]) == 19  # AIDI has 19 KSBs

    @patch("src.graph.nodes.feedback_node.OllamaClient")
    def test_feedback_node_generates_for_all_ksbs(self, MockOllama, mock_state):
        """Feedback node should produce feedback for every scored KSB."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [self.MOCK_FEEDBACK_RESPONSE] * 12 + [self.MOCK_SUMMARY_RESPONSE]
        MockOllama.return_value = mock_llm

        from src.graph.nodes.feedback_node import feedback_node
        from src.graph.nodes._specialist_common import auto_referral_score

        # Populate mock scores
        from src.criteria import get_module_criteria
        criteria = get_module_criteria("MLCC")
        for c in criteria:
            mock_state["ksb_scores"][c.code] = {
                "ksb_code": c.code,
                "grade": "PASS",
                "confidence": "HIGH",
                "pass_criteria_met": True,
                "merit_criteria_met": False,
                "rationale": "Meets basic requirements.",
                "gaps": [],
                "evidence_strength": "moderate",
                "placeholder_detected": False,
                "adversarial_detected": False,
                "extraction_method": "json_block",
                "raw_llm_response": self.MOCK_LLM_RESPONSE,
                "audit_trail": {},
            }
        mock_state["overall_recommendation"] = "PASS"

        result = feedback_node(mock_state)

        assert "ksb_feedback" in result
        assert len(result["ksb_feedback"]) == 11
        assert "overall_feedback" in result
        assert len(result["overall_feedback"]) > 0
        assert result["phase"] == "done"

    @patch("src.graph.nodes._specialist_common.OllamaClient")
    def test_adversarial_auto_referral_in_pipeline(self, MockOllama, mock_state):
        """KSBs in auto-REFERRAL set with adversarial flag should get REFERRAL without LLM call."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = self.MOCK_LLM_RESPONSE
        MockOllama.return_value = mock_llm

        from src.graph.nodes.mlcc_specialist import mlcc_specialist_node

        # Flag B5 and S23 as adversarial
        mock_state["content_quality"]["adversarial_ksbs"] = ["B5", "S23"]

        result = mlcc_specialist_node(mock_state)

        # B5 and S23 should be auto-REFERRAL
        assert result["ksb_scores"]["B5"]["grade"] == "REFERRAL"
        assert result["ksb_scores"]["B5"]["adversarial_detected"] is True
        assert result["ksb_scores"]["S23"]["grade"] == "REFERRAL"

        # Other KSBs should still be scored normally
        assert result["ksb_scores"]["K1"]["grade"] in ("MERIT", "PASS", "REFERRAL")

    @patch("src.graph.nodes._specialist_common.OllamaClient")
    def test_no_evidence_gives_referral(self, MockOllama, mock_state):
        """KSBs with no retrieved evidence should get REFERRAL."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = self.MOCK_LLM_RESPONSE
        MockOllama.return_value = mock_llm

        from src.graph.nodes.mlcc_specialist import mlcc_specialist_node

        # Clear evidence for K1
        mock_state["evidence_map"]["K1"] = {
            "ksb_code": "K1",
            "chunks": [],
            "query_variations": [],
            "search_strategy": "hybrid",
            "total_retrieved": 0,
            "similarity_scores": [],
        }

        result = mlcc_specialist_node(mock_state)
        assert result["ksb_scores"]["K1"]["grade"] == "REFERRAL"
        assert "no_evidence" in result["ksb_scores"]["K1"]["gaps"]
