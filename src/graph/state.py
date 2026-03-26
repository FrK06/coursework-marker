"""
GraphState TypedDict — Central state definition for the LangGraph assessment pipeline.

All nodes read from and write to this shared state. LangGraph merges
returned dicts into the existing state automatically.
"""
from typing import TypedDict, List, Dict, Optional, Any


class KSBEvidence(TypedDict):
    """Evidence retrieved for a single KSB criterion."""
    ksb_code: str
    chunks: List[Dict[str, Any]]        # Retrieved text chunks with content + metadata
    query_variations: List[str]
    search_strategy: str                # "hybrid" | "semantic" | "bm25"
    total_retrieved: int
    similarity_scores: List[float]


class KSBScore(TypedDict):
    """Scoring result for a single KSB criterion."""
    ksb_code: str
    grade: str                          # "MERIT" | "PASS" | "REFERRAL"
    confidence: str                     # "HIGH" | "MEDIUM" | "LOW"
    pass_criteria_met: bool
    merit_criteria_met: bool
    rationale: str
    gaps: List[str]
    evidence_strength: str
    placeholder_detected: bool
    adversarial_detected: bool
    extraction_method: str              # json_block | inline_json | regex | heuristic
    raw_llm_response: str
    audit_trail: Dict[str, Any]


class KSBFeedback(TypedDict):
    """Structured feedback for a single KSB criterion."""
    ksb_code: str
    strengths: List[str]
    improvements: List[str]
    formatted_markdown: str


class GraphState(TypedDict):
    """
    Central state for the LangGraph assessment pipeline.

    Nodes return partial dicts containing only the keys they update;
    LangGraph merges them into this state automatically.
    """
    # ── Input (set before graph invocation) ──────────────────────────
    module_code: str                    # "DSP" | "MLCC" | "AIDI"
    report_chunks: List[Dict]           # All indexed document chunks
    report_images: List[Dict]           # Processed images (base64 + metadata)
    ksb_criteria: List[Dict]            # Rubric criteria for selected module
    assignment_brief: Dict              # Parsed AssignmentBrief.to_dict()
    pages_are_accurate: bool            # True for PDF, False for DOCX

    # ── Retriever node outputs ───────────────────────────────────────
    evidence_map: Dict[str, KSBEvidence]   # ksb_code → evidence
    content_quality: Dict                   # adversarial/off_topic flags
    image_analyses: List[Dict]             # OCR results: [{image_id, caption, ocr_text}]

    # ── Specialist agent outputs ─────────────────────────────────────
    ksb_scores: Dict[str, KSBScore]        # ksb_code → grade + rationale
    overall_recommendation: str
    content_warnings: List[str]

    # ── Feedback agent outputs ───────────────────────────────────────
    ksb_feedback: Dict[str, KSBFeedback]   # ksb_code → feedback
    overall_feedback: str

    # ── Control flow ─────────────────────────────────────────────────
    current_ksb_index: int
    errors: List[str]
    phase: str                             # "retrieval" | "scoring" | "feedback" | "done"
