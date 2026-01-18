"""
Prompts Module - Carefully designed prompt templates for academic marking.
"""
from .templates import PromptTemplates
from .ksb_templates import (
    KSBPromptTemplates, 
    extract_grade_from_evaluation,
    validate_citations
)

__all__ = [
    "PromptTemplates",
    "KSBPromptTemplates",
    "extract_grade_from_evaluation",
    "validate_citations"
]