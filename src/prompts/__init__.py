"""
Prompts Module - KSB evaluation prompt templates with brief context support.
"""
from .ksb_templates import KSBPromptTemplates, extract_grade_from_evaluation

__all__ = ["KSBPromptTemplates", "extract_grade_from_evaluation"]
