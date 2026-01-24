"""
Criteria processing module for KSB-based assessment.
"""
from .ksb_parser import (
    KSBCriterion,
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)

__all__ = [
    'KSBCriterion',
    'get_module_criteria',
    'get_available_modules',
    'AVAILABLE_MODULES'
]
