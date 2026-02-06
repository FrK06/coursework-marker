"""
Brief Module - Assignment brief parsing and task extraction.
"""
from .brief_parser import (
    BriefParser,
    AssignmentBrief,
    TaskRequirement,
    get_default_brief,
    parse_uploaded_brief,
    DEFAULT_BRIEFS
)

__all__ = [
    'BriefParser',
    'AssignmentBrief', 
    'TaskRequirement',
    'get_default_brief',
    'parse_uploaded_brief',
    'DEFAULT_BRIEFS'
]
