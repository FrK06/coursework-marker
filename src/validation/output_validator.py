"""
Output Validator - Detects potential hallucinations in LLM evaluation output.

This module provides validation to catch fabricated content before it reaches students.
"""
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    warnings: List[str]
    errors: List[str]
    suggested_action: str  # 'accept', 'flag_for_review', 'reject'


class OutputValidator:
    """
    Validates LLM evaluation output for potential hallucinations.
    
    IMPORTANT: This is a defense-in-depth measure. The primary solution
    is to use a sufficiently capable model (7B+ parameters minimum).
    """
    
    # Known hallucination patterns from testing
    HALLUCINATION_PATTERNS = [
        # Fabricated company names
        (r'\b(celestial\s+solutions?|rovio|acme|contoso)\b', 'Likely fabricated company name'),
        
        # Company bio patterns (the model tends to generate these when uncertain)
        (r'(founded|headquarters|website):\s*\w+', 'Company bio pattern - likely fabricated'),
        (r'founded\s+in\s+\d{4}', 'Contains founding date - likely fabricated'),
        (r'as\s+of\s+(Q[1-4]\s+)?\d{4}', 'Contains date reference - verify against evidence'),
        
        # Conversational patterns (model breaking character)
        (r'here\s+are\s+some\s+key\s+facts', 'Conversational pattern - not assessment language'),
        (r'do\s+you\s+want\s+me\s+to\s+(provide|tell|explain)', 'Conversational pattern'),
        (r"let's\s+(discuss|talk|explore)", 'Conversational pattern'),
        
        # Citation patterns that suggest fabrication
        (r'\(page\s+\d{3,}\)', 'Page number too high - likely fabricated'),
        (r'\[E\d{2,}\]', 'Evidence reference too high - verify evidence count'),
        
        # Non-existent KSB codes
        (r'\b[KSB]\d{3,}\b', 'Invalid KSB code format'),
        
        # Content that doesn't belong in assessment
        (r'(angry\s+birds|minecraft|fortnite|netflix)', 'Unrelated content detected'),
    ]
    
    # Non-ASCII patterns that indicate model instability
    NON_ASCII_SCRIPTS = [
        (r'[\u0900-\u097F]', 'Devanagari script detected'),
        (r'[\u0C00-\u0C7F]', 'Telugu script detected'),
        (r'[\u0600-\u06FF]', 'Arabic script detected'),
        (r'[\u4E00-\u9FFF]', 'Chinese characters detected'),
        (r'[\u3040-\u309F\u30A0-\u30FF]', 'Japanese script detected'),
    ]
    
    # Valid KSB codes by module
    VALID_KSBS = {
        'DSP': {'K2', 'K5', 'K15', 'K20', 'K22', 'K24', 'K26', 'K27',
                'S1', 'S9', 'S10', 'S13', 'S17', 'S18', 'S21', 'S22', 'S26',
                'B3', 'B7'},
        'MLCC': {'K1', 'K2', 'K16', 'K18', 'K19', 'K25',
                 'S15', 'S16', 'S19', 'S23', 'B5'},
        'AIDI': {'K1', 'K4', 'K5', 'K6', 'K8', 'K9', 'K11', 'K12', 'K21', 'K24', 'K29',
                 'S3', 'S5', 'S6', 'S25', 'S26', 'B3', 'B4', 'B8'}
    }
    
    def __init__(self, module_code: str = "MLCC"):
        self.module_code = module_code
        self.valid_ksbs = self.VALID_KSBS.get(module_code, set())
    
    def validate_evaluation(
        self,
        evaluation_text: str,
        evidence_text: str,
        ksb_code: str
    ) -> ValidationResult:
        """
        Validate a single KSB evaluation for potential hallucinations.
        
        Args:
            evaluation_text: The LLM's evaluation output
            evidence_text: The evidence that was provided to the LLM
            ksb_code: The KSB being evaluated
            
        Returns:
            ValidationResult with findings
        """
        warnings = []
        errors = []
        confidence_score = 1.0
        
        # Check 1: Validate KSB code
        if ksb_code not in self.valid_ksbs:
            errors.append(f"Invalid KSB code '{ksb_code}' for module {self.module_code}")
            confidence_score -= 0.5
        
        # Check 2: Scan for hallucination patterns
        eval_lower = evaluation_text.lower()
        for pattern, message in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, eval_lower, re.IGNORECASE):
                warnings.append(f"HALLUCINATION WARNING: {message}")
                confidence_score -= 0.15
        
        # Check 3: Check for non-ASCII characters (model instability)
        for pattern, message in self.NON_ASCII_SCRIPTS:
            matches = re.findall(pattern, evaluation_text)
            if matches:
                errors.append(f"MODEL INSTABILITY: {message} - '{matches[:3]}'")
                confidence_score -= 0.3
        
        # Check 4: Verify evidence references exist
        evidence_refs = re.findall(r'\[E(\d+)\]', evaluation_text)
        if evidence_refs:
            max_ref = max(int(ref) for ref in evidence_refs)
            # Rough count of evidence chunks
            evidence_count = evidence_text.count('Evidence ') + evidence_text.count('[E')
            if evidence_count == 0:
                evidence_count = 5  # Default assumption
            
            if max_ref > evidence_count + 2:
                warnings.append(
                    f"Evidence reference [E{max_ref}] exceeds available evidence (~{evidence_count} chunks)"
                )
                confidence_score -= 0.2
        
        # Check 5: Look for quotes that don't appear in evidence (improved to reduce false positives)
        quoted_text = re.findall(r'"([^"]{20,})"', evaluation_text)

        # Phrases that indicate this is the LLM's own assessment, not a quote from the report
        assessor_phrases = [
            'the student', 'the report', 'the submission', 'the candidate',
            'the work', 'the evidence', 'the author', 'this demonstrates',
            'this shows', 'this indicates', 'based on', 'according to',
            'the brief', 'the assignment', 'the task', 'the ksb',
            'demonstrates', 'provides', 'shows', 'indicates', 'addresses'
        ]

        # Markdown table fragments and brief references (not actual quotes from report)
        non_quote_patterns = [
            r'^\s*\|',  # Starts with pipe (markdown table)
            r'^\s*[✅❌]',  # Starts with checkmark/cross
            r'task\s+\d+:',  # "Task 1:", "Task 2:", etc.
            r'requirement\s+\d+',  # "Requirement 1", etc.
            r'^\s*\[e\d+\]',  # Starts with evidence reference like [E1]
            r'met\s*\|',  # Contains "MET |" (table cell)
        ]

        for quote in quoted_text[:10]:  # Check first 10 quotes
            quote_lower = quote.lower()

            # Skip if this is clearly the LLM's own assessment statement
            is_assessor_statement = any(phrase in quote_lower[:50] for phrase in assessor_phrases)
            if is_assessor_statement:
                continue

            # Skip markdown table fragments and brief references
            is_non_quote = any(re.search(pattern, quote_lower, re.IGNORECASE) for pattern in non_quote_patterns)
            if is_non_quote:
                continue

            # Skip very short quotes (likely not actual report quotes)
            if len(quote) < 30:
                continue

            # Normalize whitespace for comparison
            quote_normalized = ' '.join(quote.split()).lower()
            evidence_normalized = ' '.join(evidence_text.split()).lower()

            # Check if the quote appears directly in evidence (exact or near-exact match)
            # Use longer substring (10 words or 60% of quote, whichever is shorter)
            words = quote_normalized.split()
            check_length = min(10, max(4, int(len(words) * 0.6)))  # 60% of quote or at least 4 words

            if len(words) >= 4:
                # Check multiple substrings to handle partial matches
                found = False

                # Try first N words
                search_phrase = ' '.join(words[:check_length])
                if search_phrase in evidence_normalized:
                    found = True

                # Try middle N words if not found
                if not found and len(words) > check_length:
                    mid_start = max(0, (len(words) - check_length) // 2)
                    search_phrase = ' '.join(words[mid_start:mid_start + check_length])
                    if search_phrase in evidence_normalized:
                        found = True

                # Try last N words if still not found
                if not found and len(words) > check_length:
                    search_phrase = ' '.join(words[-check_length:])
                    if search_phrase in evidence_normalized:
                        found = True

                # Only flag if NO substantial substring found AND quote looks like it claims to be from report
                if not found:
                    warnings.append(f"Quote may be fabricated: '{quote[:50]}...'")
                    confidence_score -= 0.05  # Reduced penalty from 0.1 to 0.05
        
        # Check 6: Detect if model is asking questions (breaking character)
        if re.search(r'\?$', evaluation_text.strip()[-100:]):
            if 'do you want' in eval_lower or 'would you like' in eval_lower:
                errors.append("Model appears to be asking questions - breaking assessor role")
                confidence_score -= 0.3
        
        # Determine suggested action
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        if confidence_score >= 0.7 and not errors:
            suggested_action = 'accept'
        elif confidence_score >= 0.4 and not errors:
            suggested_action = 'flag_for_review'
        else:
            suggested_action = 'reject'
        
        return ValidationResult(
            is_valid=len(errors) == 0 and confidence_score >= 0.5,
            confidence_score=confidence_score,
            warnings=warnings,
            errors=errors,
            suggested_action=suggested_action
        )
    
    def validate_overall_summary(
        self,
        summary_text: str,
        evaluations: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate the overall summary for consistency with individual evaluations.
        
        Args:
            summary_text: The LLM's overall summary
            evaluations: List of individual KSB evaluation results
            
        Returns:
            ValidationResult with findings
        """
        warnings = []
        errors = []
        confidence_score = 1.0
        
        # Check 1: Basic hallucination patterns
        summary_lower = summary_text.lower()
        for pattern, message in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, summary_lower, re.IGNORECASE):
                warnings.append(f"HALLUCINATION WARNING: {message}")
                confidence_score -= 0.15
        
        # Check 2: Verify KSB codes mentioned exist
        mentioned_ksbs = set(re.findall(r'\b([KSB]\d{1,2})\b', summary_text))
        evaluated_ksbs = {e.get('ksb_code', '') for e in evaluations}
        
        invalid_ksbs = mentioned_ksbs - evaluated_ksbs - self.valid_ksbs
        if invalid_ksbs:
            errors.append(f"Summary references invalid KSBs: {invalid_ksbs}")
            confidence_score -= 0.3
        
        # Check 3: Verify grade counts match
        # Extract grade counts from summary JSON if present
        json_match = re.search(r'"merit_count":\s*(\d+)', summary_text)
        if json_match:
            claimed_merit = int(json_match.group(1))
            actual_merit = sum(1 for e in evaluations if e.get('grade') == 'MERIT')
            if claimed_merit != actual_merit:
                warnings.append(
                    f"Merit count mismatch: claimed {claimed_merit}, actual {actual_merit}"
                )
                confidence_score -= 0.1
        
        # Check 4: Non-ASCII check
        for pattern, message in self.NON_ASCII_SCRIPTS:
            if re.search(pattern, summary_text):
                errors.append(f"MODEL INSTABILITY in summary: {message}")
                confidence_score -= 0.3
        
        # Determine action
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        if confidence_score >= 0.7 and not errors:
            suggested_action = 'accept'
        elif confidence_score >= 0.4:
            suggested_action = 'flag_for_review'
        else:
            suggested_action = 'reject'
        
        return ValidationResult(
            is_valid=len(errors) == 0 and confidence_score >= 0.5,
            confidence_score=confidence_score,
            warnings=warnings,
            errors=errors,
            suggested_action=suggested_action
        )
    
    def generate_validation_report(
        self,
        evaluations: List[Dict[str, Any]],
        summary_validation: Optional[ValidationResult] = None
    ) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            evaluations: List of evaluation dicts, each with 'validation' key
            summary_validation: Optional validation result for overall summary
            
        Returns:
            Markdown-formatted report
        """
        lines = [
            "# Validation Report",
            "",
            "## Summary",
            ""
        ]
        
        # Count issues
        total_warnings = 0
        total_errors = 0
        flagged_ksbs = []
        rejected_ksbs = []
        
        for eval_data in evaluations:
            validation = eval_data.get('validation')
            if validation:
                total_warnings += len(validation.warnings)
                total_errors += len(validation.errors)
                
                if validation.suggested_action == 'flag_for_review':
                    flagged_ksbs.append(eval_data.get('ksb_code', 'Unknown'))
                elif validation.suggested_action == 'reject':
                    rejected_ksbs.append(eval_data.get('ksb_code', 'Unknown'))
        
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total KSBs | {len(evaluations)} |",
            f"| Total Warnings | {total_warnings} |",
            f"| Total Errors | {total_errors} |",
            f"| Flagged for Review | {len(flagged_ksbs)} |",
            f"| Rejected | {len(rejected_ksbs)} |",
            ""
        ])
        
        if rejected_ksbs:
            lines.extend([
                "## ⚠️ REJECTED EVALUATIONS (Likely Hallucinations)",
                "",
                "The following KSB evaluations contain likely hallucinations and should be re-run:",
                ""
            ])
            for ksb in rejected_ksbs:
                lines.append(f"- **{ksb}**")
            lines.append("")
        
        if flagged_ksbs:
            lines.extend([
                "## 🔍 Flagged for Review",
                "",
                "The following KSB evaluations have potential issues and should be manually reviewed:",
                ""
            ])
            for ksb in flagged_ksbs:
                lines.append(f"- {ksb}")
            lines.append("")
        
        # Detailed findings
        if total_warnings > 0 or total_errors > 0:
            lines.extend([
                "## Detailed Findings",
                ""
            ])
            
            for eval_data in evaluations:
                validation = eval_data.get('validation')
                if validation and (validation.warnings or validation.errors):
                    ksb_code = eval_data.get('ksb_code', 'Unknown')
                    lines.append(f"### {ksb_code}")
                    lines.append(f"Confidence: {validation.confidence_score:.2f}")
                    lines.append("")
                    
                    if validation.errors:
                        lines.append("**Errors:**")
                        for error in validation.errors:
                            lines.append(f"- ❌ {error}")
                        lines.append("")
                    
                    if validation.warnings:
                        lines.append("**Warnings:**")
                        for warning in validation.warnings:
                            lines.append(f"- ⚠️ {warning}")
                        lines.append("")
        
        return "\n".join(lines)


def validate_before_display(
    evaluation: Dict[str, Any],
    evidence_text: str,
    module_code: str = "MLCC"
) -> Dict[str, Any]:
    """
    Convenience function to validate an evaluation before displaying to user.
    
    Args:
        evaluation: The evaluation dict from generate_feedback
        evidence_text: The evidence provided to the LLM
        module_code: The module being assessed
        
    Returns:
        The evaluation dict with 'validation' key added
    """
    validator = OutputValidator(module_code)
    
    validation = validator.validate_evaluation(
        evaluation_text=evaluation.get('evaluation', ''),
        evidence_text=evidence_text,
        ksb_code=evaluation.get('ksb_code', '')
    )
    
    evaluation['validation'] = {
        'is_valid': validation.is_valid,
        'confidence_score': validation.confidence_score,
        'warnings': validation.warnings,
        'errors': validation.errors,
        'suggested_action': validation.suggested_action
    }
    
    # Log issues
    if validation.errors:
        logger.error(f"Validation errors for {evaluation.get('ksb_code')}: {validation.errors}")
    if validation.warnings:
        logger.warning(f"Validation warnings for {evaluation.get('ksb_code')}: {validation.warnings}")
    
    return evaluation
