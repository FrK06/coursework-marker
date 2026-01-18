"""
KSB Evaluation Prompts - Improved version with structured output.

Key Improvements:
1. STRUCTURED OUTPUT - JSON blocks for reliable grade extraction
2. EVIDENCE-FIRST - Must cite before assessing
3. STRICT GROUNDING - Explicit rules to prevent hallucination
4. SHORTER PROMPTS - Focused instructions that won't be ignored
"""


class KSBPromptTemplates:
    """
    Prompt templates for KSB-based coursework evaluation.
    
    Design principles:
    - Structured output for reliable parsing
    - Evidence-first workflow to ground assessments
    - Explicit "not found" language to prevent invention
    - Concise instructions for better compliance
    """
    
    # ==========================================================================
    # SYSTEM PROMPT - Shorter, more focused
    # ==========================================================================
    
    SYSTEM_PROMPT_KSB_MARKER = """You are an academic assessor evaluating apprenticeship coursework against KSB criteria.

CORE RULES (MUST FOLLOW):

1. EVIDENCE-FIRST: List evidence BEFORE making any judgement
2. CITE EXACTLY: Only use page/section numbers shown in the evidence headers
3. NO INVENTION: If evidence doesn't exist, write "NOT FOUND" - never invent quotes
4. GRADE HONESTLY: Base grade ONLY on evidence present, not assumptions

GRADING SCALE:
- REFERRAL: Pass criteria NOT met (significant gaps)
- PASS: Pass criteria met (basic requirements satisfied)  
- MERIT: Pass criteria met AND Merit criteria substantially met

OUTPUT FORMAT: You MUST include a JSON block with your grade decision. This is mandatory."""

    # ==========================================================================
    # KSB EVALUATION PROMPT - Structured output
    # ==========================================================================
    
    KSB_EVALUATION_PROMPT = """Evaluate this student's work against the KSB criterion below.

## KSB CRITERION

**{ksb_code}: {ksb_title}**

| Grade | Criteria |
|-------|----------|
| PASS | {pass_criteria} |
| MERIT | {merit_criteria} |
| REFERRAL | {referral_criteria} |

## EVIDENCE FROM SUBMISSION

{evidence_text}

---

## YOUR TASK

Follow these steps IN ORDER:

### STEP 1: LIST EVIDENCE

Search the evidence above and list ALL relevant quotes. Use this EXACT format:

**Evidence Found:**
- [E1] "quote here" (page X, Section Y) 
- [E2] "quote here" (page X, Section Y)
- [E3] ...

If no relevant evidence exists, write: **Evidence Found:** NONE

RULES:
- Only use page/section numbers that appear in evidence headers above
- Do NOT invent section numbers like "2.1" or "3.2" if not shown
- Quote actual text, do not paraphrase and claim it's a quote

### STEP 2: ASSESS PASS CRITERIA

For each Pass requirement, state if MET or NOT MET with evidence reference:

| Pass Requirement | Status | Evidence |
|------------------|--------|----------|
| [requirement 1] | ✅ MET / ❌ NOT MET | [E1] or "not found" |
| [requirement 2] | ✅ MET / ❌ NOT MET | [E2] or "not found" |

### STEP 3: ASSESS MERIT CRITERIA (only if Pass is met)

| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| [requirement 1] | ✅ MET / ❌ NOT MET | [Ex] or "not found" |

### STEP 4: GRADE DECISION

You MUST output this JSON block (copy this structure exactly):

```json
{{
  "ksb_code": "{ksb_code}",
  "grade": "PASS|MERIT|REFERRAL",
  "confidence": "HIGH|MEDIUM|LOW",
  "pass_criteria_met": true|false,
  "merit_criteria_met": true|false,
  "key_evidence": ["E1", "E2"],
  "main_gap": "description or null"
}}
```

### STEP 5: BRIEF JUSTIFICATION

2-3 sentences explaining your grade decision, referencing your evidence labels [E1], [E2] etc.

### STEP 6: IMPROVEMENTS NEEDED

If REFERRAL or PASS: List 2-3 specific actions to reach the next grade level.
If MERIT: Note what made this strong.

---

REMINDER: 
- If evidence is missing, grade as REFERRAL with confidence LOW
- If Pass criteria are clearly met, do NOT give REFERRAL
- Use only the citation format from evidence headers"""

    # ==========================================================================
    # OVERALL SUMMARY PROMPT
    # ==========================================================================
    
    KSB_OVERALL_SUMMARY_PROMPT = """Summarize the KSB evaluations below into an overall assessment.

## KSB EVALUATIONS

{evaluations_text}

---

## YOUR TASK

### 1. GRADE SUMMARY TABLE

| KSB | Grade | Confidence | Key Strength or Gap |
|-----|-------|------------|---------------------|
| K1  | PASS/MERIT/REFERRAL | HIGH/MED/LOW | brief note |
| ... | ... | ... | ... |

### 2. OVERALL STATISTICS

```json
{{
  "total_ksbs": X,
  "merit_count": X,
  "pass_count": X,
  "referral_count": X,
  "overall_recommendation": "PASS|MERIT|REFERRAL",
  "confidence": "HIGH|MEDIUM|LOW"
}}
```

### 3. KEY STRENGTHS (Top 3)

1. **[Strength]** - demonstrated in [KSB codes] - [brief evidence]
2. ...
3. ...

### 4. PRIORITY IMPROVEMENTS (Top 3)

1. **[Gap]** - affects [KSB codes] - [specific action needed]
2. ...
3. ...

### 5. OVERALL ASSESSMENT

Write 2-3 paragraphs covering:
- Overall quality and effort
- Pattern of strengths across Knowledge/Skills/Behaviours
- Readiness assessment
- Key recommendations

Keep tone constructive and specific."""

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    @classmethod
    def format_ksb_evaluation(
        cls,
        ksb_code: str,
        ksb_title: str,
        pass_criteria: str,
        merit_criteria: str,
        referral_criteria: str,
        evidence_text: str
    ) -> str:
        """Format the KSB evaluation prompt."""
        return cls.KSB_EVALUATION_PROMPT.format(
            ksb_code=ksb_code,
            ksb_title=ksb_title,
            pass_criteria=pass_criteria,
            merit_criteria=merit_criteria,
            referral_criteria=referral_criteria,
            evidence_text=evidence_text
        )
    
    @classmethod
    def format_overall_summary(cls, evaluations_text: str) -> str:
        """Format the overall summary prompt."""
        return cls.KSB_OVERALL_SUMMARY_PROMPT.format(
            evaluations_text=evaluations_text
        )
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for KSB evaluation."""
        return cls.SYSTEM_PROMPT_KSB_MARKER


# =============================================================================
# GRADE EXTRACTION UTILITIES
# =============================================================================

import re
import json
from typing import Optional, Dict, Any


def extract_grade_from_evaluation(evaluation: str) -> Dict[str, Any]:
    """
    Extract grade and metadata from LLM evaluation output.
    
    Tries multiple extraction methods in order of reliability:
    1. JSON block parsing (most reliable)
    2. Regex pattern matching (fallback)
    3. Keyword heuristics (last resort)
    
    Returns:
        Dict with 'grade', 'confidence', 'method' keys
    """
    result = {
        'grade': 'UNKNOWN',
        'confidence': 'LOW',
        'method': 'none',
        'pass_criteria_met': None,
        'merit_criteria_met': None,
        'raw_json': None
    }
    
    # Method 1: Extract JSON block
    json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', evaluation, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if 'grade' in parsed:
                grade = parsed['grade'].upper().strip()
                if grade in ['PASS', 'MERIT', 'REFERRAL']:
                    result['grade'] = grade
                    result['confidence'] = parsed.get('confidence', 'MEDIUM').upper()
                    result['method'] = 'json'
                    result['pass_criteria_met'] = parsed.get('pass_criteria_met')
                    result['merit_criteria_met'] = parsed.get('merit_criteria_met')
                    result['raw_json'] = parsed
                    return result
        except json.JSONDecodeError:
            pass
    
    # Method 2: Look for inline JSON (without code block)
    inline_json = re.search(r'\{\s*"ksb_code"[^}]+\}', evaluation, re.DOTALL)
    if inline_json:
        try:
            parsed = json.loads(inline_json.group(0))
            if 'grade' in parsed:
                grade = parsed['grade'].upper().strip()
                if grade in ['PASS', 'MERIT', 'REFERRAL']:
                    result['grade'] = grade
                    result['confidence'] = parsed.get('confidence', 'MEDIUM').upper()
                    result['method'] = 'inline_json'
                    result['raw_json'] = parsed
                    return result
        except json.JSONDecodeError:
            pass
    
    # Method 3: Regex patterns for explicit grade statements
    patterns = [
        r'"grade"\s*:\s*"(PASS|MERIT|REFERRAL)"',
        r'\*\*Grade\*\*\s*:\s*\*?\*?(PASS|MERIT|REFERRAL)',
        r'Recommended\s+Grade\s*:\s*\*?\*?(PASS|MERIT|REFERRAL)',
        r'Grade\s+Decision\s*:\s*\*?\*?(PASS|MERIT|REFERRAL)',
        r'\bgrade\b[:\s]+\*?\*?(PASS|MERIT|REFERRAL)\*?\*?',
        r'\*\*(PASS|MERIT|REFERRAL)\*\*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            grade = match.group(1).upper()
            result['grade'] = grade
            result['confidence'] = 'MEDIUM'
            result['method'] = 'regex'
            return result
    
    # Method 4: Keyword heuristics (least reliable)
    eval_upper = evaluation.upper()
    
    # Count indicators
    referral_indicators = [
        'NOT MET' in eval_upper,
        'SIGNIFICANT GAP' in eval_upper,
        'MISSING' in eval_upper and 'EVIDENCE' in eval_upper,
        'REFERRAL' in eval_upper and 'RECOMMEND' in eval_upper,
        eval_upper.count('❌') > eval_upper.count('✅')
    ]
    
    merit_indicators = [
        'EXCEEDS' in eval_upper,
        'STRONG EVIDENCE' in eval_upper,
        'MERIT' in eval_upper and 'ACHIEVED' in eval_upper,
        'WELL DEMONSTRATED' in eval_upper
    ]
    
    pass_indicators = [
        'MEETS' in eval_upper and 'BASIC' in eval_upper,
        'PASS CRITERIA' in eval_upper and 'MET' in eval_upper,
        'ADEQUATE' in eval_upper
    ]
    
    referral_score = sum(referral_indicators)
    merit_score = sum(merit_indicators)
    pass_score = sum(pass_indicators)
    
    if referral_score >= 2:
        result['grade'] = 'REFERRAL'
    elif merit_score >= 2:
        result['grade'] = 'MERIT'
    elif pass_score >= 1 or '✅' in evaluation:
        result['grade'] = 'PASS'
    else:
        result['grade'] = 'PASS'  # Default to pass if unclear
    
    result['confidence'] = 'LOW'
    result['method'] = 'heuristic'
    
    return result


def validate_citations(evaluation: str, evidence_text: str) -> Dict[str, Any]:
    """
    Validate that citations in the evaluation match the provided evidence.
    
    Returns:
        Dict with validation results
    """
    # Extract page/section references from evidence headers
    evidence_refs = set()
    
    # Match patterns like "(page 1, Section 2)" or "(page 1 / Section 2)"
    ref_pattern = r'\(page\s+(\d+)[,/\s]+(?:Section\s+)?(\d+(?:\.\d+)?)\)'
    for match in re.finditer(ref_pattern, evidence_text, re.IGNORECASE):
        page, section = match.groups()
        evidence_refs.add((page, section))
    
    # Also match simpler patterns
    page_pattern = r'page\s+(\d+)'
    for match in re.finditer(page_pattern, evidence_text, re.IGNORECASE):
        evidence_refs.add((match.group(1), None))
    
    # Extract citations from evaluation
    eval_citations = []
    for match in re.finditer(ref_pattern, evaluation, re.IGNORECASE):
        page, section = match.groups()
        eval_citations.append({
            'page': page,
            'section': section,
            'valid': (page, section) in evidence_refs or (page, None) in evidence_refs
        })
    
    valid_count = sum(1 for c in eval_citations if c['valid'])
    invalid_citations = [c for c in eval_citations if not c['valid']]
    
    return {
        'total_citations': len(eval_citations),
        'valid_citations': valid_count,
        'invalid_citations': invalid_citations,
        'evidence_refs_found': list(evidence_refs),
        'citation_accuracy': valid_count / len(eval_citations) if eval_citations else 1.0
    }


# =============================================================================
# EXAMPLE OUTPUT (for reference/testing)
# =============================================================================

EXAMPLE_EVALUATION_OUTPUT = """
### STEP 1: LIST EVIDENCE

**Evidence Found:**
- [E1] "Used supervised learning (text classification) to meet a business objective: faster, more consistent support ticket triage." (page 1, Section 1)
- [E2] "The business goal is to improve triage speed, consistency, and reporting in a GDPR-compliant way while controlling cost." (page 1, Section 1)
- [E3] "Defined evaluation metrics aligned to the operational goal (e.g., per-class recall for urgent cases)." (page 7, Section 7)

### STEP 2: ASSESS PASS CRITERIA

| Pass Requirement | Status | Evidence |
|------------------|--------|----------|
| States clear business problem | ✅ MET | [E2] |
| Identifies appropriate ML approach | ✅ MET | [E1] |
| Describes why approach fits objective | ✅ MET | [E1], [E3] |

### STEP 3: ASSESS MERIT CRITERIA

| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| Strong problem framing | ✅ MET | [E2] clear business context |
| Justifies methodology choices | ⚠️ PARTIAL | [E1] explains text classification but no comparison to alternatives |
| Alternatives considered | ❌ NOT MET | not found |
| Reasoned selection tied to outcomes | ✅ MET | [E3] |

### STEP 4: GRADE DECISION

```json
{
  "ksb_code": "K1",
  "grade": "PASS",
  "confidence": "HIGH",
  "pass_criteria_met": true,
  "merit_criteria_met": false,
  "key_evidence": ["E1", "E2", "E3"],
  "main_gap": "No comparison to alternative approaches (baseline, traditional ML)"
}
```

### STEP 5: BRIEF JUSTIFICATION

The submission clearly meets all Pass criteria with strong evidence [E1], [E2], [E3] showing a well-defined business problem and appropriate ML approach. However, Merit criteria are only partially met because while the methodology is explained, there is no explicit comparison to alternatives or baseline approaches.

### STEP 6: IMPROVEMENTS NEEDED

To achieve MERIT:
1. Add a brief comparison of why neural network text classification was chosen over traditional ML (SVM, Naive Bayes) or rule-based approaches
2. Include a baseline comparison showing expected improvement over manual triage
3. Quantify the expected accuracy/speed improvement that justifies the ML approach
"""