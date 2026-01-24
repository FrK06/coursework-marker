"""
KSB Evaluation Prompts - Structured output for reliable grade extraction.
Includes vision support for analyzing charts, figures, and tables.
"""
import re
import json
from typing import Dict, Any


class KSBPromptTemplates:
    """Prompt templates for KSB-based coursework evaluation with vision support."""
    
    SYSTEM_PROMPT_KSB_MARKER = """You are an academic assessor evaluating apprenticeship coursework against KSB criteria.

CORE RULES (MUST FOLLOW):

1. EVIDENCE-FIRST: List evidence BEFORE making any judgement
2. CITE BY SECTION: Use section numbers as primary reference (e.g., "Section 3", "Section 5")
3. PAGE ESTIMATES: Page numbers with ~ are estimates - prefer section numbers when available
4. NO INVENTION: If evidence doesn't exist, write "NOT FOUND" - never invent quotes or section numbers
5. GRADE HONESTLY: Base grade ONLY on evidence present, not assumptions
6. ANALYZE IMAGES: When images/figures/charts are provided, examine them carefully for evidence

GRADING SCALE:
- REFERRAL: Pass criteria NOT met (significant gaps)
- PASS: Pass criteria met (basic requirements satisfied)  
- MERIT: Pass criteria met AND Merit criteria substantially met

IMAGE/FIGURE ANALYSIS:
When images are provided, you MUST:
- Examine charts and graphs for data presentation quality
- Check architecture diagrams for system design understanding
- Review tables for data analysis and organization
- Note figure quality, labeling, and relevance
- Reference specific figures as evidence (e.g., "Figure 1 shows...")

OUTPUT FORMAT: You MUST include a JSON block with your grade decision. This is mandatory."""

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
{image_context}
---

## YOUR TASK

Follow these steps IN ORDER:

### STEP 1: LIST EVIDENCE

Search the evidence above (including any images/figures provided) and list ALL relevant content. Use this EXACT format:

**Evidence Found:**
- [E1] "quote here" (Section X) 
- [E2] "quote here" (Section Y)
- [E3] Figure/Chart: [describe what the figure shows and why it's relevant]
- ...

If no relevant evidence exists, write: **Evidence Found:** NONE

RULES:
- Use SECTION numbers as your primary citation (e.g., "Section 3", "Section 5")
- Page numbers with ~ are estimates - prefer section numbers when available
- Only cite sections/pages that appear in evidence headers above
- Do NOT invent section numbers like "2.1" or "3.2" unless shown in headers
- Quote actual text, do not paraphrase and claim it's a quote
- For figures/images: describe what you observe and its relevance to the criterion

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
Include observations from figures/charts if they contributed to your assessment.

### STEP 6: IMPROVEMENTS NEEDED

If REFERRAL or PASS: List 2-3 specific actions to reach the next grade level.
If MERIT: Note what made this strong.

---

REMINDER: 
- If evidence is missing, grade as REFERRAL with confidence LOW
- If Pass criteria are clearly met, do NOT give REFERRAL
- Cite by SECTION number (e.g., "Section 3") - page numbers are estimates
- Analyze any provided figures/charts for relevant evidence"""

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

    @classmethod
    def format_ksb_evaluation(
        cls,
        ksb_code: str,
        ksb_title: str,
        pass_criteria: str,
        merit_criteria: str,
        referral_criteria: str,
        evidence_text: str,
        image_context: str = ""
    ) -> str:
        """Format the KSB evaluation prompt with optional image context."""
        return cls.KSB_EVALUATION_PROMPT.format(
            ksb_code=ksb_code,
            ksb_title=ksb_title,
            pass_criteria=pass_criteria,
            merit_criteria=merit_criteria,
            referral_criteria=referral_criteria,
            evidence_text=evidence_text,
            image_context=image_context
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
