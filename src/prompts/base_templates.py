"""
Base prompt templates — shared utilities used by all module specialists.

Each module specialist has its OWN prompt file (mlcc_templates.py, etc.).
This file provides only shared infrastructure: system prompt, grade extraction,
JSON repair, and the 7-step evaluation scaffold.
"""
import re
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt — injected as Ollama system parameter for every evaluation call
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an academic assessor evaluating apprenticeship coursework against KSB criteria.

CRITICAL RULES — MUST FOLLOW:

1. ONLY USE PROVIDED EVIDENCE
   - You may ONLY cite text from the EVIDENCE FROM SUBMISSION section
   - The ASSIGNMENT BRIEF shows what the student was asked to do
   - If evidence is missing → write "NOT FOUND"
   - NEVER invent quotes, section numbers, page numbers, or content

2. ZERO FABRICATION POLICY
   - NEVER invent company names, product names, or technical details
   - NEVER create fictional scenarios or examples not in the evidence
   - NEVER add information from your training data
   - NEVER assume what the student "might have meant"

3. BRIEF-AWARE EVALUATION
   - The ASSIGNMENT BRIEF tells you WHAT the student should have done
   - The KSB CRITERIA tells you HOW to evaluate it
   - The EVIDENCE shows you WHAT the student actually did
   - Compare EVIDENCE against both BRIEF requirements and KSB criteria

4. EVIDENCE-FIRST ASSESSMENT
   - List ALL evidence BEFORE making any judgement
   - Every claim must have a citation [E1], [E2], etc.
   - If no evidence exists → grade as REFERRAL with confidence LOW

5. CITE BY SECTION (not page)
   - Use section numbers as primary reference (e.g., "Section 3")
   - Only cite sections shown in the evidence headers

GRADING SCALE:
- REFERRAL: Pass criteria NOT met (significant gaps, missing evidence)
- PASS: Pass criteria met (basic requirements satisfied)
- MERIT: Pass criteria met AND Merit criteria substantially met

OUTPUT FORMAT: You MUST include a JSON block with your grade decision."""


# ═══════════════════════════════════════════════════════════════════════════════
# 7-step evaluation scaffold — used by all module-specific templates
# ═══════════════════════════════════════════════════════════════════════════════

EVALUATION_STEPS = """
## YOUR TASK

Follow these steps IN ORDER:

### STEP 1: BRIEF REQUIREMENTS CHECK

Based on the Assignment Brief above, list what the student should have demonstrated FOR THIS SPECIFIC KSB:
- [ ] Requirement 1 from the tasks shown above
- [ ] Requirement 2 from the tasks shown above

If no specific tasks are listed, assess based on overall report quality.

### STEP 2: LIST EVIDENCE FOUND

Search the evidence above and list ALL relevant content:

**Evidence Found:**
- [E1] "exact quote from evidence" (Section X) - addresses [requirement]
- [E2] "exact quote from evidence" (Section Y) - addresses [requirement]

RULES:
- Only quote TEXT that appears VERBATIM in the evidence
- Link each evidence item to a brief requirement or KSB criterion
- If no relevant evidence exists, write: **Evidence Found:** NONE
- Note if any values are placeholders (TBD, TODO, "fill with measured results")

### STEP 3: ASSESS PASS CRITERIA

| Pass Requirement | Brief Requirement | Status | Evidence |
|------------------|-------------------|--------|----------|
| [from KSB rubric] | [from assignment brief] | MET / NOT MET | [E1] or "NOT FOUND" |

### STEP 4: ASSESS MERIT CRITERIA (only if Pass is met)

You MUST evaluate EACH merit criterion individually. Do NOT skip this step.

For EACH merit criterion listed in the Merit row:
1. Search the evidence for ANY content that addresses this specific merit requirement
2. If evidence exists: Quote it with [E] reference and mark MET
3. If NO evidence exists: Mark NOT MET with explanation

| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| [Copy exact text from merit_criteria] | MET / NOT MET | [Ex] or "NOT FOUND" |

{module_specific_merit_guidance}

### STEP 5: GRADE DECISION

Set "grade" to EXACTLY ONE of: PASS, MERIT, or REFERRAL.
Set "confidence" to EXACTLY ONE of: HIGH, MEDIUM, or LOW.

```json
{{
  "ksb_code": "{ksb_code}",
  "grade": "<CHOOSE ONE: PASS or MERIT or REFERRAL>",
  "confidence": "<CHOOSE ONE: HIGH or MEDIUM or LOW>",
  "pass_criteria_met": true or false,
  "merit_criteria_met": true or false,
  "evidence_strength": "<weak or moderate or strong>",
  "gaps": ["list of gaps found"],
  "brief_requirements_met": ["list of requirements met"],
  "brief_requirements_missing": ["list of requirements missing"],
  "key_evidence": ["E1", "E2"],
  "main_gap": "description or null"
}}
```

### STEP 6: BRIEF JUSTIFICATION

2-3 sentences explaining:
1. How the student addressed the assignment brief requirements
2. How this maps to the KSB criteria
3. What grade this warrants and why

### STEP 7: SPECIFIC IMPROVEMENTS

List 2-3 specific actions referencing the assignment brief:
- "To address Task X requirement Y, the student should..."
- "The brief asked for Z but the submission only shows..."

FINAL REMINDER:
- Compare against BOTH the assignment brief AND the KSB criteria
- Grading guidance:
  * CLEAR evidence meeting criteria → PASS or MERIT
  * PARTIAL/IMPLICIT evidence showing some understanding → PASS (confidence LOW/MEDIUM)
  * NO relevant evidence OR fundamentally wrong → REFERRAL
- NEVER invent content - this is academic assessment requiring accuracy"""


# ═══════════════════════════════════════════════════════════════════════════════
# Grade extraction — 4-method cascade (json_block → inline_json → regex → heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

def _repair_json(json_str: str) -> str:
    """Attempt to fix common LLM JSON errors."""
    repaired = re.sub(r'"\s*\([^)]*\)', '"', json_str)
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def extract_grade(evaluation: str) -> Dict[str, Any]:
    """
    Extract grade and metadata from LLM evaluation output.

    Uses a 4-method cascade:
      1. json_block  — parse ```json...``` markdown block
      2. inline_json — find raw {...} JSON object
      3. regex       — match "grade": "PASS" patterns
      4. heuristic   — count checkmarks vs crosses

    Safety net: if regex finds REFERRAL but raw text contains
    pass_criteria_met: true or "grade": "PASS", override to PASS.
    """
    result = {
        'grade': 'UNKNOWN',
        'confidence': 'LOW',
        'method': 'none',
        'pass_criteria_met': None,
        'merit_criteria_met': None,
        'evidence_strength': 'unknown',
        'gaps': [],
        'brief_requirements_met': [],
        'brief_requirements_missing': [],
        'raw_json': None,
        'possible_hallucination': False,
    }

    # Check for hallucination indicators
    eval_lower = evaluation.lower()
    hallucination_indicators = [
        'celestial', 'rovio', 'angry birds', 'founded in', 'headquarters:',
        'here are some key facts', 'do you want me to provide more',
        'website:', 'best known for:',
    ]
    for indicator in hallucination_indicators:
        if indicator in eval_lower:
            result['possible_hallucination'] = True
            break

    # ── Method 1: json_block ────────────────────────────────────────
    json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', evaluation, re.DOTALL)
    if json_match:
        parsed = _try_parse_json(json_match.group(1))
        if parsed and 'grade' in parsed:
            grade = _clean_grade(parsed['grade'], parsed)
            if grade:
                _populate_result(result, grade, parsed, 'json_block')
                return result

    # ── Method 2: inline_json ───────────────────────────────────────
    inline_match = re.search(r'\{\s*"ksb_code"[^}]+\}', evaluation, re.DOTALL)
    if inline_match:
        parsed = _try_parse_json(inline_match.group(0))
        if parsed and 'grade' in parsed:
            grade = _clean_grade(parsed['grade'], parsed)
            if grade:
                _populate_result(result, grade, parsed, 'inline_json')
                return result

    # ── Method 3: regex ─────────────────────────────────────────────
    patterns = [
        r'"grade"\s*:\s*"(PASS|MERIT|REFERRAL)"',
        r'\*\*Grade\*\*\s*:\s*\*?\*?(PASS|MERIT|REFERRAL)',
        r'Grade\s+Decision\s*:\s*\*?\*?(PASS|MERIT|REFERRAL)',
        r'\bgrade\b[:\s]+\*?\*?(PASS|MERIT|REFERRAL)\*?\*?',
    ]
    for pattern in patterns:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            regex_grade = match.group(1).upper()
            # Safety net
            if regex_grade == 'REFERRAL':
                if ('"pass_criteria_met": true' in eval_lower or
                        '"pass_criteria_met":true' in eval_lower or
                        re.search(r'"grade"\s*:\s*"PASS"', evaluation, re.IGNORECASE)):
                    logger.warning("Regex found REFERRAL but PASS signals present — overriding to PASS")
                    regex_grade = 'PASS'
            result['grade'] = regex_grade
            result['confidence'] = 'MEDIUM'
            result['method'] = 'regex'
            return result

    # ── Method 4: heuristic ─────────────────────────────────────────
    eval_upper = evaluation.upper()
    if eval_upper.count('NOT MET') > eval_upper.count('MET') - eval_upper.count('NOT MET'):
        result['grade'] = 'REFERRAL'
    elif 'MERIT' in eval_upper and ('ACHIEVED' in eval_upper or 'MET' in eval_upper):
        result['grade'] = 'MERIT'
    else:
        result['grade'] = 'PASS'
    result['confidence'] = 'LOW'
    result['method'] = 'heuristic'
    return result


def _try_parse_json(json_str: str) -> Dict[str, Any] | None:
    """Try to parse JSON, with repair fallback."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json(json_str))
        except json.JSONDecodeError:
            return None


def _clean_grade(raw_grade: str, parsed: Dict) -> str | None:
    """Clean and validate a grade string from parsed JSON."""
    grade = raw_grade.upper().strip()
    # Handle template placeholders
    if '|' in grade or '<CHOOSE' in grade or 'OR' in grade:
        if parsed.get('merit_criteria_met') is True:
            grade = 'MERIT'
        elif parsed.get('pass_criteria_met') is True:
            grade = 'PASS'
        else:
            grade = 'REFERRAL'
        logger.warning(f"LLM output placeholder in grade field, derived: {grade}")
    return grade if grade in ('PASS', 'MERIT', 'REFERRAL') else None


def _populate_result(result: Dict, grade: str, parsed: Dict, method: str):
    """Fill result dict from parsed JSON."""
    result['grade'] = grade
    result['confidence'] = parsed.get('confidence', 'MEDIUM').upper().split('|')[0].strip()
    result['method'] = method
    result['pass_criteria_met'] = parsed.get('pass_criteria_met')
    result['merit_criteria_met'] = parsed.get('merit_criteria_met')
    result['evidence_strength'] = parsed.get('evidence_strength', 'unknown')
    result['gaps'] = parsed.get('gaps', [])
    result['brief_requirements_met'] = parsed.get('brief_requirements_met', [])
    result['brief_requirements_missing'] = parsed.get('brief_requirements_missing', [])
    result['raw_json'] = parsed


# ═══════════════════════════════════════════════════════════════════════════════
# Placeholder detection — weighted by chunk relevance
# ═══════════════════════════════════════════════════════════════════════════════

PLACEHOLDER_TERMS = [
    'tbd', 'todo', '[placeholder]', 'fill with measured results',
    'insert results here', 'to be completed', 'add data here',
    '[your', 'xxx', 'lorem ipsum',
]


def detect_placeholders(chunks: List[Dict], threshold: float = 5.0) -> tuple[float, bool]:
    """
    Detect placeholder content weighted by chunk relevance score.

    Args:
        chunks: Retrieved chunks with 'content' and 'similarity' keys.
        threshold: Weighted count above which placeholder_detected is True.

    Returns:
        (weighted_count, detected) tuple.
    """
    weighted_count = 0.0

    for chunk in chunks:
        content_lower = chunk.get('content', '').lower()
        similarity = chunk.get('similarity', 0.0)

        # Weight by relevance
        if similarity >= 0.15:
            weight = 1.0
        elif similarity >= 0.08:
            weight = 0.5
        else:
            weight = 0.0

        for term in PLACEHOLDER_TERMS:
            occurrences = content_lower.count(term)
            weighted_count += occurrences * weight

    return weighted_count, weighted_count >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Boilerplate filtering — skip non-evidence chunks
# ═══════════════════════════════════════════════════════════════════════════════

def is_boilerplate(chunk: Dict) -> bool:
    """
    Check if a chunk is boilerplate that should be skipped during evidence retrieval.

    Filters:
    - Chunks < 100 characters
    - Title pages
    - KSB reflection table headers / mapping tables
    - Table of contents
    - Chunks with 3+ KSB codes + pipe characters (mapping grids)
    """
    content = chunk.get('content', '')
    if len(content) < 100:
        return True

    content_lower = content.lower()

    # Title page
    if any(kw in content_lower for kw in ['title page', 'cover page', 'submission date']):
        if len(content) < 300:
            return True

    # Table of contents
    if 'table of contents' in content_lower or 'contents page' in content_lower:
        return True

    # KSB mapping grids (3+ KSB codes + pipe chars)
    ksb_codes = re.findall(r'\b[KSB]\d{1,2}\b', content)
    if len(ksb_codes) >= 3 and content.count('|') >= 3:
        return True

    # Reflection table headers
    if ('ksb' in content_lower and 'reflection' in content_lower and
            content.count('|') >= 2):
        return True

    return False
