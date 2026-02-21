"""
KSB Evaluation Prompts V2 - With Assignment Brief Context Integration.

This version includes the assignment brief tasks in the evaluation context,
giving the LLM clear understanding of what the student was asked to do.
"""
import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class KSBPromptTemplates:
    """Prompt templates for KSB-based coursework evaluation with brief context."""
    
    SYSTEM_PROMPT_KSB_MARKER = """You are an academic assessor evaluating apprenticeship coursework against KSB criteria.

╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  CRITICAL RULES - MUST FOLLOW  ⚠️                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ONLY USE PROVIDED EVIDENCE
   - You may ONLY cite text that appears in the "EVIDENCE FROM SUBMISSION" section
   - You have the ASSIGNMENT BRIEF showing what the student was asked to do
   - If evidence is missing → write "NOT FOUND" 
   - NEVER invent quotes, section numbers, page numbers, or content
   
2. ZERO FABRICATION POLICY
   ❌ NEVER invent company names, product names, or technical details
   ❌ NEVER create fictional scenarios or examples not in the evidence
   ❌ NEVER add information from your training data
   ❌ NEVER assume what the student "might have meant"
   
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

╔══════════════════════════════════════════════════════════════════════════════╗
║                           GRADING SCALE                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

- REFERRAL: Pass criteria NOT met (significant gaps, missing evidence)
- PASS: Pass criteria met (basic requirements satisfied)  
- MERIT: Pass criteria met AND Merit criteria substantially met

╔══════════════════════════════════════════════════════════════════════════════╗
║                        IMAGE/FIGURE ANALYSIS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

When images are provided:
- Examine charts and graphs for data presentation quality
- Check architecture diagrams for system design understanding
- Reference specific figures as evidence (e.g., "Figure 1 shows...")

OUTPUT FORMAT: You MUST include a JSON block with your grade decision."""

    # New template with brief context
    KSB_EVALUATION_WITH_BRIEF_PROMPT = """Evaluate this student's work against the KSB criterion below.

╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  ANTI-HALLUCINATION REMINDER  ⚠️                        ║
║                                                                               ║
║  • ONLY use text from the EVIDENCE section below                             ║
║  • The ASSIGNMENT BRIEF shows what was required                              ║
║  • If you cannot find evidence → write "NOT FOUND"                           ║
║  • NEVER invent quotes, company names, or technical details                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

## ASSIGNMENT BRIEF CONTEXT

**What the student was asked to do for this KSB:**

{brief_context}

---

## KSB CRITERION

**{ksb_code}: {ksb_title}**

| Grade | Criteria |
|-------|----------|
| PASS | {pass_criteria} |
| MERIT | {merit_criteria} |
| REFERRAL | {referral_criteria} |

---

## EVIDENCE FROM SUBMISSION (USE ONLY THIS - DO NOT INVENT)

{evidence_text}
{image_context}

════════════════════════════════════════════════════════════════════════════════

## YOUR TASK

Follow these steps IN ORDER:

### STEP 1: BRIEF REQUIREMENTS CHECK

**⚠️ IMPORTANT:** Only assess this KSB against the tasks shown in the Assignment Brief Context above.
Do NOT penalize the student for not covering tasks that are not listed for this KSB.

Based on the Assignment Brief above, list what the student should have demonstrated FOR THIS SPECIFIC KSB:
- [ ] Requirement 1 from the tasks shown above
- [ ] Requirement 2 from the tasks shown above
- [ ] etc.

If no specific tasks are listed, assess based on overall report quality.

### STEP 2: LIST EVIDENCE FOUND

Search the evidence above and list ALL relevant content:

**Evidence Found:**
- [E1] "exact quote from evidence" (Section X) - addresses [requirement]
- [E2] "exact quote from evidence" (Section Y) - addresses [requirement]
- [E3] Figure/Chart: [describe what it shows] - addresses [requirement]

⚠️ RULES:
- Only quote TEXT that appears VERBATIM in the evidence
- Link each evidence item to a brief requirement or KSB criterion
- If no relevant evidence exists, write: **Evidence Found:** NONE
- When listing evidence, note if any values are placeholders (TBD, TODO, "fill with measured results"). Placeholder values are NOT evidence of execution.

### STEP 3: ASSESS PASS CRITERIA

| Pass Requirement | Brief Requirement | Status | Evidence |
|------------------|-------------------|--------|----------|
| [from KSB rubric] | [from assignment brief] | ✅ MET / ❌ NOT MET | [E1] or "NOT FOUND" |

### STEP 4: ASSESS MERIT CRITERIA (only if Pass is met)

**⚠️ MANDATORY:** You MUST evaluate EACH merit criterion individually. Do NOT skip this step.

For EACH merit criterion listed in the Merit row of the rubric above:
1. **Search the evidence** for ANY content that addresses this specific merit requirement
2. **If evidence exists:** Quote it with [E] reference and mark ✅ MET
3. **If NO evidence exists:** Mark ❌ NOT MET with explanation
4. **Do NOT write "NONE" or skip criteria** - assess each one individually

| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| [Copy exact text from merit_criteria above] | ✅ MET / ❌ NOT MET | [Ex] or "NOT FOUND" |
| [Next merit criterion if multiple] | ✅ MET / ❌ NOT MET | [Ex] or "NOT FOUND" |

**Note:** If the merit criteria contains multiple requirements (e.g., "demonstrates X AND provides Y"),
assess each component separately and mark MET only if ALL components have evidence.

**Special guidance for statistical/hypothesis testing KSBs (K22, K26, S22):**
If the merit criteria mentions "strong statistical reasoning", check for:
- Effect size measures (Cohen's d, odds ratio, R², etc.)
- Confidence intervals (95% CI, credible intervals)
- Assumption checking (normality, homogeneity, independence)
- Uncertainty quantification (p-values WITH interpretation, confidence levels)
- Practical significance vs statistical significance
- Business/organisational decision linked to statistical results

If 4+ of these elements are present with evidence, mark merit criteria as ✅ MET.

### STEP 5: GRADE DECISION

**⚠️ IMPORTANT:** Set "grade" to EXACTLY ONE of: PASS, MERIT, or REFERRAL (not all three).
Set "confidence" to EXACTLY ONE of: HIGH, MEDIUM, or LOW.

```json
{{
  "ksb_code": "{ksb_code}",
  "grade": "<CHOOSE ONE: PASS or MERIT or REFERRAL>",
  "confidence": "<CHOOSE ONE: HIGH or MEDIUM or LOW>",
  "pass_criteria_met": true or false,
  "merit_criteria_met": true or false,
  "brief_requirements_met": ["list of requirements met"],
  "brief_requirements_missing": ["list of requirements missing"],
  "key_evidence": ["E1", "E2"],
  "main_gap": "description or null"
}}
```

**Example valid JSON:**
```json
{{
  "ksb_code": "K22",
  "grade": "MERIT",
  "confidence": "HIGH",
  "pass_criteria_met": true,
  "merit_criteria_met": true,
  "brief_requirements_met": ["Statistical test selection", "Result interpretation"],
  "brief_requirements_missing": [],
  "key_evidence": ["E1", "E3", "E5"],
  "main_gap": null
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

════════════════════════════════════════════════════════════════════════════════

⚠️ FINAL REMINDER:
- Compare against BOTH the assignment brief AND the KSB criteria
- Grading guidance:
  * CLEAR evidence meeting criteria → PASS or MERIT
  * PARTIAL/IMPLICIT evidence showing some understanding → PASS (confidence LOW/MEDIUM)
  * NO relevant evidence OR fundamentally wrong → REFERRAL
- NEVER invent content - this is academic assessment requiring accuracy"""

    # Fallback template without brief (for backward compatibility)
    KSB_EVALUATION_PROMPT = """Evaluate this student's work against the KSB criterion below.

╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  ANTI-HALLUCINATION REMINDER  ⚠️                        ║
║                                                                               ║
║  • ONLY use text from the EVIDENCE section below                             ║
║  • If you cannot find evidence → write "NOT FOUND"                           ║
║  • NEVER invent quotes, company names, or technical details                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

## KSB CRITERION

**{ksb_code}: {ksb_title}**

| Grade | Criteria |
|-------|----------|
| PASS | {pass_criteria} |
| MERIT | {merit_criteria} |
| REFERRAL | {referral_criteria} |

## EVIDENCE FROM SUBMISSION (USE ONLY THIS - DO NOT INVENT)

{evidence_text}
{image_context}

---

## YOUR TASK

### STEP 1: LIST EVIDENCE

**Evidence Found:**
- [E1] "exact quote from evidence" (Section X) 
- [E2] "exact quote from evidence" (Section Y)

If no relevant evidence exists, write: **Evidence Found:** NONE

### STEP 2: ASSESS PASS CRITERIA

| Pass Requirement | Status | Evidence |
|------------------|--------|----------|
| [requirement] | ✅ MET / ❌ NOT MET | [E1] or "NOT FOUND" |

### STEP 3: ASSESS MERIT CRITERIA (only if Pass is met)

| Merit Requirement | Status | Evidence |
|-------------------|--------|----------|
| [requirement] | ✅ MET / ❌ NOT MET | [Ex] or "NOT FOUND" |

### STEP 4: GRADE DECISION

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

### STEP 5: JUSTIFICATION

2-3 sentences explaining your grade decision.

### STEP 6: IMPROVEMENTS

List 2-3 specific actions to improve."""

    KSB_OVERALL_SUMMARY_PROMPT = """Summarize the KSB evaluations below into an overall assessment.

╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  ANTI-HALLUCINATION REMINDER  ⚠️                        ║
║                                                                               ║
║  • ONLY summarize the evaluations provided below                             ║
║  • Do NOT add information from outside these evaluations                     ║
║  • Do NOT invent company names, statistics, or details                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

## MODULE: {module_code}

## ASSIGNMENT BRIEF SUMMARY

{brief_summary}

---

## KSB EVALUATIONS (SUMMARIZE ONLY THESE)

{evaluations_text}

════════════════════════════════════════════════════════════════════════════════

## YOUR TASK

### 1. GRADE SUMMARY TABLE

| KSB | Grade | Confidence | Brief Task | Key Finding |
|-----|-------|------------|------------|-------------|
| K1  | PASS/MERIT/REFERRAL | HIGH/MED/LOW | Task X | brief note |
| ... | ... | ... | ... | ... |

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

### 3. TASK COVERAGE

For each task in the brief, assess coverage:

| Task | KSBs | Completion | Notes |
|------|------|------------|-------|
| Task 1 | K1, K2, S16 | ✅ Complete / ⚠️ Partial / ❌ Missing | ... |

### 4. KEY STRENGTHS (Top 3)

1. **[Strength]** - demonstrated in [KSB codes] / [Task X]
2. ...
3. ...

### 5. PRIORITY IMPROVEMENTS (Top 3)

1. **[Gap]** - affects [KSB codes] - from [Task X] requirement
2. ...
3. ...

### 6. OVERALL ASSESSMENT

Write 2-3 paragraphs covering:
- How well the student addressed the assignment brief
- Overall quality relative to KSB requirements
- Specific recommendations for improvement

⚠️ ONLY use information from the evaluations above."""

    @classmethod
    def format_ksb_evaluation(
        cls,
        ksb_code: str,
        ksb_title: str,
        pass_criteria: str,
        merit_criteria: str,
        referral_criteria: str,
        evidence_text: str,
        image_context: str = "",
        brief_context: str = ""
    ) -> str:
        """Format the KSB evaluation prompt with optional brief context."""
        
        if brief_context:
            # Use the enhanced template with brief context
            return cls.KSB_EVALUATION_WITH_BRIEF_PROMPT.format(
                ksb_code=ksb_code,
                ksb_title=ksb_title,
                pass_criteria=pass_criteria,
                merit_criteria=merit_criteria,
                referral_criteria=referral_criteria,
                evidence_text=evidence_text,
                image_context=image_context,
                brief_context=brief_context
            )
        else:
            # Fallback to basic template
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
    def format_overall_summary(
        cls, 
        evaluations_text: str,
        module_code: str = "",
        brief_summary: str = ""
    ) -> str:
        """Format the overall summary prompt with brief context."""
        return cls.KSB_OVERALL_SUMMARY_PROMPT.format(
            evaluations_text=evaluations_text,
            module_code=module_code or "Unknown",
            brief_summary=brief_summary or "No assignment brief provided."
        )
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for KSB evaluation."""
        return cls.SYSTEM_PROMPT_KSB_MARKER


def _repair_json(json_str: str) -> str:
    """Attempt to fix common LLM JSON errors before falling back to regex."""
    # Remove parenthetical annotations after quoted strings: "text" (partial) -> "text"
    repaired = re.sub(r'"\s*\([^)]*\)', '"', json_str)
    # Remove trailing commas before closing brackets/braces
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    # Fix single quotes to double quotes (only for obvious cases)
    if "'" in repaired and '"' not in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def extract_grade_from_evaluation(evaluation: str) -> Dict[str, Any]:
    """
    Extract grade and metadata from LLM evaluation output.
    """
    result = {
        'grade': 'UNKNOWN',
        'confidence': 'LOW',
        'method': 'none',
        'pass_criteria_met': None,
        'merit_criteria_met': None,
        'brief_requirements_met': [],
        'brief_requirements_missing': [],
        'raw_json': None,
        'possible_hallucination': False
    }
    
    # Check for hallucination indicators
    hallucination_indicators = [
        'celestial', 'rovio', 'angry birds', 'founded in', 'headquarters:',
        'here are some key facts', 'do you want me to provide more',
        'website:', 'best known for:'
    ]
    
    eval_lower = evaluation.lower()
    for indicator in hallucination_indicators:
        if indicator in eval_lower:
            result['possible_hallucination'] = True
            break
    
    # Method 1: Extract JSON block
    json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', evaluation, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        parsed = None
        method_name = 'json'
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try JSON repair before giving up
            try:
                repaired = _repair_json(json_str)
                parsed = json.loads(repaired)
                method_name = 'json_repaired'
                logger.info(f"JSON repair fixed extraction (original error: {e})")
            except json.JSONDecodeError:
                pass
        if parsed and 'grade' in parsed:
            grade = parsed['grade'].upper().strip()

            # Safety net: Detect if LLM copied placeholder text literally
            if '|' in grade or '<CHOOSE' in grade.upper() or 'OR' in grade:
                # Derive grade from criteria_met fields instead
                if parsed.get('merit_criteria_met') is True:
                    grade = 'MERIT'
                elif parsed.get('pass_criteria_met') is True:
                    grade = 'PASS'
                else:
                    grade = 'REFERRAL'
                logger.warning(f"LLM output placeholder text in grade field, derived: {grade}")

            if grade in ['PASS', 'MERIT', 'REFERRAL']:
                result['grade'] = grade
                result['confidence'] = parsed.get('confidence', 'MEDIUM').upper().split('|')[0].strip()
                result['method'] = method_name
                result['pass_criteria_met'] = parsed.get('pass_criteria_met')
                result['merit_criteria_met'] = parsed.get('merit_criteria_met')
                result['brief_requirements_met'] = parsed.get('brief_requirements_met', [])
                result['brief_requirements_missing'] = parsed.get('brief_requirements_missing', [])
                result['raw_json'] = parsed
                return result

    # Method 2: Inline JSON
    inline_json = re.search(r'\{\s*"ksb_code"[^}]+\}', evaluation, re.DOTALL)
    if inline_json:
        json_str = inline_json.group(0)
        parsed = None
        method_name = 'inline_json'
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            try:
                repaired = _repair_json(json_str)
                parsed = json.loads(repaired)
                method_name = 'inline_json_repaired'
                logger.info(f"JSON repair fixed inline extraction (original error: {e})")
            except json.JSONDecodeError:
                pass
        if parsed and 'grade' in parsed:
            grade = parsed['grade'].upper().strip()
            if grade in ['PASS', 'MERIT', 'REFERRAL']:
                result['grade'] = grade
                result['confidence'] = parsed.get('confidence', 'MEDIUM').upper()
                result['method'] = method_name
                result['raw_json'] = parsed
                return result
    
    # Method 3: Regex patterns
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
            # Safety net: if regex says REFERRAL but raw text shows pass evidence, override
            if regex_grade == 'REFERRAL':
                eval_lower = evaluation.lower()
                has_pass_signal = (
                    '"pass_criteria_met": true' in eval_lower or
                    '"pass_criteria_met":true' in eval_lower or
                    re.search(r'"grade"\s*:\s*"PASS"', evaluation, re.IGNORECASE)
                )
                if has_pass_signal:
                    logger.warning(
                        f"Regex extracted REFERRAL but raw text contains PASS signals — overriding to PASS"
                    )
                    regex_grade = 'PASS'
            result['grade'] = regex_grade
            result['confidence'] = 'MEDIUM'
            result['method'] = 'regex'
            return result
    
    # Method 4: Heuristics
    eval_upper = evaluation.upper()
    
    if eval_upper.count('❌') > eval_upper.count('✅') or 'NOT FOUND' in eval_upper:
        result['grade'] = 'REFERRAL'
    elif 'MERIT' in eval_upper and 'ACHIEVED' in eval_upper:
        result['grade'] = 'MERIT'
    else:
        result['grade'] = 'PASS'
    
    result['confidence'] = 'LOW'
    result['method'] = 'heuristic'
    
    return result
