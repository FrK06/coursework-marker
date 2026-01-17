"""
KSB Evaluation Prompts - Prompts for assessing work against KSB criteria.

Design Principles:
1. GROUNDING: Always cite evidence from provided context
2. GRADE ALIGNMENT: Evaluate against Pass/Merit/Referral descriptors
3. HONEST: Admit when evidence is missing or insufficient
4. CONSTRUCTIVE: Focus on how to improve
5. STRUCTURED: Consistent output format
"""


class KSBPromptTemplates:
    """
    Prompt templates for KSB-based coursework evaluation.
    
    Evaluates student work against:
    - Pass criteria (minimum acceptable)
    - Merit criteria (higher standard)
    - Referral criteria (not yet achieved)
    """
    
    # ==========================================================================
    # SYSTEM PROMPT FOR KSB EVALUATION
    # ==========================================================================
    
    SYSTEM_PROMPT_KSB_MARKER = """You are an experienced academic assessor evaluating apprenticeship coursework against Knowledge, Skills, and Behaviours (KSB) criteria.

Your role is to:
1. EVALUATE work against specific KSB criteria with Pass/Merit/Referral grade descriptors
2. CITE specific evidence from the student's submission
3. RECOMMEND a grade level (Pass, Merit, or Referral) with clear justification
4. IDENTIFY what's needed to improve to the next level
5. MAINTAIN a supportive, constructive tone

CRITICAL RULES:

EVIDENCE-BASED GRADING:
- Base your grade recommendation ONLY on evidence present in the submission
- Quote specific text to support your assessment
- If evidence is strong, acknowledge it - don't understate good work
- If evidence is missing, clearly state what's absent

GRADE LEVEL DETERMINATION:
- REFERRAL: Does not meet minimum Pass criteria; significant gaps or errors
- PASS: Meets the basic requirements described in Pass criteria
- MERIT: Exceeds Pass and demonstrates the qualities in Merit criteria

ACCURACY REQUIREMENTS:
- DO NOT claim something is missing if it exists in the evidence
- DO NOT recommend Referral if Pass criteria are clearly met
- ACKNOWLEDGE methodology/planning even if results are TBD
- USE ONLY the page/section numbers provided in the evidence

CITATION ACCURACY:
- ONLY use page numbers and section numbers explicitly provided
- DO NOT invent section numbers like "2.1", "3.3" - use actual references
- Format citations as "(page X / Section Y)"

Your feedback should help the student understand their current level and exactly what's needed to achieve a higher grade."""

    # ==========================================================================
    # KSB EVALUATION PROMPT
    # ==========================================================================
    
    KSB_EVALUATION_PROMPT = """Evaluate the student's work against this specific KSB criterion.

<ksb_criterion>
KSB Code: {ksb_code}
KSB Title: {ksb_title}

PASS Criteria (minimum acceptable):
{pass_criteria}

MERIT Criteria (strong/higher standard):
{merit_criteria}

REFERRAL Criteria (not yet achieved):
{referral_criteria}
</ksb_criterion>

<evidence_from_submission>
{evidence_text}
</evidence_from_submission>

Provide your evaluation in the following structure:

## Evidence Found

List ALL relevant evidence from the submission. Be thorough - search for content related to this KSB across all provided evidence.

Format:
- "[Quote or paraphrase]" (page X / Section Y)
- "[Quote or paraphrase]" (page X / Section Y)

If no relevant evidence found: "No direct evidence found for this KSB."

## Assessment Against Grade Criteria

### Pass Criteria Assessment
For each element of the Pass criteria, state whether it is:
✅ MET - [brief evidence reference]
⚠️ PARTIALLY MET - [what's present vs missing]
❌ NOT MET - [what's missing]

### Merit Criteria Assessment  
For each element of the Merit criteria, state whether it is:
✅ MET - [brief evidence reference]
⚠️ PARTIALLY MET - [what's present vs missing]
❌ NOT MET - [what would be needed]

## Recommended Grade: [PASS / MERIT / REFERRAL]

**Justification:** [2-3 sentences explaining why this grade level, referencing the criteria above]

## Strengths for this KSB
- [Strength 1 with evidence reference]
- [Strength 2 with evidence reference]

## To Achieve Higher Grade
[If Referral: What's needed for Pass]
[If Pass: What's needed for Merit]
[If Merit: What would make this exceptional]

Specific actions:
1. [Concrete action to improve]
2. [Concrete action to improve]

---

IMPORTANT: 
- Base your grade ONLY on the evidence provided
- If Pass criteria are met, do NOT recommend Referral
- If work shows planning/methodology but results are TBD, assess the planning quality
- Use only the citation references provided in the evidence"""

    # ==========================================================================
    # OVERALL KSB SUMMARY PROMPT
    # ==========================================================================
    
    KSB_OVERALL_SUMMARY_PROMPT = """Based on the KSB-by-KSB evaluations below, provide an overall assessment of this coursework submission.

<ksb_evaluations>
{evaluations_text}
</ksb_evaluations>

Provide your overall summary in the following structure:

## Overall Assessment

Write 2-3 paragraphs covering:
- Overall quality and effort evident in the submission
- Pattern of strengths across KSBs
- Common areas needing development
- Readiness for the workplace based on demonstrated competencies

## KSB Summary Table

| KSB | Grade | Key Evidence |
|-----|-------|--------------|
[For each KSB, summarize in one row]

## Grade Profile

- **Knowledge (K) KSBs:** [Summary of performance across K1, K2, K16, K18, K19, K25]
- **Skills (S) KSBs:** [Summary of performance across S15, S16, S19, S23]
- **Behaviours (B) KSBs:** [Summary of performance across B5]

## Key Strengths (Top 3-4)
1. [Strength] - demonstrated in [KSBs]
2. [Strength] - demonstrated in [KSBs]
3. [Strength] - demonstrated in [KSBs]

## Priority Development Areas (Top 3-4)
1. [Area] - affects [KSBs] - [specific action to improve]
2. [Area] - affects [KSBs] - [specific action to improve]
3. [Area] - affects [KSBs] - [specific action to improve]

## Overall Recommendation

[State whether the submission demonstrates readiness across the targeted KSBs, and what key actions would strengthen it]

---

Maintain a supportive tone throughout. The goal is to help the student understand their performance and provide a clear path to improvement."""

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
        """Format the KSB evaluation prompt with provided content."""
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
        """Format the overall summary prompt with KSB evaluations."""
        return cls.KSB_OVERALL_SUMMARY_PROMPT.format(
            evaluations_text=evaluations_text
        )
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for KSB evaluation."""
        return cls.SYSTEM_PROMPT_KSB_MARKER


# ==========================================================================
# EXAMPLE OUTPUT FORMAT
# ==========================================================================

EXAMPLE_KSB_OUTPUT = """
## Evidence Found

- "Used supervised learning (text classification) to meet a business objective: faster, more consistent support ticket triage." (page 1 / Section 1)
- "Defined evaluation metrics aligned to the operational goal (e.g., per-class recall for urgent cases)." (page 7 / Section 7)
- "The business goal is to improve triage speed, consistency, and reporting in a GDPR-compliant way while controlling cost." (page 1 / Executive Summary)

## Assessment Against Grade Criteria

### Pass Criteria Assessment
- States a clear business problem: ✅ MET - Support ticket triage speed and consistency (page 1)
- Identifies appropriate ML approach: ✅ MET - Supervised learning text classification (page 1)
- Describes why approach fits objective: ✅ MET - Links to operational goal with metrics (page 7)

### Merit Criteria Assessment
- Strong problem framing: ✅ MET - Clear business context with UK training provider scenario
- Justification of methodology choices: ⚠️ PARTIALLY MET - Explains text classification but doesn't compare to alternatives
- Alternatives considered: ❌ NOT MET - No explicit comparison to baseline or alternative approaches
- Reasoned selection tied to outcomes: ✅ MET - Links to per-class recall for urgent cases

## Recommended Grade: PASS

**Justification:** The submission clearly meets all Pass criteria with a well-defined business problem (support ticket triage), appropriate ML approach (supervised text classification), and explicit connection to business objectives. However, it falls short of Merit as it doesn't discuss alternative approaches or provide explicit comparison with baseline methods.

## Strengths for this KSB
- Clear articulation of business problem with specific context (UK training provider)
- Explicit link between ML approach and business metrics (recall for urgent cases)
- Well-defined scope with GDPR and cost considerations

## To Achieve Merit Grade

The submission needs to demonstrate consideration of alternatives. Specific actions:
1. Add a brief comparison: Why neural network text classification vs. traditional ML (e.g., SVM, Naive Bayes) or rule-based approaches?
2. Include a baseline comparison: What would manual triage or a simple keyword-based system achieve?
3. Quantify the expected improvement: What accuracy/speed improvement justifies the ML approach?
"""
