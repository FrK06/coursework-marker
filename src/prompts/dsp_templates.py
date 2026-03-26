"""
DSP-specific evaluation prompts — Data Science Principles.

KSBs: K2, K5, K15, K20, K22, K24, K26, K27, S1, S9, S10, S13, S17, S18, S21, S22, S26, B3, B7

Module-specific merit guidance supplements the shared 7-step evaluation
scaffold from base_templates.py.
"""
from .base_templates import EVALUATION_STEPS, SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# DSP-specific merit guidance per KSB category
# ═══════════════════════════════════════════════════════════════════════════════

_STATISTICAL_MERIT_GUIDANCE = """**DSP statistical KSB guidance (K22, K26, S22):**
Check for:
- Effect size measures (Cohen's d, odds ratio, R-squared, eta-squared)
- Confidence intervals (95% CI, credible intervals) with interpretation
- Assumption testing (normality tests, homogeneity of variance, independence checks)
- p-values WITH correct interpretation (not just "significant/not significant")
- Practical significance vs statistical significance discussion
- Uncertainty quantification beyond p-values
- Business/organisational decision explicitly linked to statistical results

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_UNCERTAINTY_MERIT_GUIDANCE = """**DSP uncertainty KSB guidance (S21):**
Check for:
- Confidence intervals with numerical bounds
- Effect size with magnitude interpretation
- Error bars on visualisations
- Sensitivity analysis or robustness checks
- Explicit discussion of sampling error vs measurement error
- Limitations section acknowledging uncertainty sources
- Practical significance linked to confidence level

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_VISUALISATION_MERIT_GUIDANCE = """**DSP visualisation KSB guidance (S17, S18):**
Check for:
- Chart labelling (axis titles, legends, units)
- Data sources cited on visualisations
- Insights discussed for each visualisation (not just "Figure X shows...")
- Appropriate chart type selection (bar vs line vs scatter vs heatmap)
- Dashboard or monitoring view with operational metrics
- Infrastructure diagrams with component labels
- Data pipeline health / freshness / lineage indicators

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_PROGRAMMING_MERIT_GUIDANCE = """**DSP programming/engineering KSB guidance (S1, S9, S10):**
Check for:
- Code quality evidence (modular structure, functions, error handling)
- Documentation (README, docstrings, inline comments)
- Reproducibility (requirements.txt, config files, versioning)
- Storage architecture rationale with trade-offs
- Data manipulation pipeline with clear transformation steps
- Dataset selection justification against business problem
- Methodology rationale explaining why alternatives were rejected

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_INFRASTRUCTURE_MERIT_GUIDANCE = """**DSP infrastructure KSB guidance (K2, K15, K20, K27, S13):**
Check for:
- Storage architecture design with technology choices justified
- End-to-end data flow from collection to insight
- Data product specification with stakeholder requirements
- Platform/tool selection rationale (Power BI, Python, SQL, cloud)
- Data dictionary or schema documentation
- Instrumentation design for data collection
- Cost/performance/security trade-offs

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_DATA_QUALITY_MERIT_GUIDANCE = """**DSP data quality KSB guidance (S17):**
Check for:
- Schema validation rules
- Duplicate detection and handling
- Missing value strategy (imputation, deletion, flagging)
- Data dictionary with field definitions
- Data lineage documentation
- Quality monitoring approach
- Repeatability of cleaning pipeline

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_ETHICS_MERIT_GUIDANCE = """**DSP ethics/compliance KSB guidance (K24, B3):**
Check for:
- GDPR-aware data handling (anonymisation, pseudonymisation)
- Lawful basis identification for data processing
- Data retention policy or lifecycle discussion
- Access control mechanisms
- DPIA or risk assessment approach
- Ethical considerations for future ML use of data
- Compliance-by-design decisions documented

If 3+ of these elements are present with evidence, mark merit criteria as MET."""

_REFLECTION_MERIT_GUIDANCE = """**DSP reflection/sharing KSB guidance (B7):**
Check for:
- Concrete dissemination plan (reusable assets, templates, standards)
- Stakeholder enablement evidence
- Community contribution (show-and-tell, documentation sharing)
- Reflective analysis of lessons learned
- Best practice identification from the project

If adversarial reflection table detected (KSB codes in table but off-topic) -> auto-REFERRAL."""

_ANALYSIS_MERIT_GUIDANCE = """**DSP analysis KSB guidance (K5, S26):**
Check for:
- Structured EDA approach (distributions, correlations, outliers)
- Research methodology linked to business need
- Technique selection rationale (why this method vs alternatives)
- Evaluation metrics with interpretation
- Insight discovery with supporting evidence
- Recommendation grounded in analysis findings

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

# Map KSB codes to their specific merit guidance
DSP_MERIT_GUIDANCE = {
    # Statistical KSBs
    'K22': _STATISTICAL_MERIT_GUIDANCE,
    'K26': _STATISTICAL_MERIT_GUIDANCE,
    'S22': _STATISTICAL_MERIT_GUIDANCE,
    'S21': _UNCERTAINTY_MERIT_GUIDANCE,
    # Visualisation KSBs
    'S17': _VISUALISATION_MERIT_GUIDANCE,
    'S18': _VISUALISATION_MERIT_GUIDANCE,
    # Programming KSBs
    'S1': _PROGRAMMING_MERIT_GUIDANCE,
    'S9': _PROGRAMMING_MERIT_GUIDANCE,
    'S10': _PROGRAMMING_MERIT_GUIDANCE,
    # Infrastructure KSBs
    'K2': _INFRASTRUCTURE_MERIT_GUIDANCE,
    'K15': _INFRASTRUCTURE_MERIT_GUIDANCE,
    'K20': _INFRASTRUCTURE_MERIT_GUIDANCE,
    'K27': _INFRASTRUCTURE_MERIT_GUIDANCE,
    'S13': _INFRASTRUCTURE_MERIT_GUIDANCE,
    # Data quality
    'S17': _DATA_QUALITY_MERIT_GUIDANCE,
    # Analysis KSBs
    'K5': _ANALYSIS_MERIT_GUIDANCE,
    'S26': _ANALYSIS_MERIT_GUIDANCE,
    # Ethics/compliance
    'K24': _ETHICS_MERIT_GUIDANCE,
    'B3': _ETHICS_MERIT_GUIDANCE,
    # Reflection
    'B7': _REFLECTION_MERIT_GUIDANCE,
}

# KSBs that should auto-REFERRAL on adversarial content
DSP_AUTO_REFERRAL_KSBS = {'B7'}


# ═══════════════════════════════════════════════════════════════════════════════
# DSP evaluation prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_dsp_evaluation_prompt(
    ksb_code: str,
    ksb_title: str,
    pass_criteria: str,
    merit_criteria: str,
    referral_criteria: str,
    evidence_text: str,
    brief_context: str,
    image_context: str = "",
) -> str:
    """
    Build a complete DSP evaluation prompt for a single KSB.

    Combines the shared 7-step scaffold with DSP-specific merit guidance.
    """
    merit_guidance = DSP_MERIT_GUIDANCE.get(ksb_code, "")

    steps = EVALUATION_STEPS.format(
        module_specific_merit_guidance=merit_guidance,
        ksb_code=ksb_code,
    )

    prompt = f"""Evaluate this student's work against the KSB criterion below.

ANTI-HALLUCINATION REMINDER:
- ONLY use text from the EVIDENCE section below
- The ASSIGNMENT BRIEF shows what was required
- If you cannot find evidence -> write "NOT FOUND"
- NEVER invent quotes, company names, or technical details

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

---

{steps}"""

    return prompt


def get_dsp_system_prompt() -> str:
    """Return the system prompt for DSP evaluation."""
    return SYSTEM_PROMPT
