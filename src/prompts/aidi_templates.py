"""
AIDI-specific evaluation prompts — AI-Driven Innovation.

KSBs: K1, K4, K5, K6, K8, K9, K11, K12, K21, K24, K29, S3, S5, S6, S25, S26, B3, B4, B8

Module-specific merit guidance supplements the shared 7-step evaluation
scaffold from base_templates.py.
"""
from .base_templates import EVALUATION_STEPS, SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# AIDI-specific merit guidance per KSB category
# ═══════════════════════════════════════════════════════════════════════════════

_INNOVATION_MERIT_GUIDANCE = """**AIDI innovation KSB guidance (K4, K5, K6):**
Check for:
- Business impact quantification (KPIs, ROI, cost savings, efficiency gains)
- Stakeholder identification with roles and communication plan
- Change management evidence (adoption plan, training, feedback loops)
- Data linkage across multiple systems with schema/join evidence
- Iterative delivery approach (agile, MVP, sprint evidence)
- User/domain research informing solution design
- Prioritisation rationale for feature scope

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_AI_ETHICS_MERIT_GUIDANCE = """**AIDI AI ethics KSB guidance (K8, K9):**
Check for:
- Responsible AI principles applied (fairness, transparency, accountability)
- Risk assessment with specific risks identified and mitigated
- Governance framework reference (organisational policy, SDLC compliance)
- DPIA-style analysis with lawful basis identification
- IP and licensing considerations for data and models
- Audit trail or accountability mechanisms
- Compliance-by-design decisions documented

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_SOCIAL_ETHICS_MERIT_GUIDANCE = """**AIDI social/ethical context KSB guidance (K12, K24):**
Check for:
- Balanced assessment of harms and benefits of the AI system
- Affected groups identified with impact analysis
- Practical mitigations (human-in-the-loop, transparency measures)
- Bias and error analysis with structured testing
- Sources of error identified with robustness checks
- Explainability approach for model decisions
- Foreseeable misuse scenarios addressed

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_STRATEGY_MERIT_GUIDANCE = """**AIDI strategy KSB guidance (K11, K12):**
Check for:
- Strategic alignment of AI solution to business goals
- ROI or cost-benefit analysis with numbers
- Implementation roadmap with milestones
- Role clarity (AI/DS/DE responsibilities in lifecycle)
- Governance and monitoring plan for production
- Societal impact assessment
- Scaling and retraining considerations

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_STAKEHOLDER_MERIT_GUIDANCE = """**AIDI stakeholder/communication KSB guidance (K21, S3, S5, S6):**
Check for:
- Stakeholder analysis with audience segmentation
- Communication tailored to technical vs non-technical audiences
- Requirements elicitation with user stories or acceptance criteria
- Success criteria and KPIs defined and tracked
- Technical roadmap with scaling, resourcing, and governance
- Handover documentation (runbook, adoption plan)
- Direction/guidance on AI/DS opportunities

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_ENGINEERING_MERIT_GUIDANCE = """**AIDI engineering KSB guidance (S25, S26):**
Check for:
- Version control evidence (Git commits, branching strategy)
- Unit tests or integration tests
- Modular code structure with clear separation of concerns
- Logging and error handling
- Reproducibility (requirements.txt, config, environment setup)
- Technique selection rationale with benchmarking or alternatives
- Evaluation metrics with interpretation

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_ACCESSIBILITY_MERIT_GUIDANCE = """**AIDI accessibility KSB guidance (K29):**
Check for:
- WCAG awareness or compliance discussion
- Inclusive design decisions documented
- Assistive technology considerations (screen readers, contrast, etc.)
- Accessibility testing evidence or checklists
- Trade-offs between functionality and accessibility
- Diverse user needs analysis

If 3+ of these elements are present with evidence, mark merit criteria as MET."""

_INTEGRITY_MERIT_GUIDANCE = """**AIDI integrity/ethics KSB guidance (B3):**
Check for:
- Proactive integrity controls (not just awareness)
- Transparent limitations documented
- Responsible AI approach with concrete decisions
- Data protection safeguards implemented
- Security measures for data handling
- Ethical risk documentation

If 3+ of these elements are present with evidence, mark merit criteria as MET."""

_INITIATIVE_MERIT_GUIDANCE = """**AIDI initiative/responsibility KSB guidance (B4):**
Check for:
- Evidence of overcoming specific challenges
- Iterative improvement documented (what changed and why)
- Decision-making ownership with justification
- Learning from failures with adaptation evidence
- Scope management (descoping, pivoting, prioritising)

If adversarial reflection table detected -> auto-REFERRAL."""

_TRENDS_MERIT_GUIDANCE = """**AIDI trends/innovation KSB guidance (B8):**
Check for:
- Current literature references (academic papers, industry reports)
- Trend awareness in AI/ML domain with synthesis
- Sources linked to design decisions or business value
- Innovation landscape understanding
- Benchmarking against state-of-the-art

If adversarial reflection table detected -> auto-REFERRAL."""

_METHODOLOGY_MERIT_GUIDANCE = """**AIDI methodology KSB guidance (K1):**
Check for:
- AI/ML method justified against alternatives
- Business objective clearly linked to methodology choice
- Constraints acknowledged (data, compute, time)
- Measurable business value articulated
- Method selection rationale comparing at least 2 options

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

# Map KSB codes to their specific merit guidance
AIDI_MERIT_GUIDANCE = {
    # Innovation KSBs
    'K1': _METHODOLOGY_MERIT_GUIDANCE,
    'K4': _INNOVATION_MERIT_GUIDANCE,
    'K5': _INNOVATION_MERIT_GUIDANCE,
    'K6': _INNOVATION_MERIT_GUIDANCE,
    # AI ethics KSBs
    'K8': _AI_ETHICS_MERIT_GUIDANCE,
    'K9': _AI_ETHICS_MERIT_GUIDANCE,
    # Strategy KSBs
    'K11': _STRATEGY_MERIT_GUIDANCE,
    'K12': _SOCIAL_ETHICS_MERIT_GUIDANCE,
    # Stakeholder KSBs
    'K21': _STAKEHOLDER_MERIT_GUIDANCE,
    'S3': _STAKEHOLDER_MERIT_GUIDANCE,
    'S5': _STAKEHOLDER_MERIT_GUIDANCE,
    'S6': _STAKEHOLDER_MERIT_GUIDANCE,
    # Social/ethical
    'K24': _SOCIAL_ETHICS_MERIT_GUIDANCE,
    # Accessibility
    'K29': _ACCESSIBILITY_MERIT_GUIDANCE,
    # Engineering
    'S25': _ENGINEERING_MERIT_GUIDANCE,
    'S26': _ENGINEERING_MERIT_GUIDANCE,
    # Integrity
    'B3': _INTEGRITY_MERIT_GUIDANCE,
    # Initiative
    'B4': _INITIATIVE_MERIT_GUIDANCE,
    # Trends
    'B8': _TRENDS_MERIT_GUIDANCE,
}

# KSBs that should auto-REFERRAL on adversarial content
AIDI_AUTO_REFERRAL_KSBS = {'B3', 'B4', 'B8'}


# ═══════════════════════════════════════════════════════════════════════════════
# AIDI evaluation prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_aidi_evaluation_prompt(
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
    Build a complete AIDI evaluation prompt for a single KSB.

    Combines the shared 7-step scaffold with AIDI-specific merit guidance.
    """
    merit_guidance = AIDI_MERIT_GUIDANCE.get(ksb_code, "")

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


def get_aidi_system_prompt() -> str:
    """Return the system prompt for AIDI evaluation."""
    return SYSTEM_PROMPT
