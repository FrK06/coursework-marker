"""
MLCC-specific evaluation prompts — Machine Learning with Cloud Computing.

KSBs: K1, K2, K16, K18, K19, K25, S15, S16, S19, S23, B5

Each KSB has module-specific merit guidance that supplements the shared
7-step evaluation scaffold from base_templates.py.
"""
from .base_templates import EVALUATION_STEPS, SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# MLCC-specific merit guidance per KSB category
# ═══════════════════════════════════════════════════════════════════════════════

_CLOUD_MERIT_GUIDANCE = """**MLCC cloud KSB guidance (K18, K19, S16):**
Check for:
- Specific cloud platform references (AWS, Azure, GCP) with service names
- Deployment evidence (screenshots, logs, CLI output)
- Scalability discussion (auto-scaling, load balancing, distributed training)
- Cost/performance trade-offs with specific numbers
- Security controls (IAM, VPC, encryption, KMS)

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_ML_MODEL_MERIT_GUIDANCE = """**MLCC model KSB guidance (K16, S15):**
Check for:
- Model selection rationale comparing alternatives (e.g., why NN vs gradient boosting)
- Training/validation/test split with justification
- Performance metrics (accuracy, F1, precision, recall, AUC) with interpretation
- Hyperparameter tuning methodology (grid search, Bayesian, etc.)
- Overfitting/underfitting analysis with evidence (learning curves, regularization)
- GPU vs CPU profiling with measured timings

If 4+ of these elements are present with evidence, mark merit criteria as MET."""

_ETHICS_MERIT_GUIDANCE = """**MLCC ethics KSB guidance (B5):**
Check for:
- Fairness discussion (demographic parity, equalised odds, etc.)
- Bias analysis of training data and model outputs
- Transparency / explainability measures (SHAP, LIME, feature importance)
- Accountability mechanisms (audit trail, monitoring, human oversight)
- Concrete CPD evidence (courses, certifications, experimentation logs)
- Reflective improvement loop tied to project outcomes

If 3+ of these elements are present with evidence, mark merit criteria as MET."""

_REFLECTION_MERIT_GUIDANCE = """**MLCC reflection KSB guidance (S23):**
Check for:
- Specific best practices identified from the project
- Concrete dissemination plan (playbooks, templates, documentation)
- Stakeholder communication strategy
- Governance alignment and compliance considerations
- Evidence of knowledge sharing (show-and-tell, training sessions)

If adversarial reflection table detected (KSB codes in table but off-topic) → auto-REFERRAL."""

# Map KSB codes to their specific merit guidance
MLCC_MERIT_GUIDANCE = {
    'K1': _ML_MODEL_MERIT_GUIDANCE,
    'K2': _CLOUD_MERIT_GUIDANCE,
    'K16': _ML_MODEL_MERIT_GUIDANCE,
    'K18': _CLOUD_MERIT_GUIDANCE,
    'K19': _ML_MODEL_MERIT_GUIDANCE,
    'K25': _ML_MODEL_MERIT_GUIDANCE,
    'S15': _ML_MODEL_MERIT_GUIDANCE,
    'S16': _CLOUD_MERIT_GUIDANCE,
    'S19': _CLOUD_MERIT_GUIDANCE,
    'S23': _REFLECTION_MERIT_GUIDANCE,
    'B5': _ETHICS_MERIT_GUIDANCE,
}

# KSBs that should auto-REFERRAL on adversarial content
MLCC_AUTO_REFERRAL_KSBS = {'B5', 'S23'}


# ═══════════════════════════════════════════════════════════════════════════════
# MLCC evaluation prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_mlcc_evaluation_prompt(
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
    Build a complete MLCC evaluation prompt for a single KSB.

    Combines the shared 7-step scaffold with MLCC-specific merit guidance.
    """
    merit_guidance = MLCC_MERIT_GUIDANCE.get(ksb_code, "")

    steps = EVALUATION_STEPS.format(
        module_specific_merit_guidance=merit_guidance,
        ksb_code=ksb_code,
    )

    prompt = f"""Evaluate this student's work against the KSB criterion below.

ANTI-HALLUCINATION REMINDER:
- ONLY use text from the EVIDENCE section below
- The ASSIGNMENT BRIEF shows what was required
- If you cannot find evidence → write "NOT FOUND"
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


def get_mlcc_system_prompt() -> str:
    """Return the system prompt for MLCC evaluation."""
    return SYSTEM_PROMPT
