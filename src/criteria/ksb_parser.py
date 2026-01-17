"""
KSB Rubric Parser - Extracts structured KSB criteria from rubric tables.

Parses markdown tables or text with KSB criteria including:
- KSB code and description
- Pass criteria (minimum acceptable)
- Merit criteria (higher standard)
- Referral criteria (not yet achieved)
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class KSBCriterion:
    """A single KSB criterion with grade descriptors."""
    code: str  # e.g., "K1", "S16", "B5"
    title: str  # e.g., "ML methodologies to meet business objectives"
    full_description: str  # Combined code and title
    
    # Grade level criteria
    pass_criteria: str  # What's needed for Pass
    merit_criteria: str  # What's needed for Merit
    referral_criteria: str  # What indicates Referral
    
    # Additional metadata
    category: str = ""  # "Knowledge", "Skill", or "Behaviour"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'title': self.title,
            'full_description': self.full_description,
            'pass_criteria': self.pass_criteria,
            'merit_criteria': self.merit_criteria,
            'referral_criteria': self.referral_criteria,
            'category': self.category
        }
    
    def get_evaluation_context(self) -> str:
        """Format criterion for LLM evaluation."""
        return f"""KSB: {self.code} – {self.title}

PASS Criteria (minimum acceptable):
{self.pass_criteria}

MERIT Criteria (strong/higher standard):
{self.merit_criteria}

REFERRAL Criteria (not yet achieved):
{self.referral_criteria}"""


class KSBRubricParser:
    """
    Parser for KSB rubric tables.
    
    Handles:
    - Markdown table format
    - Plain text format with KSB patterns
    - Various table structures
    """
    
    # Pattern to detect KSB codes
    KSB_PATTERN = re.compile(
        r'\*?\*?([KSB]\d{1,2})\*?\*?\s*[–-]\s*(.+?)(?:\*\*)?$',
        re.MULTILINE
    )
    
    def __init__(self):
        self.ksb_categories = {
            'K': 'Knowledge',
            'S': 'Skill',
            'B': 'Behaviour'
        }
    
    def parse_markdown_table(self, table_text: str) -> List[KSBCriterion]:
        """
        Parse a markdown table containing KSB rubric.
        
        Expected columns:
        | KSB | Criteria to PASS | Criteria to MERIT | Criteria for REFERRAL |
        """
        criteria = []
        
        # Split into lines and find data rows (skip header and separator)
        lines = table_text.strip().split('\n')
        data_rows = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|--') or line.startswith('| --'):
                continue
            if '|' in line:
                # Check if it's a header row
                line_lower = line.lower()
                if 'ksb' in line_lower and ('pass' in line_lower or 'merit' in line_lower):
                    continue  # Skip header
                data_rows.append(line)
        
        for row in data_rows:
            criterion = self._parse_table_row(row)
            if criterion:
                criteria.append(criterion)
        
        logger.info(f"Parsed {len(criteria)} KSB criteria from table")
        return criteria
    
    def _parse_table_row(self, row: str) -> Optional[KSBCriterion]:
        """Parse a single table row into a KSBCriterion."""
        # Split by | and clean up
        cells = [cell.strip() for cell in row.split('|')]
        cells = [c for c in cells if c]  # Remove empty cells
        
        if len(cells) < 4:
            return None
        
        # First cell should contain KSB code and description
        ksb_cell = cells[0]
        
        # Extract KSB code (K1, S16, B5, etc.)
        code_match = re.search(r'\*?\*?([KSB]\d{1,2})\*?\*?', ksb_cell)
        if not code_match:
            return None
        
        code = code_match.group(1)
        
        # Extract title (everything after the code and dash)
        title_match = re.search(r'[KSB]\d{1,2}\*?\*?\s*[–-]\s*(.+)', ksb_cell)
        title = title_match.group(1).strip().strip('*') if title_match else ksb_cell
        
        # Get criteria for each level
        pass_criteria = cells[1] if len(cells) > 1 else ""
        merit_criteria = cells[2] if len(cells) > 2 else ""
        referral_criteria = cells[3] if len(cells) > 3 else ""
        
        # Determine category
        category = self.ksb_categories.get(code[0], "Unknown")
        
        return KSBCriterion(
            code=code,
            title=title,
            full_description=f"{code} – {title}",
            pass_criteria=pass_criteria,
            merit_criteria=merit_criteria,
            referral_criteria=referral_criteria,
            category=category
        )
    
    def parse_text(self, text: str) -> List[KSBCriterion]:
        """
        Parse KSB criteria from general text format.
        
        Tries to identify KSB codes and associated criteria.
        """
        # First try to find markdown table
        if '|' in text and ('PASS' in text.upper() or 'MERIT' in text.upper()):
            return self.parse_markdown_table(text)
        
        # Otherwise try to extract from structured text
        criteria = []
        
        # Find all KSB references
        ksb_matches = list(self.KSB_PATTERN.finditer(text))
        
        for i, match in enumerate(ksb_matches):
            code = match.group(1)
            title = match.group(2).strip()
            
            # Get text until next KSB or end
            start_pos = match.end()
            end_pos = ksb_matches[i + 1].start() if i + 1 < len(ksb_matches) else len(text)
            content = text[start_pos:end_pos].strip()
            
            # Try to extract Pass/Merit/Referral sections
            pass_criteria = self._extract_grade_section(content, 'pass')
            merit_criteria = self._extract_grade_section(content, 'merit')
            referral_criteria = self._extract_grade_section(content, 'referral')
            
            # If no grade sections found, use the whole content as general criteria
            if not any([pass_criteria, merit_criteria, referral_criteria]):
                pass_criteria = content[:500] if content else "See assignment brief"
            
            category = self.ksb_categories.get(code[0], "Unknown")
            
            criteria.append(KSBCriterion(
                code=code,
                title=title,
                full_description=f"{code} – {title}",
                pass_criteria=pass_criteria or "Meets basic requirements",
                merit_criteria=merit_criteria or "Exceeds basic requirements with strong evidence",
                referral_criteria=referral_criteria or "Does not meet minimum requirements",
                category=category
            ))
        
        return criteria
    
    def _extract_grade_section(self, text: str, grade: str) -> str:
        """Extract criteria text for a specific grade level."""
        patterns = {
            'pass': [
                r'pass[:\s]+(.+?)(?=merit|referral|$)',
                r'minimum[:\s]+(.+?)(?=merit|strong|higher|$)',
                r'basic[:\s]+(.+?)(?=merit|strong|$)'
            ],
            'merit': [
                r'merit[:\s]+(.+?)(?=referral|pass|$)',
                r'strong[:\s]+(.+?)(?=referral|not yet|$)',
                r'higher[:\s]+(.+?)(?=referral|$)'
            ],
            'referral': [
                r'referral[:\s]+(.+?)(?=pass|merit|$)',
                r'not yet[:\s]+(.+?)(?=pass|merit|$)',
                r'fail[:\s]+(.+?)(?=pass|merit|$)'
            ]
        }
        
        for pattern in patterns.get(grade, []):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        return ""
    
    def get_ksb_by_code(
        self, 
        criteria: List[KSBCriterion], 
        code: str
    ) -> Optional[KSBCriterion]:
        """Get a specific KSB criterion by its code."""
        for c in criteria:
            if c.code.upper() == code.upper():
                return c
        return None
    
    def get_ksbs_by_category(
        self, 
        criteria: List[KSBCriterion], 
        category: str
    ) -> List[KSBCriterion]:
        """Get all KSBs of a specific category (Knowledge, Skill, Behaviour)."""
        return [c for c in criteria if c.category.lower() == category.lower()]


# =============================================================================
# MODULE RUBRICS
# =============================================================================

# MLCC Module 3 - Machine Learning with Cloud Computing
DEFAULT_MLCC_RUBRIC = """
| KSB | Criteria to PASS | Criteria to MERIT | Criteria for REFERRAL |
| --- | --- | --- | --- |
| **K1 – ML methodologies to meet business objectives** | States a clear business problem and identifies an appropriate ML approach (e.g., supervised learning). Describes why the approach fits the objective at a basic level. | Strong problem framing and justification of methodology choices (e.g., why NN vs baseline). Includes alternatives considered and a reasoned selection tied to business outcomes. | ML approach is unclear or mismatched to objective. Little/no link between model choice and business need. |
| **K2 – Apply modern storage/processing/ML methods for organisational impact** | Identifies storage/processing choices (e.g., S3, object storage, pipelines) and explains how they support the workflow. Mentions governance/security at a basic level. | Demonstrates an end-to-end data flow with clear rationale (trade-offs: cost, throughput, latency, reliability). Shows how design choices maximise impact (e.g., reproducibility, scalability). | Storage/processing decisions are missing, wrong, or unjustified. Data flow is unclear; weak/no organisational impact reasoning. |
| **K16 – High-performance architectures and effective use** | Explains CPU vs GPU (and TPU if relevant) at a basic level and relates this to training/inference. Includes at least some performance or configuration evidence. | Shows informed optimisation decisions (instance selection, profiling, bottlenecks, batch size, mixed precision, distributed training if used). Links architecture choices to measured results/costs. | No credible understanding of compute options. No evidence of effective use or incorrect claims about architecture/performance. |
| **K18 – Programming languages and techniques for data engineering** | Uses appropriate code tools (e.g., Python) with clear preprocessing steps and documented pipeline stages. Code/process is understandable and reproducible at a basic level. | Clean structure (modularity, config, logging), good engineering practice (versioning, reproducible runs), and robust handling of errors/edge cases. Clear documentation of steps and decisions. | Little/no evidence of data engineering practice. Pipeline is not reproducible, poorly described, or technically incorrect. |
| **K19 – Statistical/ML principles and properties** | Demonstrates basic ML principles: train/val/test split, evaluation metric, and explains overfitting/underfitting at a basic level. | Shows deeper analysis: bias/variance considerations, metric choice justification, calibration/threshold discussion (if relevant), error analysis, and limitations. Evidence-based reasoning. | Weak/incorrect ML reasoning (e.g., data leakage, invalid evaluation). Metrics missing or misused; conclusions not supported. |
| **K25 – ML libraries for commercially beneficial analysis/simulation** | Uses suitable ML libraries (PyTorch/TensorFlow/Keras etc.) correctly and explains what was used and why. Runs successfully in cloud context. | Demonstrates effective library usage (callbacks, checkpoints, experiment tracking, efficient data loaders). Shows awareness of best practices and production considerations. | Library usage is incorrect, undocumented, or fails to run. Choices appear arbitrary or not aligned to requirements. |
| **S15 – Develop/build/maintain services/platforms delivering AI** | Produces a working PoC artefact in the cloud (training/inference), with basic deployment/run instructions and evidence (screenshots/logs). | PoC is robust and maintainable: repeatable deployment, clear monitoring/logging, sensible automation (scripts/IaC), and operational considerations. | PoC is missing/non-functional, or cannot be reproduced. No operational evidence or unclear instructions. |
| **S16 – Define requirements and supervise data management infrastructure (cloud)** | States functional + non-functional requirements (security, availability, latency, throughput, governance) and links them to architecture choices. Includes GDPR considerations. | Requirements are well-structured and traceable to design decisions. Includes clear mitigations, risk controls, IAM/network detail, and a justified go/no-go conclusion. | Requirements are absent or not linked to design. GDPR/security largely ignored. No credible feasibility/go-no-go conclusion. |
| **S19 – Use scalable infra / services management to generate solutions** | Uses cloud resources appropriately and includes some benchmarking (e.g., CPU vs GPU) with basic cost/performance commentary. | Strong benchmarking methodology: repeatable experiments, clear comparison criteria, cost/performance trade-offs, and scalability reasoning (autoscaling, distributed options, constraints). | No meaningful benchmarking or scalability discussion. Unsupported claims about performance/cost; weak operational awareness. |
| **S23 – Disseminate AI/DS practices and best practice** | Provides a reflective section describing what would be shared with others (documentation, standards, lessons learned). | Clear plan for dissemination: playbooks/templates, stakeholder communication, training/enablement, governance alignment, and how best practice will be embedded. | Reflection is missing or superficial. No evidence of sharing/enablement or best-practice mindset. |
| **B5 – Continuous professional development (CPD)** | Identifies learning undertaken and at least one concrete next step for development based on the project. | Strong CPD: specific evidence of learning (courses, docs, experimentation), reflective improvement loop, and a credible plan tied to role/org needs. | No CPD evidence, or vague statements with no concrete learning actions or reflection. |
"""


def get_default_ksb_criteria() -> List[KSBCriterion]:
    """Get the default KSB criteria for MLCC Module 3."""
    parser = KSBRubricParser()
    return parser.parse_markdown_table(DEFAULT_MLCC_RUBRIC)


# AIDI Module - AI-Driven Innovation / Data Products
DEFAULT_AIDI_RUBRIC = """
| KSB | Criteria to PASS | Criteria to MERIT | Criteria for REFERRAL |
| --- | --- | --- | --- |
| **K1 – AI/ML methodologies to meet business objectives** | Identifies a valid AI/ML method for the product/artefact and links it to the business objective at a basic level. | Justifies methodology choices vs alternatives (incl. constraints) and links to measurable business value. | Method choice is unclear/mismatched; weak link to business objective. |
| **K4 – Extract and link data from multiple systems** | Describes data sources and a plausible approach to extraction/linkage (even if limited in PoC). Mentions identifiers/joins at a basic level. | Demonstrates linkage logic, data lineage, and integration risks; shows evidence (schemas, examples, checks). | Data sources unclear; no credible extraction/linkage approach; major gaps in feasibility. |
| **K5 – Design/deploy data analysis & research to meet needs** | Includes basic analysis/research approach (user, domain, or data) and explains how it informs solution. | Strong research/analysis design with traceable insights informing decisions; limitations acknowledged. | Little/no research or analysis; recommendations not grounded in evidence. |
| **K6 – Deliver data products using iterative/incremental approaches** | Describes delivery approach (agile/iterative/stage-gate) and shows a basic plan for iterations. | Clear iteration strategy with prioritisation, feedback loops, MVP scope control, and delivery risks managed. | No delivery approach or unrealistic plan; unclear how the product would be iterated or managed. |
| **K8 – Interpret organisational policies/standards/guidelines** | References relevant org policies/standards (or plausible equivalents) and applies them to the solution at a basic level. | Shows concrete compliance-by-design decisions (access control, retention, risk approvals, SDLC). | Policies not addressed, or addressed superficially with no application to the work. |
| **K9 – Legal/ethical/professional/regulatory frameworks** | Identifies key legal/ethical issues (GDPR, privacy, IP/licensing) and states basic mitigations. | Applies frameworks with specificity: lawful basis, DPIA-style risks, safeguards, accountability, auditability. | Ignores or misstates major legal/ethical requirements; no mitigations. |
| **K11 – Roles/impact of AI, DS & DE in industry and society** | Explains at a basic level who does what (AI/DS/DE roles) and why it matters for delivery. | Connects roles to lifecycle responsibilities (governance, monitoring, retraining) and real-world impact. | No clear understanding of roles or their impact on outcomes. |
| **K12 – Wider social context, ethical issues (automation/misuse)** | Discusses at least one social/ethical impact (automation, misuse, harm) relevant to the product. | Balanced assessment of harms/benefits, affected groups, and practical mitigations (HITL, transparency). | No meaningful social context; ignores foreseeable misuse/harms. |
| **K21 – How AI/DS supports other team members** | Shows how the solution integrates with stakeholders/teams and supports workflows. | Demonstrates collaboration touchpoints (handover, runbook, stakeholder comms, adoption plan). | No consideration of team integration; "solution in isolation." |
| **K24 – Sources of error and bias** | Identifies likely errors/bias sources and includes at least simple checks or discussion. | Provides structured bias/robustness testing, error analysis, and mitigations; ties to dataset/method choices. | No bias/error thinking or major evaluation flaws (e.g., misleading metrics). |
| **K29 – Accessibility and diverse user needs** | Mentions accessibility needs and includes basic design considerations (e.g., WCAG awareness). | Concrete accessibility plan with testing evidence or checklists, inclusive design decisions, and trade-offs. | Accessibility absent or tokenistic; no plan for diverse user needs. |
| **S3 – Critically evaluate arguments/assumptions/incomplete data; recommend** | Makes recommendations based on some evidence; acknowledges constraints/assumptions. | Strong critical evaluation: compares options, handles uncertainty, justifies recommendations with clear rationale. | Recommendations unsupported, or ignores key uncertainties/assumptions. |
| **S5 – Manage expectations & present insight/solutions/findings to stakeholders** | Identifies stakeholders and communicates findings in a clear, structured way (incl. KPIs). | Tailors messaging by audience; includes clear success criteria, risks, and decisions; strong visuals/tables. | Stakeholder comms unclear; expectations unmanaged; missing KPIs/success criteria. |
| **S6 – Provide direction and technical guidance on AI/DS opportunities** | Offers basic guidance on feasibility, scope, and next steps. | Provides actionable roadmap: scaling, resourcing, governance, monitoring, and adoption steps. | No credible guidance; next steps vague or unrealistic. |
| **S25 – Programming languages/tools & software development practices** | Artefact implemented with basic good practice (readme, dependencies, repeatable run, simple tests/logging). | Strong engineering discipline: version control evidence, unit tests, modular code, logging, reproducibility. | Artefact missing/non-functional, not reproducible, or poor practices with no evidence. |
| **S26 – Select/apply appropriate AI/DS techniques for complex problems** | Technique is appropriate for the problem; evaluation uses suitable metrics at a basic level. | Strong technique selection with benchmarking, ablation/alternatives, and clear metric justification. | Technique poorly chosen or evaluation invalid; no credible evidence of solving the problem. |
| **B3 – Integrity: ethical/legal/regulatory; protect data, safety, security** | Demonstrates awareness and basic safeguards (privacy, security, safe handling). | Proactive integrity: clear controls, transparent limitations, responsible AI approach, documented decisions. | Neglects protections; risky handling of data/security; unethical approach. |
| **B4 – Initiative and responsibility to overcome challenges** | Identifies challenges and shows ownership in resolving them (even if partial). | Evidence of strong initiative: iterates, documents decisions, learns from failures, adapts scope effectively. | Avoids ownership; challenges not addressed; no evidence of problem-solving. |
| **B8 – Awareness of trends/innovation; uses literature and sources** | Uses some relevant references (academic/industry) and links them to the project. | Strong, current literature + trend awareness; synthesises sources into decisions and business value. | Little/no referencing; weak understanding of innovation landscape. |
"""


# DSP Module - Data Science Principles
DEFAULT_DSP_RUBRIC = """
| KSB | Criteria to PASS | Criteria to MERIT | Criteria for REFERRAL |
| --- | --- | --- | --- |
| **K2 – Modern storage/processing/ML methods to maximise organisational impact** | Describes an infrastructure approach and explains, at a basic level, how storage/processing enables analysis and value. Mentions how the organisation would benefit. | Clear trade-offs (cost, scale, governance, latency), realistic technology choices, and strong linkage from infrastructure → insights → business impact. Mentions future ML enablement sensibly. | Infrastructure and impact link is unclear or incorrect; no credible justification of technologies or benefits. |
| **K5 – Design & deploy effective data analysis/research techniques** | Applies appropriate analysis methods to the dataset (EDA + summary metrics) and relates to business need. | Uses a structured approach (EDA → insights → recommendations), shows strong reasoning, and demonstrates repeatable methodology. | Analysis is superficial or misaligned; conclusions not supported by evidence. |
| **K15 – Engineering principles for designing/developing new data products** | Shows basic engineering discipline: clear artefact goal, steps, and evidence (screenshots) of build process. | Strong engineering approach: modularity, reproducibility, versioning/controls, clear assumptions, test/validation thinking. | Artefact process unclear; poor engineering practice; cannot follow how the result was produced. |
| **K20 – Collect, store, analyse, and visualise data** | Describes data collection and storage at a basic level; produces a dashboard/report with ≥2 visualisations and explains what they show. | Strong end-to-end data story: collection → prep → modelling/metrics → visuals → business decision. Visuals are well-chosen and explained. | Missing visuals / fewer than 2; unclear collection/storage; visuals not explained or irrelevant. |
| **K22 – Relationship between mathematical principles and AI/DS techniques** | Demonstrates correct use/interpretation of core statistics (distributions, sampling, p-values, confidence) in the context of analysis. | Shows strong statistical reasoning: assumptions, effect size, uncertainty, limitations, and correct interpretation tied to business decision-making. | Misinterprets statistics (e.g., p-value meaning), ignores assumptions, or draws invalid conclusions. |
| **K24 – Sources of error and bias (dataset + methods)** | Identifies at least 2 plausible sources of error/bias and proposes mitigations; applies basic cleaning/prep. | Provides structured bias/error analysis (missingness, sampling bias, measurement bias, confounding) and shows evidence of mitigation and impact. | Little/no bias discussion; incorrect handling (e.g., leakage), or no cleaning where required. |
| **K26 – Scientific method, experiment design, hypothesis testing** | States null and alternative hypotheses, chooses an appropriate test, runs it, and interprets results. | Justifies test choice vs alternatives, checks assumptions, includes effect size/CI, and links conclusion to organisational strategy. | Hypotheses missing/incorrect; wrong test or invalid interpretation; no decision or business implication. |
| **K27 – Engineering principles to create instruments/apps for data collection** | Describes plausible data collection mechanisms (systems, forms, logs, sensors, pipelines) and how data quality would be controlled. | Details instrumentation design: event schema, validation, monitoring, governance, and how it supports scaling and future ML. | Data collection is vague/unrealistic; no thought to instrumentation or quality controls. |
| **S1 – Use applied research & modelling to design/refine storage architectures** | Proposes a coherent storage architecture and explains how it supports secure/stable/scalable data products. | Strong architecture rationale with applied research references, security controls, scaling approach, and clear mapping to use cases. | Architecture is missing or inconsistent; security/stability/scalability not addressed. |
| **S9 – Manipulate, analyse, and visualise complex datasets** | Demonstrates data manipulation and analysis steps (e.g., filtering, aggregations) with evidence (screenshots) and sensible visuals. | Efficient and accurate transformation pipeline; strong explanation; visuals reveal meaningful patterns and support decisions. | No evidence of manipulation/analysis; results are unreliable or unclear. |
| **S10 – Select datasets and methodologies appropriate to business problem** | Dataset is relevant to organisation/domain; methodology matches question. | Strong justification of dataset and methods; acknowledges constraints and explains why other options were rejected. | Dataset irrelevant/unrealistic; methodology mismatched; weak rationale. |
| **S13 – Identify appropriate resources/architectures for computational problem** | Chooses reasonable tools/platforms (Power BI/Python/SQL; warehouse/lake) and explains why at a basic level. | Strong resource selection with cost/performance/security considerations and realistic operational plan. | Tools/architecture choice unjustified or inappropriate. |
| **S17 – Implement data curation and data quality controls** | Includes basic quality controls (schema checks, duplicates, missing values) and documents cleaning steps. | Strong data quality approach: validation rules, monitoring, data dictionary, lineage, and repeatability. | No curation/quality controls; data quality issues ignored. |
| **S18 – Develop tools that visualise data systems/structures for monitoring/performance** | Includes an infrastructure diagram and explains how monitoring/performance could be observed (at a basic level). | Adds meaningful monitoring view: data pipeline health, latency, freshness, quality KPIs, access audit; diagram ties to operations. | Diagram missing or not explained; no monitoring/performance thinking. |
| **S21 – Identify and quantify uncertainty in outputs** | Mentions uncertainty sources (sampling, measurement) and uses at least one way to quantify/express it (e.g., CI, variability, error bars). | Strong uncertainty treatment: effect size + CI, practical significance, sensitivity checks, and limitations. | Uncertainty ignored; overconfident claims; no quantification where needed. |
| **S22 – Apply scientific methods through EDA + hypothesis testing for business decisions** | Shows EDA + hypothesis test and ties outcome to a business decision or strategy implication. | Strong decision framing: confirms/contradicts expectations, quantifies impact, and proposes next experiments/actions. | No EDA and/or hypothesis testing, or no business decision link. |
| **S26 – Select/apply appropriate AI/DS techniques to solve complex business problems** | Applies suitable DS techniques for the task (visual analytics + hypothesis testing) and explains rationale. | Goes beyond basics appropriately (segmentation, forecasting, anomaly detection—if justified) without overcomplicating; evaluates properly. | Techniques are inappropriate, misapplied, or create misleading conclusions. |
| **B3 – Integrity: ethical/legal/regulatory compliance; protect personal data** | Shows GDPR-aware handling: anonymisation/synthetic data rationale, minimal personal data, and ethical considerations. | Strong compliance-by-design: retention, access controls, lawful basis thinking, and ethical risk mitigations (incl. future ML concerns). | GDPR/ethics ignored or mishandled (e.g., personal data exposed, no anonymisation rationale). |
| **B7 – Shares best practice in org/community (AI & DS)** | Reflects on learning and states at least one way to share best practice (documentation, show-and-tell, template). | Concrete dissemination plan: reusable assets (dashboard standards, QA checks), stakeholder enablement, community contribution. | No meaningful reflection or sharing; vague statements only. |
"""


# =============================================================================
# MODULE DEFINITIONS
# =============================================================================

AVAILABLE_MODULES = {
    'DSP': {
        'name': 'DSP - Data Science Principles',
        'description': 'Data Science fundamentals: EDA, hypothesis testing, visualisation, and statistical reasoning',
        'rubric': DEFAULT_DSP_RUBRIC,
        'ksb_count': 19,
        'ksbs': ['K2', 'K5', 'K15', 'K20', 'K22', 'K24', 'K26', 'K27',
                 'S1', 'S9', 'S10', 'S13', 'S17', 'S18', 'S21', 'S22', 'S26',
                 'B3', 'B7']
    },
    'MLCC': {
        'name': 'MLCC - Machine Learning on Cloud Computing',
        'description': 'End-to-end ML system on public cloud with performance benchmarking',
        'rubric': DEFAULT_MLCC_RUBRIC,
        'ksb_count': 11,
        'ksbs': ['K1', 'K2', 'K16', 'K18', 'K19', 'K25', 'S15', 'S16', 'S19', 'S23', 'B5']
    },
    'AIDI': {
        'name': 'AIDI - AI & Digital Innovation',
        'description': 'AI/Data Products module focusing on business value, ethics, and stakeholder management',
        'rubric': DEFAULT_AIDI_RUBRIC,
        'ksb_count': 19,
        'ksbs': ['K1', 'K4', 'K5', 'K6', 'K8', 'K9', 'K11', 'K12', 'K21', 'K24', 'K29', 
                 'S3', 'S5', 'S6', 'S25', 'S26', 'B3', 'B4', 'B8']
    }
}


def get_module_criteria(module_code: str) -> List[KSBCriterion]:
    """
    Get KSB criteria for a specific module.
    
    Args:
        module_code: 'DSP', 'MLCC', or 'AIDI'
        
    Returns:
        List of KSBCriterion for the module
    """
    if module_code not in AVAILABLE_MODULES:
        raise ValueError(f"Unknown module: {module_code}. Available: {list(AVAILABLE_MODULES.keys())}")
    
    parser = KSBRubricParser()
    rubric = AVAILABLE_MODULES[module_code]['rubric']
    return parser.parse_markdown_table(rubric)


def get_available_modules() -> dict:
    """Get dictionary of available modules and their metadata."""
    return AVAILABLE_MODULES


def get_dsp_criteria() -> List[KSBCriterion]:
    """Get KSB criteria for DSP module."""
    return get_module_criteria('DSP')


def get_mlcc_criteria() -> List[KSBCriterion]:
    """Get KSB criteria for MLCC module."""
    return get_module_criteria('MLCC')


def get_aidi_criteria() -> List[KSBCriterion]:
    """Get KSB criteria for AIDI module."""
    return get_module_criteria('AIDI')