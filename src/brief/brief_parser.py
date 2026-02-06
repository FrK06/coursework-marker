"""
Assignment Brief Parser - Extracts task requirements from module assignment briefs.

This module parses assignment briefs to provide task-level context for KSB evaluation.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskRequirement:
    """A single task from the assignment brief."""
    task_number: int
    task_title: str
    description: str
    deliverables: List[str]
    mapped_ksbs: List[str]  # KSBs this task addresses
    word_count: Optional[int] = None
    weighting: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_number': self.task_number,
            'task_title': self.task_title,
            'description': self.description,
            'deliverables': self.deliverables,
            'mapped_ksbs': self.mapped_ksbs,
            'word_count': self.word_count,
            'weighting': self.weighting
        }


@dataclass
class AssignmentBrief:
    """Complete parsed assignment brief."""
    module_code: str
    module_title: str
    tasks: List[TaskRequirement]
    overall_requirements: str
    submission_guidelines: str
    ksb_task_mapping: Dict[str, List[int]]  # KSB code -> list of task numbers
    raw_text: str
    
    def get_tasks_for_ksb(self, ksb_code: str) -> List[TaskRequirement]:
        """Get all tasks that map to a specific KSB."""
        task_numbers = self.ksb_task_mapping.get(ksb_code, [])
        return [t for t in self.tasks if t.task_number in task_numbers]
    
    def get_context_for_ksb(self, ksb_code: str) -> str:
        """Generate context string for a specific KSB evaluation."""
        tasks = self.get_tasks_for_ksb(ksb_code)
        
        if not tasks:
            # KSB might apply across all tasks
            return f"This KSB ({ksb_code}) should be demonstrated across the submission."
        
        lines = [f"## Assignment Tasks for {ksb_code}:", ""]
        
        for task in tasks:
            lines.append(f"### Task {task.task_number}: {task.task_title}")
            lines.append(task.description)
            
            if task.deliverables:
                lines.append("\n**Expected Deliverables:**")
                for d in task.deliverables:
                    lines.append(f"- {d}")
            
            if task.word_count:
                lines.append(f"\n*Word count guidance: {task.word_count}*")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'module_code': self.module_code,
            'module_title': self.module_title,
            'tasks': [t.to_dict() for t in self.tasks],
            'overall_requirements': self.overall_requirements,
            'submission_guidelines': self.submission_guidelines,
            'ksb_task_mapping': self.ksb_task_mapping
        }


class BriefParser:
    """
    Parses assignment brief documents to extract structured task information.
    
    Supports common brief formats:
    - Task 1 / Task 2 structure
    - Part A / Part B structure
    - Numbered sections with KSB mappings
    """
    
    # Patterns to detect task sections
    TASK_PATTERNS = [
        r'^(?:Task|TASK)\s*(\d+)[:\s\-–]*(.*)$',
        r'^(?:Part|PART)\s*([A-Za-z\d]+)[:\s\-–]*(.*)$',
        r'^(\d+)\.\s*(?:Task|Activity)[:\s\-–]*(.*)$',
        r'^(?:Section|SECTION)\s*(\d+)[:\s\-–]*(.*)$',
    ]
    
    # Patterns to detect KSB references
    KSB_PATTERN = re.compile(r'\b([KSB]\d{1,2})\b')
    
    # Patterns to detect deliverables
    DELIVERABLE_PATTERNS = [
        r'(?:deliverable|output|produce|submit|create|provide)[s]?[:\s]+([^.]+)',
        r'(?:you should|you must|you need to)[:\s]+([^.]+)',
    ]
    
    def __init__(self):
        self.task_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                             for p in self.TASK_PATTERNS]
    
    def parse(self, text: str, module_code: str = "UNKNOWN") -> AssignmentBrief:
        """
        Parse assignment brief text into structured format.
        
        Args:
            text: Raw text from the assignment brief document
            module_code: The module code (DSP, MLCC, AIDI)
            
        Returns:
            AssignmentBrief with extracted tasks and KSB mappings
        """
        # Extract module title from first few lines
        module_title = self._extract_module_title(text, module_code)
        
        # Split into tasks
        tasks = self._extract_tasks(text)
        
        # Build KSB to task mapping
        ksb_task_mapping = self._build_ksb_mapping(tasks)
        
        # Extract overall requirements
        overall_requirements = self._extract_overall_requirements(text)
        
        # Extract submission guidelines
        submission_guidelines = self._extract_submission_guidelines(text)
        
        logger.info(f"Parsed brief: {len(tasks)} tasks, {len(ksb_task_mapping)} KSB mappings")
        
        return AssignmentBrief(
            module_code=module_code,
            module_title=module_title,
            tasks=tasks,
            overall_requirements=overall_requirements,
            submission_guidelines=submission_guidelines,
            ksb_task_mapping=ksb_task_mapping,
            raw_text=text
        )
    
    def _extract_module_title(self, text: str, module_code: str) -> str:
        """Extract the module title from the brief."""
        lines = text.split('\n')[:20]
        
        for line in lines:
            line = line.strip()
            if module_code.upper() in line.upper():
                return line
            if any(kw in line.lower() for kw in ['module', 'assignment', 'coursework']):
                return line
        
        return f"{module_code} Assignment"
    
    def _extract_tasks(self, text: str) -> List[TaskRequirement]:
        """Extract individual tasks from the brief."""
        tasks = []
        
        # Find all task headers and their positions
        task_positions = []
        
        for pattern in self.task_patterns:
            for match in pattern.finditer(text):
                task_num = match.group(1)
                task_title = match.group(2).strip() if match.group(2) else ""
                
                # Convert letter to number if needed (Part A -> 1)
                if task_num.isalpha():
                    task_num = ord(task_num.upper()) - ord('A') + 1
                else:
                    task_num = int(task_num)
                
                task_positions.append({
                    'number': task_num,
                    'title': task_title,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Remove duplicates and sort by position
        seen_positions = set()
        unique_tasks = []
        for tp in sorted(task_positions, key=lambda x: x['start']):
            if tp['start'] not in seen_positions:
                seen_positions.add(tp['start'])
                unique_tasks.append(tp)
        
        # Extract content for each task
        for i, task_info in enumerate(unique_tasks):
            # Get text until next task or end
            start = task_info['end']
            end = unique_tasks[i + 1]['start'] if i + 1 < len(unique_tasks) else len(text)
            
            task_text = text[start:end].strip()
            
            # Extract KSBs mentioned in this task
            ksbs = list(set(self.KSB_PATTERN.findall(task_text)))
            
            # Extract deliverables
            deliverables = self._extract_deliverables(task_text)
            
            # Extract word count if mentioned
            word_count = self._extract_word_count(task_text)
            
            tasks.append(TaskRequirement(
                task_number=task_info['number'],
                task_title=task_info['title'] or f"Task {task_info['number']}",
                description=task_text[:2000],  # Limit description length
                deliverables=deliverables,
                mapped_ksbs=ksbs,
                word_count=word_count
            ))
        
        # If no tasks found, create a single task from the whole document
        if not tasks:
            ksbs = list(set(self.KSB_PATTERN.findall(text)))
            tasks.append(TaskRequirement(
                task_number=1,
                task_title="Main Assignment",
                description=text[:3000],
                deliverables=self._extract_deliverables(text),
                mapped_ksbs=ksbs
            ))
        
        return tasks
    
    def _extract_deliverables(self, text: str) -> List[str]:
        """Extract expected deliverables from task text."""
        deliverables = []
        
        # Look for bullet points after "deliverables" or similar keywords
        deliverable_section = re.search(
            r'(?:deliverable|output|submit|produce)[s]?[:\s]*\n((?:[\s]*[-•*]\s*[^\n]+\n?)+)',
            text, re.IGNORECASE
        )
        
        if deliverable_section:
            bullets = re.findall(r'[-•*]\s*([^\n]+)', deliverable_section.group(1))
            deliverables.extend([b.strip() for b in bullets if len(b.strip()) > 5])
        
        # Also look for "you should/must" patterns
        should_patterns = re.findall(
            r'(?:you should|you must|you need to|required to)\s+([^.]{10,100})\.',
            text, re.IGNORECASE
        )
        deliverables.extend([p.strip() for p in should_patterns])
        
        return deliverables[:10]  # Limit to 10 deliverables
    
    def _extract_word_count(self, text: str) -> Optional[int]:
        """Extract word count guidance if present."""
        match = re.search(r'(\d{3,5})\s*(?:words?|word count)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _build_ksb_mapping(self, tasks: List[TaskRequirement]) -> Dict[str, List[int]]:
        """Build a mapping from KSB codes to task numbers."""
        mapping = {}
        
        for task in tasks:
            for ksb in task.mapped_ksbs:
                if ksb not in mapping:
                    mapping[ksb] = []
                if task.task_number not in mapping[ksb]:
                    mapping[ksb].append(task.task_number)
        
        return mapping
    
    def _extract_overall_requirements(self, text: str) -> str:
        """Extract overall assignment requirements."""
        # Look for sections about overall requirements
        patterns = [
            r'(?:overall|general)\s*(?:requirements?|guidance)[:\s]*([^#]+?)(?=\n\s*(?:task|part|\d+\.))',
            r'(?:assessment|marking)\s*(?:criteria|requirements)[:\s]*([^#]+?)(?=\n\s*(?:task|part|\d+\.))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:1500]
        
        return ""
    
    def _extract_submission_guidelines(self, text: str) -> str:
        """Extract submission guidelines."""
        patterns = [
            r'(?:submission|submit)[:\s]*([^#]+?)(?=\n\s*(?:task|part|assessment|\Z))',
            r'(?:format|formatting)[:\s]*([^#]+?)(?=\n\s*(?:task|part|assessment|\Z))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:1000]
        
        return ""


# Pre-defined assignment brief templates for common modules
# These can be used if no brief is uploaded

MLCC_DEFAULT_BRIEF = """
# Machine Learning with Cloud Computing (MLCC) - Assignment Brief

## Overview
Design, implement, and benchmark a machine learning system on a public cloud platform.

## Task 1: Technical Feasibility Study
Produce a technical feasibility report that:
- Defines functional and non-functional requirements
- Selects appropriate cloud platform and services
- Addresses security, GDPR, and governance requirements
- Provides architecture diagrams

**Deliverables:**
- Requirements specification
- Cloud architecture design
- Security and compliance assessment

**KSBs addressed:** K1, K2, K16, S16

## Task 2: Proof of Concept Artefact
Build a working PoC demonstrating:
- Data pipeline implementation
- Model training with appropriate ML libraries
- Deployment/inference capability
- Reproducible experiments

**Deliverables:**
- Working cloud-based ML system
- Code repository with documentation
- Training logs and metrics

**KSBs addressed:** K18, K25, S15

## Task 3: Benchmarking
Compare CPU vs GPU training performance:
- Wall-clock time comparison
- Cost analysis
- Scalability assessment

**Deliverables:**
- Benchmarking methodology
- Results comparison table
- Cost-benefit analysis

**KSBs addressed:** K16, K19, S19

## Task 4: Reflection
Reflect on:
- Best practices and lessons learned
- CPD activities undertaken
- Dissemination of knowledge

**KSBs addressed:** S23, B5
"""

DSP_DEFAULT_BRIEF = """
# Data Science Principles (DSP) - Assignment Brief

## Overview
Apply data science principles to analyze a real-world dataset and deliver actionable insights.

## Task 1: Data Infrastructure & Storage
Design a data storage and processing architecture:
- Select appropriate storage solutions
- Define data quality controls
- Create infrastructure diagrams

**Deliverables:**
- Architecture documentation
- Data dictionary
- Quality control procedures

**KSBs addressed:** K2, K27, S1, S17

## Task 2: Exploratory Data Analysis
Perform comprehensive EDA:
- Data visualization (minimum 2 visualizations)
- Statistical summary
- Pattern identification

**Deliverables:**
- Dashboard/report with visualizations
- Statistical analysis findings

**KSBs addressed:** K5, K20, S9, S22

## Task 3: Hypothesis Testing
Apply scientific method:
- State hypotheses
- Select appropriate statistical tests
- Interpret results with uncertainty quantification

**Deliverables:**
- Hypothesis documentation
- Test results with confidence intervals
- Business recommendations

**KSBs addressed:** K22, K26, S21, S22

## Task 4: Compliance & Best Practice
Address:
- GDPR compliance
- Ethical considerations
- Best practice sharing

**KSBs addressed:** K24, B3, B7
"""

AIDI_DEFAULT_BRIEF = """
# AI & Digital Innovation (AIDI) - Assignment Brief

## Overview
Develop an AI/Data product concept that delivers business value while addressing ethical and stakeholder requirements.

## Task 1: Product Definition & Requirements
Define the AI/Data product:
- Business problem and objectives
- User requirements
- Success criteria and KPIs

**Deliverables:**
- Product specification
- Requirements document
- Stakeholder analysis

**KSBs addressed:** K1, K5, K6, S3, S5

## Task 2: Technical Implementation
Build a working artefact:
- Apply appropriate AI/DS techniques
- Implement with good engineering practices
- Include accessibility considerations

**Deliverables:**
- Working artefact/prototype
- Technical documentation
- Accessibility assessment

**KSBs addressed:** K4, K29, S25, S26

## Task 3: Ethics & Compliance
Address ethical and legal aspects:
- Legal/regulatory framework compliance
- Bias and error analysis
- Social impact assessment

**Deliverables:**
- Ethics assessment
- Compliance documentation
- Bias mitigation strategy

**KSBs addressed:** K8, K9, K12, K24, B3

## Task 4: Stakeholder Management & Direction
Provide guidance:
- Stakeholder communication plan
- Technical roadmap
- Innovation trends analysis

**KSBs addressed:** K11, K21, S5, S6, B4, B8
"""

DEFAULT_BRIEFS = {
    'MLCC': MLCC_DEFAULT_BRIEF,
    'DSP': DSP_DEFAULT_BRIEF,
    'AIDI': AIDI_DEFAULT_BRIEF
}


def get_default_brief(module_code: str) -> Optional[AssignmentBrief]:
    """Get the default assignment brief for a module."""
    if module_code not in DEFAULT_BRIEFS:
        return None
    
    parser = BriefParser()
    return parser.parse(DEFAULT_BRIEFS[module_code], module_code)


def parse_uploaded_brief(text: str, module_code: str) -> AssignmentBrief:
    """Parse an uploaded assignment brief."""
    parser = BriefParser()
    return parser.parse(text, module_code)
