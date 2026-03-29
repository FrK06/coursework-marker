"""
Multi-Agent Coursework Marker - Core Architecture

Three-agent system for KSB assessment:
1. Analysis Agent - Multimodal document analysis
2. Scoring Agent - Rubric application and weighted scoring  
3. Feedback Agent - Personalized feedback generation
"""
import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import json
import re

# Type hints only - avoids circular imports
if TYPE_CHECKING:
    from .analysis_agent import AnalysisAgent
    from .scoring_agent import ScoringAgent
    from .feedback_agent import FeedbackAgent

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    ANALYSIS = "analysis"
    SCORING = "scoring"
    FEEDBACK = "feedback"


@dataclass
class AgentContext:
    """Shared context passed between agents."""
    # Input data
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    ksb_criteria: List[Dict[str, Any]] = field(default_factory=list)
    assignment_brief: Optional[Dict[str, Any]] = None
    
    # Analysis results (populated by Analysis Agent)
    section_analyses: List[Dict[str, Any]] = field(default_factory=list)
    image_analyses: List[Dict[str, Any]] = field(default_factory=list)
    table_analyses: List[Dict[str, Any]] = field(default_factory=list)
    evidence_map: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    evidence_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Stores query_variations, search_strategy, etc.
    content_quality: Dict[str, Any] = field(default_factory=dict)  # Content relevance assessment
    
    # Scoring results (populated by Scoring Agent)
    ksb_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_scores: Dict[str, Any] = field(default_factory=dict)
    
    # Feedback results (populated by Feedback Agent)
    ksb_feedback: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_feedback: str = ""


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class BaseTool:
    """Base class for agent tools."""
    name: str = "base_tool"
    description: str = "Base tool"

    def execute(self, context: AgentContext, **kwargs) -> ToolResult:
        raise NotImplementedError


class BaseAgent:
    """Base class for all agents."""
    
    # Class-level callback for UI updates (set by UI when verbose mode enabled)
    verbose_callback = None
    
    def __init__(self, llm, role: AgentRole, tools: List[BaseTool], verbose: bool = False):
        self.llm = llm
        self.role = role
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose
    
    def _log_verbose(self, message: str, data: Any = None):
        """Log verbose output if enabled."""
        if self.verbose:
            logger.info(f"[{self.role.value.upper()}] {message}")
            if BaseAgent.verbose_callback:
                BaseAgent.verbose_callback(self.role.value, message, data)
    
    def _log_tool_call(self, tool_name: str, inputs: Dict = None, result: 'ToolResult' = None):
        """Log tool calls if verbose."""
        if self.verbose:
            if inputs:
                self._log_verbose(f"🔧 {tool_name}({list(inputs.keys()) if inputs else ''})")
            if result:
                status = "✓" if result.success else "✗"
                self._log_verbose(f"   {status} {tool_name} → {list(result.data.keys())[:3]}...")
    
    def _call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0.2, images: List[str] = None) -> str:
        """Call the LLM with optional images."""
        if self.verbose:
            self._log_verbose(f"🤖 LLM call ({len(prompt)} chars, temp={temperature})")
        if images:
            return self.llm.generate(prompt, system_prompt=system_prompt, temperature=temperature, images=images)
        return self.llm.generate(prompt, system_prompt=system_prompt, temperature=temperature)
    
    def _parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    @abstractmethod
    def process(self, context: AgentContext) -> AgentContext:
        """Process the context and return updated context."""
        raise NotImplementedError


class AgentOrchestrator:
    """Orchestrates the three-agent pipeline."""
    
    def __init__(
        self,
        analysis_agent: 'AnalysisAgent',
        scoring_agent: 'ScoringAgent', 
        feedback_agent: 'FeedbackAgent',
        verbose: bool = False
    ):
        self.analysis_agent = analysis_agent
        self.scoring_agent = scoring_agent
        self.feedback_agent = feedback_agent
        self.verbose = verbose
    
    def process(
        self,
        chunks: List[Dict[str, Any]],
        ksb_criteria: List[Dict[str, Any]],
        assignment_brief: Optional[Dict[str, Any]] = None,
        images: List[Any] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Run the full pipeline with error handling and graceful degradation."""
        import time

        errors = []
        warnings = []

        # Initialize context
        context = AgentContext(
            chunks=chunks,
            images=images or [],
            ksb_criteria=ksb_criteria,
            assignment_brief=assignment_brief
        )

        # Phase 1: Analysis Agent (CRITICAL - if this fails, can't continue)
        try:
            if progress_callback:
                progress_callback("Analysis Agent: Processing sections...", 0.1)
            logger.info("Phase 1: Analysis Agent")

            start_time = time.time()
            context = self.analysis_agent.process(context)
            duration = time.time() - start_time

            if duration > 600:  # 10 minutes
                warnings.append(f"Analysis agent took {duration/60:.1f} minutes (unusually long)")

            logger.info(f"Analysis complete in {duration:.1f}s")

        except Exception as e:
            logger.exception("Analysis Agent failed")
            errors.append(f"Analysis Agent failed: {str(e)}")
            return {
                "analysis_results": [],
                "scoring_results": [],
                "feedback_results": [],
                "overall_summary": {
                    "total_ksbs": 0,
                    "merit_count": 0,
                    "pass_count": 0,
                    "referral_count": 0,
                    "overall_recommendation": "ERROR",
                    "key_strengths": [],
                    "priority_improvements": []
                },
                "overall_feedback": "",
                "errors": errors,
                "warnings": warnings,
                "status": "FAILED_ANALYSIS"
            }

        # Phase 2: Scoring Agent (IMPORTANT - try to continue even if this fails)
        try:
            if progress_callback:
                progress_callback("Scoring Agent: Applying rubric...", 0.4)
            logger.info("Phase 2: Scoring Agent")

            start_time = time.time()
            context = self.scoring_agent.process(context)
            duration = time.time() - start_time

            if duration > 600:  # 10 minutes
                warnings.append(f"Scoring agent took {duration/60:.1f} minutes (unusually long)")

            logger.info(f"Scoring complete in {duration:.1f}s")

        except Exception as e:
            logger.exception("Scoring Agent failed")
            errors.append(f"Scoring Agent failed: {str(e)}")
            # Return analysis results at least
            return {
                "analysis_results": context.section_analyses,
                "scoring_results": [],
                "feedback_results": [],
                "overall_summary": {
                    "total_ksbs": len(ksb_criteria),
                    "merit_count": 0,
                    "pass_count": 0,
                    "referral_count": 0,
                    "overall_recommendation": "ERROR",
                    "key_strengths": [],
                    "priority_improvements": []
                },
                "overall_feedback": "Scoring failed - only analysis available",
                "errors": errors,
                "warnings": warnings,
                "status": "FAILED_SCORING"
            }

        # Phase 3: Feedback Agent (NICE TO HAVE - return scores even if this fails)
        try:
            if progress_callback:
                progress_callback("Feedback Agent: Generating feedback...", 0.7)
            logger.info("Phase 3: Feedback Agent")

            start_time = time.time()
            context = self.feedback_agent.process(context)
            duration = time.time() - start_time

            if duration > 600:  # 10 minutes
                warnings.append(f"Feedback agent took {duration/60:.1f} minutes (unusually long)")

            logger.info(f"Feedback complete in {duration:.1f}s")

        except Exception as e:
            logger.exception("Feedback Agent failed")
            errors.append(f"Feedback Agent failed: {str(e)}")
            warnings.append("Feedback generation failed - grades and scores available")

        if progress_callback:
            progress_callback("Complete!", 1.0)

        # Compile results (always returns valid dict even if some phases failed)
        results = self._compile_results(context)
        results["errors"] = errors
        results["warnings"] = warnings
        results["status"] = "COMPLETE" if not errors else "PARTIAL"

        return results
    
    def _compile_results(self, context: AgentContext) -> Dict[str, Any]:
        """Compile final results from context."""
        
        # Build scoring results list
        scoring_results = []
        for ksb_code, score_data in context.ksb_scores.items():
            scoring_results.append({
                "ksb_code": ksb_code,
                "ksb_title": score_data.get("ksb_title", ""),
                "grade": score_data.get("grade", "UNKNOWN"),
                "confidence": score_data.get("confidence", "LOW"),
                "pass_criteria_met": score_data.get("pass_met", False),
                "merit_criteria_met": score_data.get("merit_met", False),
                "weighted_score": score_data.get("weighted_score", 0),
                "evidence_strength": score_data.get("evidence_strength", "weak"),
                "gaps_identified": score_data.get("gaps", []),
                "rationale": score_data.get("rationale", ""),
                "audit_trail": score_data.get("audit_trail", {})  # Include full audit trail
            })
        
        # Build feedback results list
        feedback_results = []
        for ksb_code, fb_data in context.ksb_feedback.items():
            feedback_results.append({
                "ksb_code": ksb_code,
                "strengths": fb_data.get("strengths", []),
                "improvements": fb_data.get("improvements", []),
                "formatted_feedback": fb_data.get("formatted", "")
            })
        
        # Calculate summary
        grades = [r["grade"] for r in scoring_results]
        merit_count = grades.count("MERIT")
        pass_count = grades.count("PASS")
        referral_count = grades.count("REFERRAL")
        
        if referral_count > 0:
            overall = "REFERRAL"
        elif merit_count > len(grades) / 2:
            overall = "MERIT"
        else:
            overall = "PASS"
        
        return {
            "analysis_results": context.section_analyses,
            "scoring_results": scoring_results,
            "feedback_results": feedback_results,
            "overall_summary": {
                "total_ksbs": len(scoring_results),
                "merit_count": merit_count,
                "pass_count": pass_count,
                "referral_count": referral_count,
                "overall_recommendation": overall,
                "key_strengths": self._extract_strengths(feedback_results),
                "priority_improvements": self._extract_improvements(feedback_results),
                "content_warnings": context.overall_scores.get("content_warnings", [])
            },
            "content_quality": context.content_quality,
            "overall_feedback": context.overall_feedback
        }
    
    def _extract_strengths(self, feedback_results: List[Dict]) -> List[str]:
        strengths = []
        for fb in feedback_results:
            for s in fb.get("strengths", [])[:2]:
                if isinstance(s, dict):
                    strengths.append(s.get("strength", str(s)))
                else:
                    strengths.append(str(s))
        return strengths[:5]
    
    def _extract_improvements(self, feedback_results: List[Dict]) -> List[str]:
        improvements = []
        for fb in feedback_results:
            for i in fb.get("improvements", [])[:2]:
                if isinstance(i, dict):
                    improvements.append(i.get("suggestion", str(i)))
                else:
                    improvements.append(str(i))
        return improvements[:5]
