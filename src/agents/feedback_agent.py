"""
Feedback Agent - Personalized Feedback Generation

Takes scores and analysis results to generate:
- Section-specific feedback
- Strengths with evidence
- Actionable improvement suggestions
- Encouraging, constructive tone
"""
import json
import re
from typing import Dict, Any, List
import logging

from .core import BaseAgent, BaseTool, AgentContext, AgentRole, ToolResult

logger = logging.getLogger(__name__)


FEEDBACK_TEMPLATES = {
    "strength_intro": {
        "MERIT": "Excellent work demonstrating strong understanding in this area.",
        "PASS": "Good work meeting the requirements for this criterion.",
        "REFERRAL": "There are some positive elements, though significant improvement is needed."
    },
    "improvement_intro": {
        "MERIT": "To further strengthen this already good work:",
        "PASS": "To achieve Merit level, consider:",
        "REFERRAL": "To meet the Pass requirements, you need to:"
    },
    "encouragement": {
        "MERIT": "Keep up this excellent standard of work!",
        "PASS": "You're on the right track - a bit more depth will help you reach Merit.",
        "REFERRAL": "Don't be discouraged - focused improvements will help you meet the criteria."
    }
}


class StrengthIdentifierTool(BaseTool):
    """Identifies strengths from assessment results."""
    name = "identify_strengths"
    description = "Identify and articulate strengths from assessment results"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, ksb_code: str, grade: str,
                section_analyses: List[Dict], evidence: List[str] = None) -> ToolResult:
        
        evidence = evidence or []
        
        # Compile strengths from analyses
        all_strengths = []
        high_scores = []
        
        for analysis in section_analyses:
            strengths = analysis.get("strengths", [])
            all_strengths.extend(strengths[:2])
            
            for criterion in ["clarity", "accuracy", "completeness"]:
                score_data = analysis.get(criterion, {})
                if isinstance(score_data, dict) and score_data.get("score", 0) >= 4:
                    high_scores.append(f"{criterion}: {score_data.get('score')}/5")
        
        prompt = f"""Based on these results, identify strengths for {ksb_code} (Grade: {grade}).

STRENGTHS FROM ANALYSIS: {json.dumps(all_strengths[:8])}
HIGH SCORES: {high_scores[:5]}
EVIDENCE: {evidence[:5]}

Identify 2-3 specific strengths. Respond with ONLY a JSON object:
{{
    "strengths": [
        {{"strength": "what was done well", "evidence": "specific example", "impact": "why valuable"}}
    ],
    "summary": "one sentence summary"
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=500)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, True, {"strengths": all_strengths[:3], "summary": ""})
        except Exception as e:
            return ToolResult(self.name, True, {"strengths": all_strengths[:3], "summary": "", "error": str(e)})


class GapAnalyzerTool(BaseTool):
    """Analyzes gaps and areas for improvement."""
    name = "analyze_gaps"
    description = "Analyze gaps identified in scoring"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, ksb_code: str, grade: str,
                gaps: List[str], pass_criteria: str, merit_criteria: str) -> ToolResult:
        
        target = "PASS" if grade == "REFERRAL" else "MERIT"
        target_criteria = pass_criteria if grade == "REFERRAL" else merit_criteria
        
        prompt = f"""Generate improvement suggestions for {ksb_code} (Current: {grade}, Target: {target}).

GAPS IDENTIFIED: {json.dumps(gaps)}
TARGET CRITERIA: {target_criteria[:500]}

Generate 2-3 actionable suggestions. Respond with ONLY a JSON object:
{{
    "improvements": [
        {{"area": "what needs work", "suggestion": "specific action", "example": "concrete example", "priority": "high|medium|low"}}
    ],
    "quick_wins": ["easy improvements"]
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=600)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, True, {"improvements": [], "quick_wins": []})
        except Exception as e:
            return ToolResult(self.name, True, {"improvements": [], "quick_wins": [], "error": str(e)})


class ImprovementSuggesterTool(BaseTool):
    """Generates specific improvement suggestions."""
    name = "suggest_improvements"
    description = "Generate specific, actionable improvement suggestions"
    
    def execute(self, context: AgentContext, gaps: List[str], grade: str) -> ToolResult:
        suggestions = []
        
        for gap in gaps[:3]:
            suggestions.append({
                "area": gap,
                "suggestion": f"Address: {gap}",
                "priority": "high" if grade == "REFERRAL" else "medium"
            })
        
        return ToolResult(self.name, True, {"suggestions": suggestions})


class FeedbackFormatterTool(BaseTool):
    """Formats feedback into professional output."""
    name = "format_feedback"
    description = "Format strengths and improvements into cohesive feedback"
    
    def execute(self, context: AgentContext, ksb_code: str, grade: str,
                strengths: List[Dict], improvements: List[Dict]) -> ToolResult:
        
        strength_intro = FEEDBACK_TEMPLATES["strength_intro"].get(grade, "")
        improvement_intro = FEEDBACK_TEMPLATES["improvement_intro"].get(grade, "")
        encouragement = FEEDBACK_TEMPLATES["encouragement"].get(grade, "")
        
        # Format strengths
        strengths_md = f"**What You Did Well**\n\n{strength_intro}\n\n"
        for s in strengths[:3]:
            if isinstance(s, dict):
                strengths_md += f"- **{s.get('strength', '')}**: {s.get('evidence', '')}\n"
            else:
                strengths_md += f"- {s}\n"
        
        # Format improvements
        improvements_md = f"\n**Areas for Development**\n\n{improvement_intro}\n\n"
        for i in improvements[:3]:
            if isinstance(i, dict):
                priority = i.get('priority', 'medium')
                icon = "🔴" if priority == "high" else "🟠" if priority == "medium" else "🟢"
                improvements_md += f"- {icon} **{i.get('area', '')}**: {i.get('suggestion', '')}\n"
                if i.get('example'):
                    improvements_md += f"  _Example: {i.get('example')}_\n"
            else:
                improvements_md += f"- {i}\n"
        
        formatted = f"""## {ksb_code} - Grade: {grade}

{strengths_md}
{improvements_md}

---
_{encouragement}_
"""
        
        return ToolResult(self.name, True, {"formatted": formatted})


class KSBFeedbackGeneratorTool(BaseTool):
    """Generates complete feedback for a single KSB."""
    name = "generate_ksb_feedback"
    description = "Generate complete feedback for one KSB"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, ksb_code: str, ksb_title: str,
                grade: str, score_data: Dict[str, Any]) -> ToolResult:
        
        gaps = score_data.get("gaps", [])
        evidence_strength = score_data.get("evidence_strength", "adequate")
        rationale = score_data.get("rationale", "")
        
        prompt = f"""Generate feedback for {ksb_code} - {ksb_title}.

GRADE: {grade}
EVIDENCE STRENGTH: {evidence_strength}
RATIONALE: {rationale}
GAPS: {gaps}

Respond with ONLY a JSON object:
{{
    "summary": "one line grade summary",
    "strengths": ["specific strengths"],
    "improvements": ["specific improvement actions"],
    "next_steps": ["concrete next steps"]
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=500)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, True, {
                "summary": f"{grade} - {rationale}",
                "strengths": [],
                "improvements": gaps,
                "next_steps": []
            })
        except Exception as e:
            return ToolResult(self.name, True, {
                "summary": f"{grade}",
                "strengths": [],
                "improvements": gaps,
                "next_steps": [],
                "error": str(e)
            })


class FeedbackAgent(BaseAgent):
    """Feedback Agent - Personalized feedback generation."""
    
    def __init__(self, llm, verbose: bool = False):
        tools = [
            StrengthIdentifierTool(llm),
            GapAnalyzerTool(llm),
            ImprovementSuggesterTool(),
            FeedbackFormatterTool(),
            KSBFeedbackGeneratorTool(llm)
        ]
        super().__init__(llm, AgentRole.FEEDBACK, tools, verbose)
    
    def process(self, context: AgentContext) -> AgentContext:
        """Generate personalized feedback for all KSBs."""
        
        self._log_verbose("Starting feedback generation...")
        
        strength_tool = self.tools["identify_strengths"]
        gap_tool = self.tools["analyze_gaps"]
        format_tool = self.tools["format_feedback"]
        ksb_gen = self.tools["generate_ksb_feedback"]
        
        all_strengths = []
        all_improvements = []
        
        for ksb_code, score_data in context.ksb_scores.items():
            grade = score_data.get("grade", "UNKNOWN")
            ksb_title = score_data.get("ksb_title", "")
            gaps = score_data.get("gaps", [])
            
            self._log_verbose(f"Generating feedback for {ksb_code} ({grade})...")
            
            # Find relevant KSB criteria
            ksb_info = next(
                (k for k in context.ksb_criteria if k.get("code") == ksb_code),
                {}
            )
            
            # Find relevant analyses
            relevant_analyses = [
                a for a in context.section_analyses
                if ksb_code in a.get("ksb_mappings", []) or 
                   ksb_code in a.get("evidence_found", {})
            ]
            
            # Get evidence
            evidence = []
            for ev in context.evidence_map.get(ksb_code, [])[:3]:
                if isinstance(ev, dict):
                    evidence.append(ev.get("content", "")[:150])
            
            # Step 1: Identify strengths
            self._log_tool_call("identify_strengths", {"ksb": ksb_code})
            strength_result = strength_tool.execute(
                context, ksb_code, grade, relevant_analyses, evidence
            )
            self._log_tool_call("identify_strengths", result=strength_result)
            strengths = strength_result.data.get("strengths", []) if strength_result.success else []
            
            # Step 2: Analyze gaps and generate improvements
            self._log_tool_call("analyze_gaps", {"ksb": ksb_code, "gaps": len(gaps)})
            gap_result = gap_tool.execute(
                context, ksb_code, grade, gaps,
                ksb_info.get("pass_criteria", ""),
                ksb_info.get("merit_criteria", "")
            )
            self._log_tool_call("analyze_gaps", result=gap_result)
            improvements = gap_result.data.get("improvements", []) if gap_result.success else []
            quick_wins = gap_result.data.get("quick_wins", []) if gap_result.success else []
            
            # Step 3: Format feedback
            self._log_tool_call("format_feedback", {"ksb": ksb_code})
            format_result = format_tool.execute(
                context, ksb_code, grade, strengths, improvements
            )
            self._log_tool_call("format_feedback", result=format_result)
            formatted = format_result.data.get("formatted", "") if format_result.success else ""
            
            # Store feedback
            context.ksb_feedback[ksb_code] = {
                "grade": grade,
                "strengths": strengths,
                "improvements": improvements,
                "quick_wins": quick_wins,
                "formatted": formatted
            }
            
            # Collect for overall summary
            all_strengths.extend(strengths[:2])
            all_improvements.extend(improvements[:2])
        
        # Generate overall feedback summary
        context.overall_feedback = self._generate_overall_feedback(context, all_strengths, all_improvements)
        
        logger.info(f"Feedback complete: {len(context.ksb_feedback)} KSBs")
        
        return context
    
    def _generate_overall_feedback(self, context: AgentContext, 
                                   strengths: List, improvements: List) -> str:
        """Generate overall summary feedback."""
        
        patterns = context.overall_scores.get("patterns", {})
        recommendation = patterns.get("recommendation", "UNKNOWN")
        merits = patterns.get("merits", 0)
        passes = patterns.get("passing", 0) - patterns.get("merits", 0)
        referrals = patterns.get("referrals", 0)
        total = patterns.get("total", 0)
        
        # Opening based on grade
        if recommendation == "MERIT":
            opening = "Congratulations on achieving an overall Merit grade! Your work demonstrates strong understanding across multiple areas."
        elif recommendation == "PASS":
            opening = "Well done on achieving an overall Pass grade. Your work meets the required standards with room for development."
        else:
            opening = "Your work shows potential but requires further development to meet all criteria."
        
        summary = f"""# Overall Assessment Feedback

{opening}

## Grade Summary

| Grade | Count |
|-------|-------|
| Merit | {merits} |
| Pass | {passes} |
| Referral | {referrals} |
| **Total** | **{total}** |

**Overall Recommendation: {recommendation}**

## Key Strengths

"""
        for s in strengths[:5]:
            if isinstance(s, dict):
                summary += f"- {s.get('strength', str(s))}\n"
            else:
                summary += f"- {s}\n"
        
        summary += "\n## Priority Improvements\n\n"
        for i in improvements[:5]:
            if isinstance(i, dict):
                summary += f"- **{i.get('area', '')}**: {i.get('suggestion', '')}\n"
            else:
                summary += f"- {i}\n"
        
        summary += "\n## Recommended Next Steps\n\n"
        if referrals > 0:
            summary += "1. Focus on addressing Referral KSBs first\n"
        summary += "2. Review feedback for each KSB and address specific gaps\n"
        summary += "3. Seek feedback from your mentor on priority areas\n"
        summary += "4. Build on strengths to demonstrate deeper understanding\n"
        
        return summary
