"""
Scoring Agent - Rubric Application and Weighted Scoring

Takes analysis results and:
- Applies KSB rubric criteria (Pass/Merit/Referral)
- Calculates weighted scores across categories (K/S/B)
- Determines final grades with confidence levels
"""
import json
import re
from typing import Dict, Any, List
import logging

from .core import BaseAgent, BaseTool, AgentContext, AgentRole, ToolResult

logger = logging.getLogger(__name__)


class RubricApplierTool(BaseTool):
    """Applies rubric criteria to evidence."""
    name = "apply_rubric"
    description = "Apply KSB rubric criteria to evidence and determine if Pass/Merit criteria are met"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, ksb_code: str, ksb_title: str,
                pass_criteria: str, merit_criteria: str, referral_criteria: str,
                evidence_summary: str, section_scores: Dict[str, float] = None) -> ToolResult:
        
        section_scores = section_scores or {}
        
        prompt = f"""Apply the KSB rubric to evaluate this evidence.

KSB: {ksb_code} - {ksb_title}

PASS CRITERIA:
{pass_criteria}

MERIT CRITERIA:
{merit_criteria}

REFERRAL INDICATORS:
{referral_criteria}

EVIDENCE:
{evidence_summary[:2000]}

SECTION QUALITY SCORES: {json.dumps(section_scores)}

Evaluate whether criteria are met. Respond with ONLY a JSON object:
{{
    "ksb_code": "{ksb_code}",
    "pass_assessment": {{
        "met": true|false,
        "confidence": "high|medium|low",
        "evidence_used": ["quotes used"]
    }},
    "merit_assessment": {{
        "met": true|false,
        "confidence": "high|medium|low",
        "evidence_used": []
    }},
    "evidence_strength": "strong|adequate|weak",
    "gaps": ["gaps identified"]
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=800)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, False, {"ksb_code": ksb_code}, "No JSON")
        except Exception as e:
            return ToolResult(self.name, False, {"ksb_code": ksb_code}, str(e))


class WeightCalculatorTool(BaseTool):
    """Calculates weighted scores."""
    name = "calculate_weights"
    description = "Calculate weighted scores for all KSBs based on category weights"
    
    def execute(self, context: AgentContext, ksb_scores: Dict[str, Dict[str, Any]],
                category_weights: Dict[str, float] = None) -> ToolResult:
        
        category_weights = category_weights or {"K": 0.4, "S": 0.4, "B": 0.2}
        grade_to_score = {"MERIT": 3, "PASS": 2, "REFERRAL": 1, "UNKNOWN": 0}
        
        weighted = {}
        category_totals = {"K": [], "S": [], "B": []}
        
        for ksb_code, data in ksb_scores.items():
            grade = data.get("grade", "UNKNOWN")
            base_score = grade_to_score.get(grade, 0)
            
            confidence = data.get("confidence", "medium").lower()
            conf_mult = {"high": 1.0, "medium": 0.9, "low": 0.7}.get(confidence, 0.8)
            
            weighted_score = base_score * conf_mult / 3  # Normalize to 0-1
            weighted[ksb_code] = {"raw": grade, "weighted": round(weighted_score, 3)}
            
            category = ksb_code[0] if ksb_code else "K"
            if category in category_totals:
                category_totals[category].append(weighted_score)
        
        # Calculate category averages
        category_avgs = {
            cat: round(sum(scores) / len(scores), 3) if scores else 0
            for cat, scores in category_totals.items()
        }
        
        # Overall weighted score
        overall = sum(category_avgs.get(cat, 0) * w for cat, w in category_weights.items())
        
        return ToolResult(self.name, True, {
            "ksb_weighted": weighted,
            "category_averages": category_avgs,
            "overall_weighted": round(overall, 3)
        })


class CriteriaCheckerTool(BaseTool):
    """Checks criteria patterns across all KSBs."""
    name = "check_criteria"
    description = "Check pass rates and critical criteria across all KSBs"
    
    def execute(self, context: AgentContext, all_grades: Dict[str, str],
                critical_ksbs: List[str] = None) -> ToolResult:
        
        critical_ksbs = critical_ksbs or []
        total = len(all_grades)
        passing = sum(1 for g in all_grades.values() if g in ["PASS", "MERIT"])
        merits = sum(1 for g in all_grades.values() if g == "MERIT")
        referrals = sum(1 for g in all_grades.values() if g == "REFERRAL")
        
        pass_rate = passing / total if total > 0 else 0
        
        critical_failures = [k for k in critical_ksbs if all_grades.get(k) == "REFERRAL"]
        
        # Category breakdown
        categories = {"K": {"pass": 0, "total": 0}, "S": {"pass": 0, "total": 0}, "B": {"pass": 0, "total": 0}}
        for ksb, grade in all_grades.items():
            cat = ksb[0] if ksb else "K"
            if cat in categories:
                categories[cat]["total"] += 1
                if grade in ["PASS", "MERIT"]:
                    categories[cat]["pass"] += 1
        
        # Determine recommendation
        if critical_failures:
            recommendation = "REFERRAL"
            reason = f"Critical failures: {', '.join(critical_failures)}"
        elif referrals > 0:
            recommendation = "REFERRAL"
            reason = f"{referrals} KSBs need resubmission"
        elif merits > total / 2:
            recommendation = "MERIT"
            reason = f"Majority ({merits}/{total}) achieved Merit"
        else:
            recommendation = "PASS"
            reason = f"All criteria met, pass rate {pass_rate:.0%}"
        
        return ToolResult(self.name, True, {
            "total": total,
            "passing": passing,
            "merits": merits,
            "referrals": referrals,
            "pass_rate": round(pass_rate, 3),
            "critical_failures": critical_failures,
            "category_breakdown": categories,
            "recommendation": recommendation,
            "reason": reason
        })


class BriefMapperTool(BaseTool):
    """Maps KSB scores to assignment brief tasks."""
    name = "map_to_brief"
    description = "Map KSB grades to assignment brief tasks"
    
    def execute(self, context: AgentContext, ksb_grades: Dict[str, str]) -> ToolResult:
        if not context.assignment_brief:
            return ToolResult(self.name, True, {"tasks": [], "note": "No brief available"})
        
        task_results = []
        for task in context.assignment_brief.get("tasks", []):
            task_ksbs = task.get("mapped_ksbs", [])
            task_grades = {k: ksb_grades.get(k, "UNKNOWN") for k in task_ksbs}
            
            passing = sum(1 for g in task_grades.values() if g in ["PASS", "MERIT"])
            total = len(task_grades)
            rate = passing / total if total > 0 else 0
            
            task_results.append({
                "task": task.get("task_title", f"Task {task.get('task_number', '?')}"),
                "ksbs": task_ksbs,
                "grades": task_grades,
                "completion": round(rate, 2),
                "status": "COMPLETE" if rate >= 0.8 else "PARTIAL" if rate >= 0.5 else "INCOMPLETE"
            })
        
        return ToolResult(self.name, True, {"tasks": task_results})


class ScoringAgent(BaseAgent):
    """Scoring Agent - Rubric application and weighted scoring."""
    
    def __init__(self, llm, verbose: bool = False):
        tools = [
            RubricApplierTool(llm),
            WeightCalculatorTool(),
            CriteriaCheckerTool(),
            BriefMapperTool()
        ]
        super().__init__(llm, AgentRole.SCORING, tools, verbose)
    
    def process(self, context: AgentContext) -> AgentContext:
        """Score all KSBs against rubric."""
        
        self._log_verbose("Starting scoring phase...")
        
        rubric_tool = self.tools["apply_rubric"]
        weight_tool = self.tools["calculate_weights"]
        check_tool = self.tools["check_criteria"]
        brief_tool = self.tools["map_to_brief"]
        
        # Step 1: Apply rubric to each KSB
        for ksb in context.ksb_criteria:
            ksb_code = ksb.get("code", "")
            self._log_verbose(f"Scoring {ksb_code}...")
            
            # Gather evidence from analysis
            evidence_parts = []
            section_scores = {}
            
            # From section analyses
            for analysis in context.section_analyses:
                ksb_evidence = analysis.get("evidence_found", {}).get(ksb_code, [])
                if ksb_evidence:
                    evidence_parts.extend(ksb_evidence[:3])
                
                if ksb_code in analysis.get("ksb_mappings", []):
                    section_id = analysis.get("section_id", "unknown")
                    clarity = analysis.get("clarity", {}).get("score", 0)
                    accuracy = analysis.get("accuracy", {}).get("score", 0)
                    section_scores[section_id] = (clarity + accuracy) / 2
            
            # From evidence map
            for ev in context.evidence_map.get(ksb_code, [])[:3]:
                if isinstance(ev, dict):
                    evidence_parts.append(ev.get("content", "")[:200])
            
            evidence_summary = "\n".join(f"- {e}" for e in evidence_parts[:8])
            if not evidence_summary:
                evidence_summary = "No direct evidence found."
            
            # Apply rubric
            self._log_tool_call("apply_rubric", {"ksb": ksb_code})
            result = rubric_tool.execute(
                context,
                ksb_code=ksb_code,
                ksb_title=ksb.get("title", ""),
                pass_criteria=ksb.get("pass_criteria", ""),
                merit_criteria=ksb.get("merit_criteria", ""),
                referral_criteria=ksb.get("referral_criteria", ""),
                evidence_summary=evidence_summary,
                section_scores=section_scores
            )
            self._log_tool_call("apply_rubric", result=result)
            
            if result.success:
                data = result.data
                pass_met = data.get("pass_assessment", {}).get("met", False)
                merit_met = data.get("merit_assessment", {}).get("met", False)
                evidence_strength = data.get("evidence_strength", "weak")
                
                # Determine grade
                if not pass_met:
                    grade = "REFERRAL"
                    confidence = "HIGH" if evidence_strength == "weak" else "MEDIUM"
                elif merit_met and evidence_strength != "weak":
                    grade = "MERIT"
                    confidence = "HIGH" if evidence_strength == "strong" else "MEDIUM"
                else:
                    grade = "PASS"
                    confidence = "HIGH" if pass_met else "MEDIUM"
                
                # Calculate weighted score
                avg_score = sum(section_scores.values()) / len(section_scores) if section_scores else 2.5
                weighted = avg_score / 5
                
                context.ksb_scores[ksb_code] = {
                    "ksb_title": ksb.get("title", ""),
                    "grade": grade,
                    "confidence": confidence,
                    "pass_met": pass_met,
                    "merit_met": merit_met,
                    "evidence_strength": evidence_strength,
                    "weighted_score": round(weighted, 3),
                    "gaps": data.get("gaps", []),
                    "rationale": f"Pass {'met' if pass_met else 'NOT met'}. Merit {'met' if merit_met else 'not met'}. Evidence: {evidence_strength}."
                }
            else:
                context.ksb_scores[ksb_code] = {
                    "ksb_title": ksb.get("title", ""),
                    "grade": "UNKNOWN",
                    "confidence": "LOW",
                    "pass_met": False,
                    "merit_met": False,
                    "evidence_strength": "weak",
                    "weighted_score": 0,
                    "gaps": ["Could not evaluate"],
                    "rationale": f"Evaluation failed: {result.error}"
                }
        
        # Step 2: Calculate weighted scores
        all_grades = {k: v.get("grade", "UNKNOWN") for k, v in context.ksb_scores.items()}
        weight_result = weight_tool.execute(context, context.ksb_scores)
        
        # Step 3: Check criteria patterns
        check_result = check_tool.execute(context, all_grades)
        
        # Step 4: Map to brief
        brief_result = brief_tool.execute(context, all_grades)
        
        context.overall_scores = {
            "weighted": weight_result.data if weight_result.success else {},
            "patterns": check_result.data if check_result.success else {},
            "brief_mapping": brief_result.data if brief_result.success else {}
        }
        
        logger.info(f"Scoring complete: {len(context.ksb_scores)} KSBs scored, "
                   f"recommendation: {check_result.data.get('recommendation', 'UNKNOWN')}")
        
        return context
