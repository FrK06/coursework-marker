"""
Scoring Agent - Rubric Application and Weighted Scoring

Takes analysis results and:
- Applies KSB rubric criteria (Pass/Merit/Referral)
- Calculates weighted scores across categories (K/S/B)
- Determines final grades with confidence levels
"""
import json
import re
import time
from typing import Dict, Any, List
import logging

from .core import BaseAgent, BaseTool, AgentContext, AgentRole, ToolResult
from ..validation.output_validator import OutputValidator
from ..prompts.ksb_templates import KSBPromptTemplates, extract_grade_from_evaluation
from ..utils.logger import create_logger, LogLevel
from config import ModelConfig

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

        # Get brief context if available - ONLY include tasks that map to this specific KSB
        brief_context = ""
        if context.assignment_brief:
            # Reconstruct AssignmentBrief object for better context formatting
            from ..brief.brief_parser import AssignmentBrief, TaskRequirement

            try:
                # Rebuild AssignmentBrief from dict
                brief_dict = context.assignment_brief
                tasks = [
                    TaskRequirement(
                        task_number=t.get('task_number', 0),
                        task_title=t.get('task_title', ''),
                        description=t.get('description', ''),
                        deliverables=t.get('deliverables', []),
                        mapped_ksbs=t.get('mapped_ksbs', []),
                        word_count=t.get('word_count'),
                        weighting=t.get('weighting')
                    )
                    for t in brief_dict.get('tasks', [])
                ]

                brief_obj = AssignmentBrief(
                    module_code=brief_dict.get('module_code', ''),
                    module_title=brief_dict.get('module_title', ''),
                    tasks=tasks,
                    overall_requirements=brief_dict.get('overall_requirements', ''),
                    submission_guidelines=brief_dict.get('submission_guidelines', ''),
                    ksb_task_mapping=brief_dict.get('ksb_task_mapping', {}),
                    raw_text=brief_dict.get('raw_text', '')
                )

                # Use the built-in method to get KSB-specific context
                brief_context = brief_obj.get_context_for_ksb(ksb_code)

                # Add explicit instruction if no tasks map to this KSB
                if not brief_obj.get_tasks_for_ksb(ksb_code):
                    brief_context = (
                        f"This KSB ({ksb_code}) should be assessed based on the overall report evidence, "
                        f"not tied to a specific assignment task.\n\n"
                        f"**IMPORTANT:** Do NOT penalize the student for not covering requirements from "
                        f"other tasks that are not mapped to this KSB."
                    )
                else:
                    # Add warning to ONLY assess against the shown tasks
                    brief_context = (
                        f"**⚠️ CRITICAL:** The tasks shown below are the ONLY tasks relevant to {ksb_code}. "
                        f"Do NOT assess this KSB against tasks that are not listed here.\n\n"
                        f"{brief_context}"
                    )
            except Exception as e:
                logger.warning(f"Failed to reconstruct AssignmentBrief: {e}, using fallback")
                # Fallback to original simple approach
                tasks = context.assignment_brief.get("tasks", [])
                for task in tasks:
                    mapped_ksbs = task.get("mapped_ksbs", [])
                    if ksb_code in mapped_ksbs:
                        brief_context += f"**{task.get('task_title', '')}**: {task.get('description', '')}\n"

                if not brief_context:
                    brief_context = (
                        f"This KSB ({ksb_code}) should be assessed based on the overall report evidence."
                    )

        # Use mature template with anti-hallucination guards
        prompt = KSBPromptTemplates.format_ksb_evaluation(
            ksb_code=ksb_code,
            ksb_title=ksb_title,
            pass_criteria=pass_criteria,
            merit_criteria=merit_criteria,
            referral_criteria=referral_criteria,
            evidence_text=evidence_summary[:3000],  # Increased from 2000
            brief_context=brief_context if brief_context else ""
        )

        try:
            # Get model-specific configuration
            model_config = ModelConfig.get_model_config(self.llm.model)

            # Adjust system prompt based on model strictness tendencies
            system_prompt = KSBPromptTemplates.get_system_prompt()
            if model_config.get("strictness_adjustment") == "lenient":
                # Add clarification for models that are too strict
                system_prompt += """

═══════════════════════════════════════════════════════════════════════════════
⚠️  GRADING CALIBRATION NOTE
═══════════════════════════════════════════════════════════════════════════════

When evidence is PARTIAL or IMPLICIT:
- If the student demonstrates understanding even if not perfectly articulated → PASS
- Only use REFERRAL when evidence is clearly ABSENT or fundamentally wrong
- "Evidence is weak but present" → PASS with LOW confidence, NOT REFERRAL

The goal is fair assessment, not finding reasons to fail students.
"""

            # Use model-specific temperature and max_tokens
            response = self.llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=model_config.get("temperature", 0.1),
                max_tokens=model_config.get("max_tokens", 1500)
            )

            # Use robust grade extraction with 4 fallback methods
            extracted = extract_grade_from_evaluation(response)

            # Map to existing ToolResult format
            data = {
                "ksb_code": ksb_code,
                "pass_assessment": {
                    "met": extracted.get('pass_criteria_met', False),
                    "confidence": extracted.get('confidence', 'medium').lower(),
                    "evidence_used": extracted.get('brief_requirements_met', [])
                },
                "merit_assessment": {
                    "met": extracted.get('merit_criteria_met', False),
                    "confidence": extracted.get('confidence', 'medium').lower(),
                    "evidence_used": []
                },
                "evidence_strength": "strong" if extracted.get('confidence') == 'HIGH' else "adequate" if extracted.get('confidence') == 'MEDIUM' else "weak",
                "gaps": extracted.get('brief_requirements_missing', []),
                "_raw_response": response,  # For validation
                "_extraction_method": extracted.get('method', 'unknown'),
                "_possible_hallucination": extracted.get('possible_hallucination', False)
            }

            # Override with parsed JSON if available
            if extracted.get('raw_json'):
                raw_json = extracted['raw_json']
                data['pass_assessment']['met'] = raw_json.get('pass_criteria_met', data['pass_assessment']['met'])
                data['merit_assessment']['met'] = raw_json.get('merit_criteria_met', data['merit_assessment']['met'])
                if 'main_gap' in raw_json and raw_json['main_gap']:
                    data['gaps'].append(raw_json['main_gap'])

            return ToolResult(self.name, True, data)

        except Exception as e:
            logger.error(f"RubricApplierTool failed for {ksb_code}: {e}")
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

    def __init__(self, llm, module_code: str = "MLCC", verbose: bool = False):
        tools = [
            RubricApplierTool(llm),
            WeightCalculatorTool(),
            CriteriaCheckerTool(),
            BriefMapperTool()
        ]
        super().__init__(llm, AgentRole.SCORING, tools, verbose)
        self.validator = OutputValidator(module_code)
        self.module_code = module_code

        # Initialize enhanced logger
        from config import LOG_LEVEL
        log_level = LogLevel.VERBOSE if verbose else LogLevel(LOG_LEVEL)
        self.enhanced_logger = create_logger("SCORING", log_level, verbose)
    
    def process(self, context: AgentContext) -> AgentContext:
        """Score all KSBs against rubric."""

        self.enhanced_logger.phase("Starting scoring phase...")
        scoring_start = time.time()

        rubric_tool = self.tools["apply_rubric"]
        weight_tool = self.tools["calculate_weights"]

        grade_counts = {"MERIT": 0, "PASS": 0, "REFERRAL": 0}
        low_confidence_grades = []
        check_tool = self.tools["check_criteria"]
        brief_tool = self.tools["map_to_brief"]
        
        # Step 1: Apply rubric to each KSB
        total_ksbs = len(context.ksb_criteria)
        for idx, ksb in enumerate(context.ksb_criteria, 1):
            ksb_code = ksb.get("code", "")
            self.enhanced_logger.progress(idx, total_ksbs, f"KSB {ksb_code}")
            
            # Gather evidence from analysis
            evidence_parts = []
            section_scores = {}

            # From section analyses (take more evidence - up to 5 per section)
            for analysis in context.section_analyses:
                ksb_evidence = analysis.get("evidence_found", {}).get(ksb_code, [])
                if ksb_evidence:
                    evidence_parts.extend(ksb_evidence[:5])  # Increased from 3 to 5

                if ksb_code in analysis.get("ksb_mappings", []):
                    section_id = analysis.get("section_id", "unknown")
                    clarity = analysis.get("clarity", {}).get("score", 0)
                    accuracy = analysis.get("accuracy", {}).get("score", 0)
                    section_scores[section_id] = (clarity + accuracy) / 2

            # From evidence map (take ALL up to 10, don't truncate content)
            for ev in context.evidence_map.get(ksb_code, [])[:10]:  # Increased from 3 to 10
                if isinstance(ev, dict):
                    content = ev.get("content", "")
                    # Don't truncate - include full content (up to 500 chars for readability)
                    evidence_parts.append(content[:500] if len(content) > 500 else content)
                elif isinstance(ev, str):
                    evidence_parts.append(ev[:500] if len(ev) > 500 else ev)

            # Build evidence summary with ALL evidence (up to 15 items total)
            evidence_summary = "\n".join(f"- {e}" for e in evidence_parts[:15])  # Increased from 8 to 15

            # DEBUG: Log evidence gathering
            self.enhanced_logger.debug(
                f"{ksb_code}: Gathered {len(evidence_parts)} evidence items "
                f"(total chars: {sum(len(str(e)) for e in evidence_parts)})"
            )

            if not evidence_summary:
                evidence_summary = "No direct evidence found."
            
            # Apply rubric (with validation and retry)
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

            # Validate the LLM response
            validation = None
            was_retried = False  # Track if we retried due to validation failure
            if result.success and "_raw_response" in result.data:
                raw_response = result.data.get("_raw_response", "")
                validation = self.validator.validate_evaluation(
                    raw_response, evidence_summary, ksb_code
                )

                self._log_verbose(
                    f"Validation for {ksb_code}: {validation.suggested_action} "
                    f"(confidence: {validation.confidence_score:.2f})"
                )

                # Handle validation results
                if validation.suggested_action == 'reject':
                    self._log_verbose(f"⚠️ Rejecting evaluation for {ksb_code}, retrying once...")
                    was_retried = True
                    # Retry once
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
                    # Re-validate
                    if result.success and "_raw_response" in result.data:
                        validation = self.validator.validate_evaluation(
                            result.data.get("_raw_response", ""), evidence_summary, ksb_code
                        )
                        self._log_verbose(f"Retry validation: {validation.suggested_action}")

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

                # Track grade statistics
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
                if confidence == "LOW":
                    low_confidence_grades.append(ksb_code)

                # === DEBUG LOGGING FOR REFERRAL GRADES ===
                if grade == "REFERRAL":
                    self.enhanced_logger.warning(f"\n{'='*80}\n🔍 REFERRAL DEBUG: {ksb_code} - {ksb.get('title', '')}\n{'='*80}", LogLevel.STANDARD)

                    # Show evidence chunks breakdown
                    self.enhanced_logger.warning(f"\n📊 Evidence Chunks Collected: {len(evidence_parts)}", LogLevel.STANDARD)
                    for idx, ev in enumerate(evidence_parts[:10], 1):  # Show first 10
                        preview = str(ev)[:200] + "..." if len(str(ev)) > 200 else str(ev)
                        self.enhanced_logger.warning(f"  [{idx}] {preview}", LogLevel.STANDARD)

                    # Show evidence summary sent to LLM
                    evidence_preview = evidence_summary[:500] + "..." if len(evidence_summary) > 500 else evidence_summary
                    self.enhanced_logger.warning(f"\n📝 Evidence Summary Sent to LLM (first 500 chars):\n{evidence_preview}\n", LogLevel.STANDARD)

                    # Show raw LLM response
                    raw_response = result.data.get("_raw_response", "")
                    response_preview = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
                    self.enhanced_logger.warning(f"\n🤖 LLM Response (first 500 chars):\n{response_preview}\n", LogLevel.STANDARD)

                    # Show validation warnings
                    if validation:
                        self.enhanced_logger.warning(f"\n⚠️ Validation Warnings ({len(validation.warnings)}):", LogLevel.STANDARD)
                        for idx, warning in enumerate(validation.warnings, 1):
                            self.enhanced_logger.warning(f"  {idx}. {warning}", LogLevel.STANDARD)
                        self.enhanced_logger.warning(f"\n🎯 Validation Action: {validation.suggested_action}", LogLevel.STANDARD)
                        self.enhanced_logger.warning(f"📊 Validation Confidence: {validation.confidence_score:.2f}\n", LogLevel.STANDARD)
                    else:
                        self.enhanced_logger.warning("\n⚠️ No validation performed\n", LogLevel.STANDARD)

                    # Show extraction details
                    self.enhanced_logger.warning(f"🔧 Grade Decision Details:", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"  - Pass criteria met: {pass_met}", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"  - Merit criteria met: {merit_met}", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"  - Evidence strength: {evidence_strength}", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"  - Evidence count: {len(evidence_parts)}", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"  - Confidence: {confidence}", LogLevel.STANDARD)
                    self.enhanced_logger.warning(f"\n{'='*80}\n", LogLevel.STANDARD)

                # Calculate weighted score
                avg_score = sum(section_scores.values()) / len(section_scores) if section_scores else 2.5
                weighted = avg_score / 5

                # Build audit trail for transparency/explainability
                audit_trail = {
                    'evidence': {
                        'chunks': [],
                        'total_chunks_retrieved': len(context.evidence_map.get(ksb_code, [])),
                        'chunks_after_filtering': len(evidence_parts),
                        'search_strategy': {
                            'query_variations': 0,  # Will be populated from evidence metadata
                            'mode': 'hybrid',  # Default assumption
                            'boilerplate_filtered': 0
                        }
                    },
                    'llm_evaluation': {
                        'raw_response': result.data.get('_raw_response', ''),
                        'prompt_length_chars': 0,  # Prompt not stored currently
                        'evidence_summary_length': len(evidence_summary),
                        'model': getattr(self.llm, 'model', 'unknown')
                    },
                    'validation': {
                        'action': validation.suggested_action if validation else 'not_validated',
                        'confidence': validation.confidence_score if validation else 0.0,
                        'warnings': validation.warnings if validation else [],
                        'retried': was_retried  # True if validation triggered retry
                    },
                    'grade_decision': {
                        'grade': grade,
                        'pass_criteria_met': pass_met,
                        'merit_criteria_met': merit_met,
                        'evidence_strength': evidence_strength,
                        'extraction_method': data.get('_extraction_method', 'unknown'),
                        'confidence': confidence
                    }
                }

                # Populate evidence chunks with actual text and metadata
                for ev in context.evidence_map.get(ksb_code, [])[:10]:
                    if isinstance(ev, dict):
                        chunk_entry = {
                            'text': ev.get('content', '')[:500],  # First 500 chars
                            'section_id': ev.get('section', 'unknown'),
                            'relevance_score': float(ev.get('relevance', 0.0)),
                            'search_method': ev.get('search_method', 'hybrid')
                        }
                        audit_trail['evidence']['chunks'].append(chunk_entry)
                    elif isinstance(ev, str):
                        # If evidence is just a string, create minimal entry
                        chunk_entry = {
                            'text': ev[:500],
                            'section_id': 'unknown',
                            'relevance_score': 0.0,
                            'search_method': 'unknown'
                        }
                        audit_trail['evidence']['chunks'].append(chunk_entry)

                # Calculate boilerplate filtering (difference between retrieved and used)
                audit_trail['evidence']['search_strategy']['boilerplate_filtered'] = \
                    audit_trail['evidence']['total_chunks_retrieved'] - audit_trail['evidence']['chunks_after_filtering']

                # Build KSB score entry
                ksb_score_entry = {
                    "ksb_title": ksb.get("title", ""),
                    "grade": grade,
                    "confidence": confidence,
                    "pass_met": pass_met,
                    "merit_met": merit_met,
                    "evidence_strength": evidence_strength,
                    "weighted_score": round(weighted, 3),
                    "gaps": data.get("gaps", []),
                    "rationale": f"Pass {'met' if pass_met else 'NOT met'}. Merit {'met' if merit_met else 'not met'}. Evidence: {evidence_strength}.",
                    "audit_trail": audit_trail  # Add full audit trail
                }

                # Add validation warnings if flagged for review
                if validation and validation.suggested_action == 'flag_for_review':
                    ksb_score_entry["flagged"] = True
                    ksb_score_entry["validation_warnings"] = validation.warnings
                    self.enhanced_logger.warning(f"{ksb_code} flagged for review: {len(validation.warnings)} warnings", LogLevel.STANDARD)

                # Log grade decision
                evidence_count = len(evidence_parts)
                self.enhanced_logger.grade_decision(
                    ksb_code,
                    grade,
                    confidence,
                    evidence_count,
                    pass_met
                )

                context.ksb_scores[ksb_code] = ksb_score_entry
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

        # Log scoring summary
        scoring_duration = time.time() - scoring_start
        self.enhanced_logger.metric("scoring_duration_sec", f"{scoring_duration:.1f}")
        self.enhanced_logger.metric("merits", grade_counts.get("MERIT", 0))
        self.enhanced_logger.metric("passes", grade_counts.get("PASS", 0))
        self.enhanced_logger.metric("referrals", grade_counts.get("REFERRAL", 0))
        self.enhanced_logger.metric("low_confidence_count", len(low_confidence_grades))

        if low_confidence_grades:
            self.enhanced_logger.warning(
                f"Low-confidence grades for: {', '.join(low_confidence_grades)}",
                LogLevel.STANDARD
            )

        self.enhanced_logger.success(
            f"Scoring complete: {grade_counts.get('MERIT', 0)} MERIT, "
            f"{grade_counts.get('PASS', 0)} PASS, {grade_counts.get('REFERRAL', 0)} REFERRAL "
            f"({len(low_confidence_grades)} low-confidence)"
        )

        logger.info(f"Scoring complete: {len(context.ksb_scores)} KSBs scored, "
                   f"recommendation: {check_result.data.get('recommendation', 'UNKNOWN')}")
        
        return context
