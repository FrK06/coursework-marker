"""
Analysis Agent - Multimodal Document Analysis

Breaks down report into sections and analyzes:
- Text content (clarity, accuracy, completeness, relevance)
- Charts and graphs
- Tables
- Images and diagrams

Maps findings to KSB criteria.
"""
import json
import re
from typing import Dict, Any, List, Optional
import logging

from .core import BaseAgent, BaseTool, AgentContext, AgentRole, ToolResult
from ..retrieval.retriever import Retriever, QueryExpander

logger = logging.getLogger(__name__)


class TextAnalyzerTool(BaseTool):
    """Analyzes text sections for quality and KSB evidence."""
    name = "analyze_text"
    description = "Analyze a text section for clarity, accuracy, completeness, and relevance"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, section_id: str, content: str, 
                title: str = "", relevant_ksbs: List[str] = None) -> ToolResult:
        relevant_ksbs = relevant_ksbs or []
        
        prompt = f"""Analyze this text section from a student report.

SECTION: {title or section_id}
TEXT:
{content[:2500]}

{f"RELEVANT KSBs: {', '.join(relevant_ksbs)}" if relevant_ksbs else ""}

Evaluate and respond with ONLY a JSON object:
{{
    "section_id": "{section_id}",
    "clarity": {{"score": <1-5>, "notes": "..."}},
    "accuracy": {{"score": <1-5>, "notes": "..."}},
    "completeness": {{"score": <1-5>, "notes": "...", "missing": []}},
    "relevance": {{"score": <1-5>, "notes": "..."}},
    "key_points": ["main points found"],
    "evidence_found": {{"KSB_CODE": ["evidence quotes"]}},
    "strengths": ["list"],
    "issues": ["list"],
    "overall": "strong|adequate|weak"
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.2, max_tokens=1000)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return ToolResult(self.name, True, data)
            return ToolResult(self.name, False, {"section_id": section_id}, "No JSON in response")
        except Exception as e:
            return ToolResult(self.name, False, {"section_id": section_id}, str(e))


class ChartAnalyzerTool(BaseTool):
    """Analyzes charts and graphs using vision."""
    name = "analyze_chart"
    description = "Analyze a chart or graph for data presentation quality"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, chart_id: str, description: str,
                chart_type: str = "unknown", image_base64: str = None) -> ToolResult:
        
        prompt = f"""Analyze this chart from a student report.

CHART ID: {chart_id}
TYPE: {chart_type}
DESCRIPTION: {description}

Evaluate and respond with ONLY a JSON object:
{{
    "chart_id": "{chart_id}",
    "type_appropriate": true|false,
    "presentation": {{"score": <1-5>, "has_title": true|false, "has_labels": true|false, "has_legend": true|false}},
    "data_quality": {{"score": <1-5>, "clear": true|false, "trends_visible": true|false}},
    "insights": ["key insights"],
    "issues": [],
    "strengths": [],
    "overall": "strong|adequate|weak"
}}"""

        try:
            if image_base64:
                response = self.llm.generate(prompt, temperature=0.2, images=[image_base64])
            else:
                response = self.llm.generate(prompt, temperature=0.2)
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, False, {"chart_id": chart_id}, "No JSON")
        except Exception as e:
            return ToolResult(self.name, False, {"chart_id": chart_id}, str(e))


class TableAnalyzerTool(BaseTool):
    """Analyzes tables for structure and content."""
    name = "analyze_table"
    description = "Analyze a table for structure and content quality"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, table_id: str, content: str,
                table_context: str = "") -> ToolResult:
        
        prompt = f"""Analyze this table from a student report.

TABLE ID: {table_id}
CONTENT:
{content[:1500]}
{f"CONTEXT: {table_context}" if table_context else ""}

Respond with ONLY a JSON object:
{{
    "table_id": "{table_id}",
    "structure": {{"rows": <n>, "columns": <n>, "has_headers": true|false, "organized": true|false}},
    "content_quality": {{"score": <1-5>, "complete": true|false, "accurate": true|false}},
    "key_data": ["important data points"],
    "issues": [],
    "strengths": [],
    "overall": "strong|adequate|weak"
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.2)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, False, {"table_id": table_id}, "No JSON")
        except Exception as e:
            return ToolResult(self.name, False, {"table_id": table_id}, str(e))


class ImageAnalyzerTool(BaseTool):
    """Analyzes images and diagrams using vision."""
    name = "analyze_image"
    description = "Analyze an image or diagram for quality and relevance"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, context: AgentContext, image_id: str, image_type: str,
                caption: str = "", image_base64: str = None) -> ToolResult:
        
        prompt = f"""Analyze this {image_type} from a student report.

IMAGE ID: {image_id}
TYPE: {image_type}
{f"CAPTION: {caption}" if caption else ""}

Respond with ONLY a JSON object:
{{
    "image_id": "{image_id}",
    "type": "{image_type}",
    "quality": {{"score": <1-5>, "clarity": "clear|acceptable|unclear"}},
    "annotations": {{"has_labels": true|false, "readable": true|false}},
    "content": {{"description": "what it shows", "key_elements": [], "relevance": "high|medium|low"}},
    "issues": [],
    "strengths": [],
    "overall": "strong|adequate|weak"
}}"""

        try:
            if image_base64:
                response = self.llm.generate(prompt, temperature=0.2, images=[image_base64])
            else:
                response = self.llm.generate(prompt, temperature=0.2)
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return ToolResult(self.name, True, json.loads(json_match.group()))
            return ToolResult(self.name, False, {"image_id": image_id}, "No JSON")
        except Exception as e:
            return ToolResult(self.name, False, {"image_id": image_id}, str(e))


class SectionExtractorTool(BaseTool):
    """Extracts and classifies document sections."""
    name = "extract_sections"
    description = "Extract and classify sections from document chunks"
    
    def execute(self, context: AgentContext) -> ToolResult:
        sections = []
        
        for i, chunk in enumerate(context.chunks):
            chunk_type = chunk.get("chunk_type", "text")
            title = chunk.get("section_title", "")
            content = chunk.get("content", "")
            
            # Classify content type
            title_lower = title.lower() if title else ""
            if any(kw in title_lower for kw in ["introduction", "overview", "summary", "executive"]):
                content_type = "introduction"
            elif any(kw in title_lower for kw in ["requirement", "specification"]):
                content_type = "requirements"
            elif any(kw in title_lower for kw in ["architecture", "design", "infrastructure"]):
                content_type = "design"
            elif any(kw in title_lower for kw in ["implementation", "development", "code"]):
                content_type = "implementation"
            elif any(kw in title_lower for kw in ["result", "evaluation", "test", "benchmark"]):
                content_type = "results"
            elif any(kw in title_lower for kw in ["conclusion", "reflection", "cpd"]):
                content_type = "conclusion"
            else:
                content_type = "content"
            
            sections.append({
                "section_id": f"section_{i}",
                "chunk_type": chunk_type,
                "content_type": content_type,
                "title": title or f"Section {i+1}",
                "content": content,
                "has_figures": chunk.get("has_figure_reference", False)
            })
        
        return ToolResult(self.name, True, {"sections": sections, "count": len(sections)})


class EvidenceFinderTool(BaseTool):
    """Finds evidence for KSB requirements using hybrid search (BM25 + semantic)."""
    name = "find_evidence"
    description = "Search for evidence matching a KSB requirement using hybrid search"

    def __init__(self, embedder=None, vector_store=None, retriever=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever

    def execute(self, context: AgentContext, ksb_code: str, requirement: str) -> ToolResult:
        # Prefer Retriever if available (hybrid search)
        if self.retriever:
            try:
                # Use the battle-tested hybrid search from Retriever
                retrieval_result = self.retriever.retrieve_for_criterion(
                    requirement, ksb_code
                )

                # Extract evidence from RetrievalResult
                evidence = [{
                    "content": chunk.get("content", "")[:400],
                    "section": chunk.get("metadata", {}).get("section_title", "Unknown"),
                    "relevance": chunk.get("similarity", 0),
                    "chunk_id": chunk.get("chunk_id", "")
                } for chunk in retrieval_result.retrieved_chunks[:10]]  # Up to 10 results

                return ToolResult(self.name, True, {
                    "ksb_code": ksb_code,
                    "found": len(evidence) > 0,
                    "evidence": evidence,
                    "search_strategy": retrieval_result.search_strategy,
                    "query_variations": len(retrieval_result.query_variations)
                })
            except Exception as e:
                logger.warning(f"Retriever failed for {ksb_code}, falling back to simple search: {e}")
                # Fall through to simple search

        # Fallback: Simple semantic search
        if self.embedder and self.vector_store:
            try:
                from config import RetrievalConfig
                query = f"{ksb_code} {requirement}"
                query_embedding = self.embedder.embed_query(query)
                results = self.vector_store.query_report(
                    query_embedding,
                    n_results=RetrievalConfig.REPORT_TOP_K  # Use config instead of hardcoded 5
                )

                evidence = [{
                    "content": r.get("content", "")[:400],
                    "section": r.get("metadata", {}).get("section_title", "Unknown"),
                    "relevance": r.get("similarity", 0)
                } for r in results]

                return ToolResult(self.name, True, {
                    "ksb_code": ksb_code,
                    "found": len(evidence) > 0,
                    "evidence": evidence,
                    "search_strategy": "semantic_only"
                })
            except Exception as e:
                return ToolResult(self.name, False, {"ksb_code": ksb_code}, str(e))

        return ToolResult(self.name, True, {"ksb_code": ksb_code, "found": False, "evidence": []})


class AnalysisAgent(BaseAgent):
    """Analysis Agent - Multimodal document analysis."""

    def __init__(self, llm, embedder=None, vector_store=None, verbose: bool = False):
        # Build Retriever if embedder and vector_store are available
        retriever = None
        if embedder and vector_store:
            try:
                from config import RetrievalConfig
                retriever = Retriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    criteria_top_k=RetrievalConfig.CRITERIA_TOP_K,
                    report_top_k=RetrievalConfig.REPORT_TOP_K,
                    max_context_tokens=RetrievalConfig.MAX_CONTEXT_TOKENS,
                    similarity_threshold=RetrievalConfig.SIMILARITY_THRESHOLD,
                    use_hybrid=RetrievalConfig.USE_HYBRID_SEARCH,
                    semantic_weight=RetrievalConfig.SEMANTIC_WEIGHT,
                    keyword_weight=RetrievalConfig.KEYWORD_WEIGHT
                )
                logger.info("Retriever initialized with hybrid search enabled")
            except Exception as e:
                logger.warning(f"Failed to build Retriever, using simple search: {e}")

        tools = [
            TextAnalyzerTool(llm),
            ChartAnalyzerTool(llm),
            TableAnalyzerTool(llm),
            ImageAnalyzerTool(llm),
            SectionExtractorTool(),
            EvidenceFinderTool(embedder, vector_store, retriever)
        ]
        super().__init__(llm, AgentRole.ANALYSIS, tools, verbose)
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
    
    def process(self, context: AgentContext) -> AgentContext:
        """Analyze all document content."""
        
        self._log_verbose("Starting analysis phase...")
        
        # Step 1: Extract sections
        extractor = self.tools["extract_sections"]
        self._log_tool_call("extract_sections")
        sections_result = extractor.execute(context)
        self._log_tool_call("extract_sections", result=sections_result)
        sections = sections_result.data.get("sections", [])
        self._log_verbose(f"Extracted {len(sections)} sections")
        
        # Step 2: Analyze each section
        text_analyzer = self.tools["analyze_text"]
        table_analyzer = self.tools["analyze_table"]
        
        # Get KSB codes list
        ksb_codes = [k.get("code", "") for k in context.ksb_criteria]
        
        # Cap at 15 sections (down from 20) - last sections usually references/appendices
        for section in sections[:15]:
            section_id = section["section_id"]
            content = section["content"]
            title = section["title"]

            # Skip very short sections (likely empty headers or page breaks)
            if len(content.strip()) < 50:
                self._log_verbose(f"Skipping short section {section_id} ({len(content)} chars)")
                continue

            # Find relevant KSBs from brief
            relevant_ksbs = []
            if context.assignment_brief:
                for task in context.assignment_brief.get("tasks", []):
                    if section["content_type"] in task.get("description", "").lower():
                        relevant_ksbs = task.get("mapped_ksbs", [])[:3]
                        break

            if section["chunk_type"] == "table":
                self._log_tool_call("analyze_table", {"section_id": section_id})
                result = table_analyzer.execute(context, section_id, content, title)
                self._log_tool_call("analyze_table", result=result)
                if result.success:
                    context.table_analyses.append(result.data)
            else:
                self._log_tool_call("analyze_text", {"section_id": section_id, "ksbs": relevant_ksbs})
                result = text_analyzer.execute(
                    context, section_id, content, title, relevant_ksbs
                )
                self._log_tool_call("analyze_text", result=result)
                if result.success:
                    result.data["content_type"] = section["content_type"]
                    result.data["ksb_mappings"] = relevant_ksbs
                    context.section_analyses.append(result.data)

        self._log_verbose(f"Analyzed {len(context.section_analyses)} text sections, {len(context.table_analyses)} tables")

        # Step 3: Analyze images (ONLY if images exist and have base64 data)
        if context.images:
            image_analyzer = self.tools["analyze_image"]
            images_with_data = []

            # Filter images that have base64 data (required for vision models)
            for img in context.images[:10]:
                if hasattr(img, 'image_id'):
                    img_data = {"image_id": img.image_id, "caption": getattr(img, 'caption', ''),
                               "base64": getattr(img, 'base64_data', '')}
                else:
                    img_data = img

                # Only analyze if base64 data exists
                if img_data.get("base64"):
                    images_with_data.append(img_data)

            if images_with_data:
                self._log_verbose(f"Analyzing {len(images_with_data)} images with base64 data")
                for img_data in images_with_data:
                    result = image_analyzer.execute(
                        context,
                        img_data.get("image_id", "unknown"),
                        "diagram",
                        img_data.get("caption", ""),
                        img_data.get("base64", None)
                    )
                    if result.success:
                        context.image_analyses.append(result.data)
            else:
                self._log_verbose("Skipping image analysis - no base64 data available")
        
        # Step 4: Find evidence for each KSB
        evidence_finder = self.tools["find_evidence"]
        search_mode = "hybrid" if self.retriever else "semantic-only"
        self._log_verbose(f"Evidence search mode: {search_mode}")

        for ksb in context.ksb_criteria:
            ksb_code = ksb.get("code", "")
            result = evidence_finder.execute(
                context, ksb_code, ksb.get("pass_criteria", "")
            )
            if result.success:
                context.evidence_map[ksb_code] = result.data.get("evidence", [])
                search_strategy = result.data.get("search_strategy", "unknown")
                query_variations = result.data.get("query_variations", 0)
                self._log_verbose(
                    f"  {ksb_code}: {len(result.data.get('evidence', []))} chunks found "
                    f"({search_strategy}, {query_variations} query variations)"
                )
        
        logger.info(f"Analysis complete: {len(context.section_analyses)} sections, "
                   f"{len(context.image_analyses)} images, "
                   f"{len(context.evidence_map)} KSBs mapped")
        
        return context
