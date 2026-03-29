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
import time
from typing import Dict, Any, List, Optional
import logging

import numpy as np

from .core import BaseAgent, BaseTool, AgentContext, AgentRole, ToolResult
from ..retrieval.retriever import Retriever, QueryExpander
from ..criteria.ksb_parser import MODULE_RELEVANCE_TERMS, AVAILABLE_MODULES
from ..utils.logger import create_logger, LogLevel

logger = logging.getLogger(__name__)


class TextAnalyzerTool(BaseTool):
    """Analyzes text sections for quality and KSB evidence."""
    name = "analyze_text"
    description = "Analyze a text section for clarity, accuracy, completeness, and relevance"
    
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, _context: AgentContext, section_id: str, content: str,
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
    
    def execute(self, _context: AgentContext, chart_id: str, description: str,
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
    
    def execute(self, _context: AgentContext, table_id: str, content: str,
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
    
    def execute(self, _context: AgentContext, image_id: str, image_type: str,
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

    @staticmethod
    def _is_boilerplate(chunk_text: str) -> bool:
        """Filter out boilerplate chunks (title pages, headers, reflection tables)."""
        text_lower = chunk_text.lower().strip()

        # Too short to be meaningful evidence
        if len(text_lower) < 100:
            return True

        # Title page patterns (module name + "workplace activity report")
        if any(pattern in text_lower for pattern in [
            "workplace activity report",
            "level 7 artificial intelligence",
            "data science principles -",
            "machine learning with cloud computing -",
            "ai-driven innovation -"
        ]):
            # Check if it's JUST a title (not substantive content with title mention)
            if len(text_lower) < 200:
                return True

        # KSB reflection table headers
        if any(pattern in text_lower for pattern in [
            "reflection on progress against the knowledge",
            "| ksb | reflection",
            "ksb code | reflection",
            "knowledge, skills and behaviours"
        ]):
            # If chunk is primarily table header (short and contains table markers)
            if text_lower.count("|") > 3 and len(text_lower) < 300:
                return True

        # Table of contents
        if all(task in text_lower for task in ["task 1", "task 2", "task 3"]):
            # If it contains all task numbers but is short, likely ToC
            if len(text_lower) < 400:
                return True

        # Schema/structure tables (common boilerplate)
        if "table" in text_lower and "|" in text_lower and len(text_lower) < 150:
            return True

        # KSB Mapping and Reflection table body — contains multiple KSB codes
        # with short one-liner descriptions in a table-like format
        if "ksb mapping" in text_lower or "mapping and reflection" in text_lower:
            # Count KSB code references (K1, S15, B5, etc.)
            ksb_refs = re.findall(r'\b[KSB]\d{1,2}\b', chunk_text)
            if len(ksb_refs) >= 3:
                return True

        # Detect reflection table rows even without the section header:
        # Multiple KSB codes + pipe characters (table format) + short per-row descriptions
        if text_lower.count("|") >= 4:
            ksb_refs = re.findall(r'\b[KSB]\d{1,2}\b', chunk_text)
            if len(ksb_refs) >= 3:
                # Calculate average text length between KSB codes — short = reflection table
                # Split on KSB codes and check if segments are all short (< 100 chars avg)
                segments = re.split(r'\b[KSB]\d{1,2}\b', chunk_text)
                non_empty = [s.strip() for s in segments if s.strip()]
                if non_empty:
                    avg_len = sum(len(s) for s in non_empty) / len(non_empty)
                    if avg_len < 100:
                        return True

        # KSB Reflection Table as first-class table chunk (chunk_type="table")
        # These have [TABLE in ...] headers and contain structured KSB reflection rows
        if chunk_text.lstrip().startswith("[TABLE"):
            ksb_refs = re.findall(r'\b[KSB]\d{1,2}\b', chunk_text)
            if len(ksb_refs) >= 3:
                return True
            # Header row contains "KSB" + "How demonstrated" or "Reflection"
            header_lower = chunk_text[:300].lower()
            if "ksb" in header_lower and any(
                kw in header_lower for kw in ["how demonstrated", "reflection", "where addressed", "evidence location"]
            ):
                return True

        return False

    def _is_off_topic(self, content: str, context: AgentContext) -> bool:
        """Check if content matches an off-topic flagged chunk."""
        # Use first 200 chars as key for matching (handles truncation)
        content_key = content[:200].strip()
        for chunk in context.chunks:
            flag = chunk.get("relevance_flag", "")
            if flag in ("off_topic", "adversarial_reflection"):
                chunk_key = chunk.get("content", "")[:200].strip()
                if chunk_key and chunk_key == content_key:
                    return True
        return False

    def execute(self, context: AgentContext, ksb_code: str, requirement: str) -> ToolResult:
        # Prefer Retriever if available (hybrid search)
        if self.retriever:
            try:
                # Use the battle-tested hybrid search from Retriever
                retrieval_result = self.retriever.retrieve_for_criterion(
                    requirement, ksb_code
                )

                # Extract evidence from RetrievalResult (up to 10 results)
                raw_evidence = [{
                    "content": chunk.get("content", ""),
                    "section": chunk.get("metadata", {}).get("section_title", "Unknown"),
                    "relevance": chunk.get("similarity", 0),
                    "chunk_id": chunk.get("chunk_id", "")
                } for chunk in retrieval_result.retrieved_chunks[:10]]

                # Filter out boilerplate chunks
                filtered_evidence = [
                    ev for ev in raw_evidence
                    if not self._is_boilerplate(ev["content"])
                ]

                # Filter out off-topic chunks
                pre_offtopic = len(filtered_evidence)
                filtered_evidence = [
                    ev for ev in filtered_evidence
                    if not self._is_off_topic(ev["content"], context)
                ]
                off_topic_removed = pre_offtopic - len(filtered_evidence)
                if off_topic_removed > 0:
                    logger.info(
                        f"Filtered {off_topic_removed} off-topic chunks from evidence for {ksb_code}"
                    )

                # If filtering removed too much (less than 3 chunks), keep originals
                evidence = filtered_evidence if len(filtered_evidence) >= 3 else raw_evidence

                # Log filtering results
                if len(filtered_evidence) < len(raw_evidence):
                    logger.debug(
                        f"{ksb_code}: Filtered {len(raw_evidence) - len(filtered_evidence)} boilerplate chunks "
                        f"({len(raw_evidence)} → {len(evidence)})"
                    )

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

                # Extract raw evidence
                raw_evidence = [{
                    "content": r.get("content", ""),
                    "section": r.get("metadata", {}).get("section_title", "Unknown"),
                    "relevance": r.get("similarity", 0)
                } for r in results]

                # Filter out boilerplate chunks
                filtered_evidence = [
                    ev for ev in raw_evidence
                    if not self._is_boilerplate(ev["content"])
                ]

                # Filter out off-topic chunks
                filtered_evidence = [
                    ev for ev in filtered_evidence
                    if not self._is_off_topic(ev["content"], context)
                ]

                # If filtering removed too much, keep originals
                evidence = filtered_evidence if len(filtered_evidence) >= 3 else raw_evidence

                # Log filtering
                if len(filtered_evidence) < len(raw_evidence):
                    logger.debug(
                        f"{ksb_code}: Filtered {len(raw_evidence) - len(filtered_evidence)} boilerplate/off-topic chunks "
                        f"({len(raw_evidence)} → {len(evidence)})"
                    )

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

    def __init__(self, llm, embedder=None, vector_store=None, verbose: bool = False, module_code: str = "MLCC"):
        self.module_code = module_code
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

        # Initialize ImageProcessor for OCR capabilities
        from ..document_processing.image_processor import ImageProcessor
        self.image_processor = ImageProcessor(ollama_client=llm)

        # Initialize enhanced logger
        from config import LOG_LEVEL
        log_level = LogLevel.VERBOSE if verbose else LogLevel(LOG_LEVEL)
        self.enhanced_logger = create_logger("ANALYSIS", log_level, verbose)
    
    def process(self, context: AgentContext) -> AgentContext:
        """Analyze all document content."""

        self.enhanced_logger.phase("Starting analysis phase...")
        analysis_start = time.time()

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

        # Step 3.5: OCR preprocessing for images (extract text from code/charts/diagrams)
        if context.images and self.image_processor:
            from config import OCR_ENABLED

            if OCR_ENABLED:
                with self.enhanced_logger.timer(f"OCR extraction ({len(context.images)} images)", warn_threshold_ms=len(context.images) * 2000):
                    self.enhanced_logger.info(f"Running OCR on {len(context.images)} images...")
                    ocr_chunks = []
                    total_chars_extracted = 0

                    for idx, img in enumerate(context.images, 1):  # Process all images (charts, diagrams, graphs, code)
                        self.enhanced_logger.progress(idx, len(context.images), "image")

                        # Convert to dict if it's a ProcessedImage object
                        if hasattr(img, 'image_id'):
                            img_dict = {
                                'image_id': img.image_id,
                                'base64_data': getattr(img, 'base64_data', ''),
                                'caption': getattr(img, 'caption', '')
                            }
                        else:
                            img_dict = img

                        if not img_dict.get('base64_data') and not img_dict.get('base64'):
                            continue

                        # Create a simple object for OCR extraction
                        class ImageData:
                            def __init__(self, data):
                                self.image_id = data.get('image_id', 'unknown')
                                self.base64_data = data.get('base64_data') or data.get('base64', '')
                                self.caption = data.get('caption', '')

                        img_obj = ImageData(img_dict)

                        # Time individual OCR operation
                        ocr_start = time.time()
                        ocr_text = self.image_processor.extract_text_with_ocr(img_obj)
                        ocr_duration_ms = (time.time() - ocr_start) * 1000

                        if ocr_text and len(ocr_text) > 10:  # Skip empty/trivial results
                            # Create searchable chunk from OCR text
                            ocr_chunk = {
                                'content': f"[OCR from {img_obj.image_id}]: {ocr_text}",
                                'chunk_type': 'ocr',
                                'chunk_index': len(context.chunks),
                                'metadata': {
                                    'source_image': img_obj.image_id,
                                    'caption': img_obj.caption,
                                    'has_figure_reference': True
                                }
                            }
                            ocr_chunks.append(ocr_chunk)
                            total_chars_extracted += len(ocr_text)
                            self.enhanced_logger.ocr_result(img_obj.image_id, len(ocr_text), ocr_duration_ms)
                        else:
                            self.enhanced_logger.ocr_result(img_obj.image_id, 0, ocr_duration_ms)

                    # Add OCR chunks to context for embedding/search
                    if ocr_chunks:
                        with self.enhanced_logger.timer(f"Embedding {len(ocr_chunks)} OCR chunks", warn_threshold_ms=5000):
                            context.chunks.extend(ocr_chunks)

                            # Re-embed OCR chunks so they're searchable
                            if self.embedder and self.vector_store:
                                try:
                                    ocr_texts = [c['content'] for c in ocr_chunks]
                                    ocr_embeddings = self.embedder.embed_documents(ocr_texts)
                                    self.vector_store.add_report(ocr_chunks, ocr_embeddings)
                                    self.enhanced_logger.success(f"Added {len(ocr_chunks)} OCR chunks ({total_chars_extracted} chars) to vector store")
                                except Exception as e:
                                    self.enhanced_logger.error(f"Failed to embed OCR chunks", e)
                    else:
                        self.enhanced_logger.warning("No text extracted from images via OCR", LogLevel.STANDARD)
            else:
                self.enhanced_logger.debug("OCR disabled in config")

        # Step 3.6: Content relevance validation (runs after OCR so image chunks are included)
        context.content_quality = self._validate_content_relevance(context)
        quality_flag = context.content_quality.get("quality_flag", "OK")
        if quality_flag != "OK":
            self.enhanced_logger.warning(
                f"Content quality: {quality_flag} — "
                f"{context.content_quality.get('off_topic_chunks', 0)} off-topic, "
                f"{context.content_quality.get('off_topic_images', 0)} off-topic images, "
                f"{context.content_quality.get('adversarial_tables_detected', 0)} adversarial tables",
                LogLevel.STANDARD
            )

        # Step 4: Find evidence for each KSB
        evidence_finder = self.tools["find_evidence"]
        search_mode = "hybrid" if self.retriever else "semantic-only"
        self.enhanced_logger.info(f"Evidence search mode: {search_mode}")

        total_ksbs = len(context.ksb_criteria)
        low_evidence_count = 0

        with self.enhanced_logger.timer(f"Evidence search for {total_ksbs} KSBs", warn_threshold_ms=total_ksbs * 1000):
            for idx, ksb in enumerate(context.ksb_criteria, 1):
                self.enhanced_logger.progress(idx, total_ksbs, "KSB")

                ksb_code = ksb.get("code", "")
                result = evidence_finder.execute(
                    context, ksb_code, ksb.get("pass_criteria", "")
                )
                if result.success:
                    evidence = result.data.get("evidence", [])
                    context.evidence_map[ksb_code] = evidence
                    search_strategy = result.data.get("search_strategy", "unknown")
                    query_variations = result.data.get("query_variations", 0)

                    # Count OCR chunks in evidence
                    ocr_chunk_count = sum(1 for e in evidence if '[OCR from' in e)

                    # Store metadata for audit trail (used by ScoringAgent)
                    context.evidence_metadata[ksb_code] = {
                        'search_strategy': search_strategy,
                        'query_variations': query_variations,
                        'total_chunks': len(evidence),
                        'ocr_chunks': ocr_chunk_count
                    }

                    # Calculate average similarity if available
                    avg_similarity = None
                    if evidence and isinstance(evidence[0], dict) and 'similarity' in evidence[0]:
                        avg_similarity = sum(e.get('similarity', 0) for e in evidence) / len(evidence)

                    # Log evidence stats
                    self.enhanced_logger.evidence_stats(
                        ksb_code,
                        len(evidence),
                        search_strategy,
                        query_variations,
                        ocr_chunk_count,
                        avg_similarity
                    )

                    if len(evidence) < 3:
                        low_evidence_count += 1

        # Log summary metrics
        analysis_duration = time.time() - analysis_start
        self.enhanced_logger.metric("total_sections", len(context.section_analyses))
        self.enhanced_logger.metric("total_images", len(context.image_analyses))
        self.enhanced_logger.metric("ksbs_mapped", len(context.evidence_map))
        self.enhanced_logger.metric("ksbs_with_low_evidence", low_evidence_count)
        self.enhanced_logger.metric("analysis_duration_sec", f"{analysis_duration:.1f}")

        self.enhanced_logger.success(
            f"Analysis complete: {len(context.section_analyses)} sections, "
            f"{len(context.image_analyses)} images, "
            f"{len(context.evidence_map)} KSBs mapped, "
            f"{low_evidence_count} with low evidence (<3 chunks)"
        )

        return context

    def _validate_content_relevance(self, context: AgentContext) -> Dict[str, Any]:
        """Validate content relevance across all chunks (text + OCR) using embedding similarity."""
        chunks = context.chunks
        if not chunks or not self.embedder:
            return {"total_chunks": len(chunks), "on_topic_chunks": len(chunks),
                    "off_topic_chunks": 0, "low_relevance_chunks": 0,
                    "off_topic_images": 0, "adversarial_tables_detected": 0,
                    "off_topic_sections": [], "off_topic_image_ids": [],
                    "adversarial_tables": [], "quality_flag": "OK"}

        # Build module reference text from description + KSB titles
        module_info = AVAILABLE_MODULES.get(self.module_code, {})
        ksb_titles = []
        for ksb in context.ksb_criteria:
            title = ksb.get("title", "")
            if title:
                ksb_titles.append(title)
        reference_text = (module_info.get("description", "") + " " +
                          " ".join(ksb_titles))

        # Compute reference embedding (as a document, not query)
        ref_embedding = self.embedder.embed_documents([reference_text])[0]

        # Compute chunk embeddings (use existing content)
        chunk_texts = [c.get("content", "") for c in chunks]
        chunk_embeddings = self.embedder.embed_documents(chunk_texts)

        # Compute cosine similarities
        similarities = self.embedder.similarity(ref_embedding, chunk_embeddings)

        off_topic_chunks = 0
        low_relevance_chunks = 0
        on_topic_chunks = 0
        off_topic_images = 0
        off_topic_sections = []
        off_topic_image_ids = []
        adversarial_tables = []

        relevance_terms = MODULE_RELEVANCE_TERMS.get(self.module_code, [])
        relevance_terms_lower = [t.lower() for t in relevance_terms]

        for i, chunk in enumerate(chunks):
            sim = float(similarities[i])
            content = chunk.get("content", "")
            section_title = chunk.get("section_title", "") or ""
            chunk_type = chunk.get("chunk_type", "")

            if sim < 0.05:
                chunk["relevance_flag"] = "off_topic"
                off_topic_chunks += 1
                label = section_title or f"Chunk {i}"
                if label not in off_topic_sections:
                    off_topic_sections.append(label)
                # Track off-topic images separately
                if chunk_type == "ocr":
                    off_topic_images += 1
                    source_img = chunk.get("metadata", {}).get("source_image", f"image_{i}")
                    if source_img not in off_topic_image_ids:
                        off_topic_image_ids.append(source_img)
            elif sim < 0.10:
                chunk["relevance_flag"] = "low_relevance"
                low_relevance_chunks += 1
            else:
                chunk["relevance_flag"] = "on_topic"
                on_topic_chunks += 1

            # Adversarial reflection table check for table chunks
            if chunk_type == "table" or content.lstrip().startswith("[TABLE"):
                if self._is_adversarial_table(content, relevance_terms_lower):
                    chunk["relevance_flag"] = "adversarial_reflection"
                    label = section_title or f"Table chunk {i}"
                    adversarial_tables.append(label)
                    # Reclassify: move from whatever bucket to off-topic
                    if sim >= 0.10:
                        on_topic_chunks -= 1
                    elif sim >= 0.05:
                        low_relevance_chunks -= 1
                    else:
                        off_topic_chunks -= 1  # Already counted
                    off_topic_chunks += 1

        # Determine quality flag
        if off_topic_chunks >= 3 or len(adversarial_tables) > 0:
            quality_flag = "CRITICAL"
        elif off_topic_chunks >= 1 or low_relevance_chunks > 0:
            quality_flag = "WARNING"
        else:
            quality_flag = "OK"

        return {
            "total_chunks": len(chunks),
            "on_topic_chunks": on_topic_chunks,
            "off_topic_chunks": off_topic_chunks,
            "low_relevance_chunks": low_relevance_chunks,
            "off_topic_images": off_topic_images,
            "adversarial_tables_detected": len(adversarial_tables),
            "off_topic_sections": off_topic_sections,
            "off_topic_image_ids": off_topic_image_ids,
            "adversarial_tables": adversarial_tables,
            "quality_flag": quality_flag
        }

    @staticmethod
    def _is_adversarial_table(content: str, relevance_terms_lower: list) -> bool:
        """Check if a table chunk is an adversarial reflection table.

        Heuristic: table has a column with KSB codes but content column
        contains no module-relevant terms in >=80% of rows.
        """
        lines = content.split("\n")
        # Need at least header + separator + 1 data row
        if len(lines) < 3:
            return False

        # Check if table has KSB codes in first column
        ksb_pattern = re.compile(r'\b[KSB]\d{1,2}\b')
        data_rows = []
        for line in lines:
            if not line.strip() or line.strip().startswith("---") or line.strip().startswith("[TABLE"):
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                if ksb_pattern.search(cells[0]):
                    data_rows.append(cells[1:])  # Content columns

        if len(data_rows) < 2:
            return False

        # Check how many rows contain module-relevant terms
        relevant_rows = 0
        for row_cells in data_rows:
            row_text = " ".join(row_cells).lower()
            if any(term in row_text for term in relevance_terms_lower):
                relevant_rows += 1

        # If fewer than 20% of rows contain relevant terms, flag as adversarial
        return relevant_rows / len(data_rows) < 0.20
