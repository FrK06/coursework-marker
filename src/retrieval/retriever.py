"""
Retriever - Per-criterion retrieval strategy for coursework evaluation.

Strategy:
1. Extract individual criteria from the rubric
2. For each criterion, retrieve relevant report chunks
3. Build focused context for LLM evaluation
4. Limit total context to avoid overwhelming the model
"""
from typing import List, Dict, Any, Optional, Tuple                                # "Optional" is not accessed
from dataclasses import dataclass
import re
import logging

import numpy as np                                                                  # "np" is not accessed

from ..embeddings import Embedder
from ..vector_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval for a single criterion."""
    criterion_id: str
    criterion_text: str
    retrieved_chunks: List[Dict[str, Any]]
    total_tokens: int
    has_figure_evidence: bool


@dataclass  
class RetrievalContext:
    """Complete retrieval context for evaluation."""
    criteria_results: List[RetrievalResult]
    report_summary: str
    total_criteria: int
    total_tokens: int


class Retriever:
    """
    Retriever for per-criterion document comparison.
    
    Strategy:
    - Parse criteria document to identify individual criteria
    - For each criterion, query report for relevant evidence
    - Build context that fits within token budget
    - Prioritize most relevant chunks
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaStore,
        criteria_top_k: int = 3,
        report_top_k: int = 5,
        max_context_tokens: int = 2000,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for query embedding
            vector_store: ChromaStore instance with indexed documents
            criteria_top_k: Number of criteria chunks to retrieve
            report_top_k: Number of report chunks per criterion
            max_context_tokens: Maximum tokens per criterion context
            similarity_threshold: Minimum similarity for inclusion
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.criteria_top_k = criteria_top_k
        self.report_top_k = report_top_k
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold
    
    def extract_criteria_list(self) -> List[Dict[str, str]]:
        """
        Extract the list of criteria from the stored criteria document.
        
        Returns:
            List of dicts with 'id' and 'text' for each criterion
        """
        all_criteria = self.vector_store.get_all_criteria()
        
        if not all_criteria:
            logger.warning("No criteria found in vector store")
            return []
        
        # Group by criterion_id
        criteria_map = {}
        
        for chunk in all_criteria:
            criterion_id = chunk['metadata'].get('criterion_id', '')
            
            if criterion_id:
                if criterion_id not in criteria_map:
                    criteria_map[criterion_id] = {
                        'id': criterion_id,
                        'chunks': []
                    }
                criteria_map[criterion_id]['chunks'].append(chunk['content'])
            else:
                # General criteria text, assign to "general"
                if 'general' not in criteria_map:
                    criteria_map['general'] = {
                        'id': 'general',
                        'chunks': []
                    }
                criteria_map['general']['chunks'].append(chunk['content'])
        
        # Combine chunks for each criterion
        criteria_list = []
        for crit_id, data in sorted(criteria_map.items()):                                                         # "crit_id" is not accessed
            criteria_list.append({
                'id': data['id'],
                'text': '\n\n'.join(data['chunks'])
            })
        
        return criteria_list
    
    def retrieve_for_criterion(
        self,
        criterion_text: str,
        criterion_id: str = ""
    ) -> RetrievalResult:
        """
        Retrieve relevant report chunks for a single criterion.
        Uses multi-query expansion for better recall.
        
        Args:
            criterion_text: The criterion text to match against
            criterion_id: Identifier for the criterion
            
        Returns:
            RetrievalResult with relevant chunks
        """
        # Generate multiple query variations for better coverage
        queries = self._expand_query(criterion_text)
        
        all_results = {}
        
        for query in queries:
            # Embed the query
            query_embedding = self.embedder.embed_query(query)
            
            # Query report collection
            results = self.vector_store.query_report(
                query_embedding=query_embedding,
                n_results=self.report_top_k
            )
            
            # Add to results, keeping best similarity for each chunk
            for r in results:
                chunk_id = r.get('id', str(hash(r['content'][:100])))
                if chunk_id not in all_results or r.get('similarity', 0) > all_results[chunk_id].get('similarity', 0):
                    all_results[chunk_id] = r
        
        # Sort by similarity
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in sorted_results
            if r.get('similarity', 0) >= self.similarity_threshold
        ]
        
        # Limit to token budget
        selected_chunks = []
        total_tokens = 0
        has_figure_evidence = False
        
        for result in filtered_results[:self.report_top_k * 2]:  # Allow more chunks with multi-query
            chunk_tokens = result['metadata'].get('token_count', 100)
            
            if total_tokens + chunk_tokens <= self.max_context_tokens:
                selected_chunks.append(result)
                total_tokens += chunk_tokens
                
                if result['metadata'].get('has_figure_reference', False):
                    has_figure_evidence = True
        
        logger.info(
            f"Multi-query retrieval for '{criterion_id}': "
            f"{len(queries)} queries, {len(all_results)} unique chunks, "
            f"{len(selected_chunks)} selected"
        )
        
        return RetrievalResult(
            criterion_id=criterion_id,
            criterion_text=criterion_text,
            retrieved_chunks=selected_chunks,
            total_tokens=total_tokens,
            has_figure_evidence=has_figure_evidence
        )
    
    def _expand_query(self, criterion_text: str) -> List[str]:
        """
        Expand a criterion into multiple search queries for better recall.
        
        This helps retrieve content that may be in different sections
        using different terminology.
        """
        queries = [criterion_text[:500]]  # Original query (truncated)
        
        # Extract key terms and create focused queries
        text_lower = criterion_text.lower()
        
        # Common criterion themes and their variations
        theme_expansions = {
            'gdpr': ['GDPR compliance', 'data protection', 'privacy', 'personal data', 'data subject rights', 'lawful basis', 'data minimisation'],
            'security': ['security controls', 'encryption', 'access control', 'IAM', 'authentication', 'audit logging'],
            'architecture': ['architecture diagram', 'system design', 'cloud services', 'infrastructure', 'VPC', 'network'],
            'cost': ['cost analysis', 'pricing', 'budget', 'cost optimization', 'expenses'],
            'benchmark': ['benchmarking', 'performance comparison', 'CPU vs GPU', 'training time', 'throughput'],
            'feasibility': ['feasibility', 'go/no-go', 'constraints', 'risks', 'mitigation'],
            'requirements': ['requirements', 'functional requirements', 'non-functional', 'specifications'],
        }
        
        for theme, expansions in theme_expansions.items():
            if theme in text_lower:
                queries.extend(expansions[:3])  # Add up to 3 variations per theme
        
        # Extract quoted or emphasized terms
        quoted = re.findall(r'"([^"]+)"', criterion_text)
        queries.extend(quoted[:2])
        
        # Add section-based queries for common report sections
        if 'compliance' in text_lower or 'gdpr' in text_lower:
            queries.extend(['ethics legal social', 'data storage encryption', 'compliance controls'])
        
        if 'architecture' in text_lower or 'design' in text_lower:
            queries.extend(['target architecture', 'architecture components', 'justification requirements'])
        
        if 'benchmark' in text_lower or 'performance' in text_lower:
            queries.extend(['training epochs cost', 'CPU GPU comparison', 'metrics measurement'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen and len(q) > 5:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:8]  # Limit to 8 queries max
    
    def retrieve_all(self) -> RetrievalContext:
        """
        Retrieve evidence for all criteria.
        
        Returns:
            RetrievalContext with all criteria results
        """
        criteria_list = self.extract_criteria_list()
        
        if not criteria_list:
            logger.warning("No criteria extracted, using full document comparison")
            # Fall back to general retrieval
            all_criteria = self.vector_store.get_all_criteria()
            combined_text = '\n'.join([c['content'] for c in all_criteria])
            
            result = self.retrieve_for_criterion(combined_text[:1000], "all")
            return RetrievalContext(
                criteria_results=[result],
                report_summary="",
                total_criteria=1,
                total_tokens=result.total_tokens
            )
        
        # Retrieve for each criterion
        results = []
        total_tokens = 0
        
        for criterion in criteria_list:
            result = self.retrieve_for_criterion(
                criterion['text'],
                criterion['id']
            )
            results.append(result)
            total_tokens += result.total_tokens
            
            logger.info(
                f"Criterion {criterion['id']}: "
                f"retrieved {len(result.retrieved_chunks)} chunks, "
                f"{result.total_tokens} tokens"
            )
        
        # Generate report summary from most relevant chunks
        report_summary = self._generate_report_summary()
        
        return RetrievalContext(
            criteria_results=results,
            report_summary=report_summary,
            total_criteria=len(criteria_list),
            total_tokens=total_tokens
        )
    
    def _generate_report_summary(self) -> str:
        """
        Generate a brief summary of the report structure.
        
        Uses section titles and headings from indexed chunks.
        """
        # Get a sample of report chunks for structure
        # Use a general query to get diverse chunks
        general_query = "introduction methodology results discussion conclusion"
        query_embedding = self.embedder.embed_query(general_query)
        
        results = self.vector_store.query_report(
            query_embedding=query_embedding,
            n_results=10
        )
        
        # Extract unique sections
        sections = set()
        for result in results:
            section = result['metadata'].get('section_title', '')
            if section:
                sections.add(section)
        
        if sections:
            return "Report sections identified: " + ", ".join(sorted(sections))
        
        return "Report structure analysis not available"
    
    def retrieve_with_figures(
        self,
        criterion_text: str,
        criterion_id: str = ""
    ) -> Tuple[RetrievalResult, List[str]]:
        """
        Retrieve chunks and identify relevant figures.
        
        Returns:
            Tuple of (RetrievalResult, list of figure_ids to analyze)
        """
        result = self.retrieve_for_criterion(criterion_text, criterion_id)
        
        # Collect figure IDs from retrieved chunks
        figure_ids = []
        for chunk in result.retrieved_chunks:
            # Check if chunk references figures
            content = chunk.get('content', '')
            figure_refs = re.findall(r'[Ff]igure\s+(\d+)', content)
            figure_ids.extend([f"figure_{ref}" for ref in figure_refs])
        
        return result, list(set(figure_ids))
    
    def format_context_for_llm(
        self,
        result: RetrievalResult,
        include_page_refs: bool = True
    ) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            result: RetrievalResult to format
            include_page_refs: Include page number references
            
        Returns:
            Formatted context string with accurate page/section references
        """
        if not result.retrieved_chunks:
            return "No relevant evidence found in the report for this criterion."
        
        context_parts = []
        
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Build location reference
            location_parts = []
            
            # Page info
            if include_page_refs:
                page_start = metadata.get('page_start', '?')
                page_end = metadata.get('page_end', page_start)
                if page_start == page_end:
                    location_parts.append(f"page {page_start}")
                else:
                    location_parts.append(f"pages {page_start}-{page_end}")
            
            # Section number (use actual number from document, not hallucinated)
            section_number = metadata.get('section_number', '')
            section_title = metadata.get('section_title', '')
            
            if section_number:
                location_parts.append(f"Section {section_number}")
            elif section_title:
                # Extract section number from title if present
                import re
                num_match = re.match(r'^(\d+(?:\.\d+)*)', section_title)
                if num_match:
                    location_parts.append(f"Section {num_match.group(1)}")
                else:
                    location_parts.append(f"Section: {section_title[:50]}")
            
            location_str = " / ".join(location_parts) if location_parts else ""
            location_display = f" ({location_str})" if location_str else ""
            
            context_parts.append(
                f"Evidence {i}{location_display}:\n{content}"
            )
        
        # Add instruction for LLM to use only these references
        header = "IMPORTANT: Use ONLY the page numbers and section numbers shown below when citing evidence. Do not invent section numbers like 2.1, 3.3 - use the exact references provided.\n\n"
        
        return header + "\n\n---\n\n".join(context_parts)
