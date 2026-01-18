"""
Retriever - Advanced retrieval with hybrid search and query expansion.

Improvements:
1. Hybrid search combining BM25 (keyword) and semantic similarity
2. Multi-query expansion for better recall
3. Reciprocal Rank Fusion (RRF) for result combination
4. Contextual reranking based on section relevance
5. Configurable search strategies
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
import math
import logging

import numpy as np

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
    search_strategy: str = "hybrid"  # 'semantic', 'keyword', 'hybrid'
    query_variations: List[str] = field(default_factory=list)


@dataclass  
class RetrievalContext:
    """Complete retrieval context for evaluation."""
    criteria_results: List[RetrievalResult]
    report_summary: str
    total_criteria: int
    total_tokens: int


class BM25:
    """
    BM25 implementation for keyword-based search.
    
    BM25 is a probabilistic retrieval model that ranks documents
    based on term frequency and document length normalization.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0-1, 0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = defaultdict(int)  # Term -> number of docs containing term
        self.term_freqs = []  # List of {term: freq} for each doc
        self.num_docs = 0
        self.documents = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.num_docs = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.term_freqs.append(dict(tf))
            
            # Update document frequencies
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a query against a document.
        
        Args:
            query: Query string
            doc_idx: Index of document in corpus
            
        Returns:
            BM25 score
        """
        query_tokens = self._tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0
        for token in query_tokens:
            if token not in doc_tf:
                continue
            
            tf = doc_tf[token]
            idf = self._idf(token)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        scores = []
        for idx in range(self.num_docs):
            score = self.score(query, idx)
            if score > 0:
                scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class QueryExpander:
    """
    Expands queries for better retrieval coverage.
    
    Strategies:
    1. Synonym expansion
    2. Acronym expansion
    3. Related concept extraction
    4. Question reformulation
    """
    
    # Domain-specific synonyms and related terms
    TERM_EXPANSIONS = {
        # Technical terms
        'ml': ['machine learning', 'model', 'algorithm'],
        'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
        'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'deployment'],
        'gpu': ['graphics processing unit', 'cuda', 'accelerator', 'compute'],
        'cpu': ['processor', 'compute', 'central processing unit'],
        'api': ['interface', 'endpoint', 'service'],
        
        # Data terms
        'gdpr': ['data protection', 'privacy', 'compliance', 'personal data'],
        'data': ['dataset', 'information', 'records'],
        'storage': ['database', 's3', 'blob', 'warehouse', 'lake'],
        'etl': ['pipeline', 'extraction', 'transformation', 'loading'],
        
        # ML terms
        'training': ['learning', 'fitting', 'optimization'],
        'inference': ['prediction', 'serving', 'deployment'],
        'model': ['algorithm', 'classifier', 'neural network'],
        'accuracy': ['performance', 'metrics', 'evaluation', 'precision', 'recall'],
        'benchmark': ['comparison', 'evaluation', 'performance test'],
        
        # Architecture terms
        'architecture': ['design', 'structure', 'system', 'infrastructure'],
        'scalability': ['scaling', 'elasticity', 'auto-scaling'],
        'security': ['encryption', 'authentication', 'access control', 'iam'],
        
        # Business terms
        'requirements': ['specifications', 'needs', 'criteria'],
        'stakeholder': ['user', 'customer', 'client', 'business'],
        'feasibility': ['viability', 'assessment', 'evaluation'],
        'cost': ['pricing', 'budget', 'expense', 'economics'],
    }
    
    # KSB-specific expansions
    KSB_CONTEXT = {
        'K1': ['business objective', 'ml methodology', 'problem framing'],
        'K2': ['storage', 'processing', 'organizational impact', 'data pipeline'],
        'K16': ['high performance', 'gpu', 'cpu', 'architecture', 'compute'],
        'K18': ['programming', 'data engineering', 'pipeline', 'code'],
        'K19': ['statistical', 'ml principles', 'evaluation', 'overfitting'],
        'K25': ['ml libraries', 'pytorch', 'tensorflow', 'keras'],
        'S15': ['deployment', 'service', 'platform', 'poc'],
        'S16': ['requirements', 'data management', 'cloud', 'governance'],
        'S19': ['scalable', 'infrastructure', 'benchmarking'],
        'S23': ['best practice', 'dissemination', 'documentation'],
        'B5': ['cpd', 'professional development', 'learning'],
    }
    
    def expand_query(self, query: str, ksb_code: str = "") -> List[str]:
        """
        Expand a query into multiple variations.
        
        Args:
            query: Original query
            ksb_code: Optional KSB code for context-aware expansion
            
        Returns:
            List of query variations including original
        """
        queries = [query]
        
        # Extract key terms from query
        query_lower = query.lower()
        words = set(re.findall(r'\b[a-z]+\b', query_lower))
        
        # Add term expansions
        expansion_terms = set()
        for word in words:
            if word in self.TERM_EXPANSIONS:
                expansion_terms.update(self.TERM_EXPANSIONS[word])
        
        # Create expanded queries
        if expansion_terms:
            # Add individual expansion terms as queries
            for term in list(expansion_terms)[:5]:
                if term not in query_lower:
                    queries.append(term)
            
            # Create combined expansion query
            combined = f"{query} {' '.join(list(expansion_terms)[:3])}"
            queries.append(combined)
        
        # Add KSB-specific context
        if ksb_code:
            ksb_upper = ksb_code.upper()
            # Handle multiple KSBs (e.g., "K1,K2,S16")
            for code in ksb_upper.split(','):
                code = code.strip()
                if code in self.KSB_CONTEXT:
                    for context_term in self.KSB_CONTEXT[code][:3]:
                        if context_term not in query_lower:
                            queries.append(context_term)
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        queries.extend(quoted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 2:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:10]  # Limit to 10 variations
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for focused retrieval."""
        concepts = []
        
        # Look for specific patterns
        patterns = [
            r'\b(?:using|with|via|through)\s+([A-Z][a-zA-Z0-9\s]+)',  # Technologies
            r'\b([A-Z]{2,})\b',  # Acronyms
            r'\b(\d+(?:\.\d+)?)\s*(?:GB|MB|TB|ms|seconds|hours)',  # Metrics
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            concepts.extend([m.strip() for m in matches if len(m.strip()) > 1])
        
        return list(set(concepts))[:5]


class Retriever:
    """
    Advanced retriever with hybrid search capabilities.
    
    Features:
    - Hybrid search: combines BM25 and semantic similarity
    - Query expansion for better recall
    - Reciprocal Rank Fusion for result combination
    - Section-aware retrieval
    - Configurable search strategies
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaStore,
        criteria_top_k: int = 3,
        report_top_k: int = 8,
        max_context_tokens: int = 3000,
        similarity_threshold: float = 0.2,
        use_hybrid: bool = True,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
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
            use_hybrid: Whether to use hybrid search
            semantic_weight: Weight for semantic search in hybrid
            keyword_weight: Weight for keyword search in hybrid
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.criteria_top_k = criteria_top_k
        self.report_top_k = report_top_k
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Initialize query expander
        self.query_expander = QueryExpander()
        
        # BM25 index (built lazily)
        self._bm25_index = None
        self._bm25_docs = None
    
    def _build_bm25_index(self) -> Optional[BM25]:
        """Build BM25 index from report chunks."""
        try:
            # Get all report chunks
            results = self.vector_store.report_collection.get(
                include=["documents", "metadatas"]
            )
            
            if not results['documents']:
                return None
            
            self._bm25_docs = results
            bm25 = BM25()
            bm25.fit(results['documents'])
            
            logger.info(f"Built BM25 index with {len(results['documents'])} documents")
            return bm25
            
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            return None
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[str, float]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each list
        
        Args:
            result_lists: List of [(doc_id, score), ...] for each search method
            k: Ranking constant (higher = less weight to top results)
            
        Returns:
            Combined ranked list
        """
        scores = defaultdict(float)
        
        for result_list in result_lists:
            for rank, (doc_id, _) in enumerate(result_list):
                scores[doc_id] += 1 / (k + rank + 1)
        
        # Sort by RRF score
        combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return combined
    
    def extract_criteria_list(self) -> List[Dict[str, str]]:
        """Extract the list of criteria from stored documents."""
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
                if 'general' not in criteria_map:
                    criteria_map['general'] = {'id': 'general', 'chunks': []}
                criteria_map['general']['chunks'].append(chunk['content'])
        
        # Combine chunks for each criterion
        criteria_list = []
        for _, data in sorted(criteria_map.items()):
            criteria_list.append({
                'id': data['id'],
                'text': '\n\n'.join(data['chunks'])
            })
        
        return criteria_list
    
    def _semantic_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """Perform semantic search using embeddings."""
        query_embedding = self.embedder.embed_query(query)
        
        results = self.vector_store.query_report(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        return [
            (r.get('chunk_id', str(i)), r.get('similarity', 0))
            for i, r in enumerate(results)
        ]
    
    def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """Perform keyword search using BM25."""
        if self._bm25_index is None:
            self._bm25_index = self._build_bm25_index()
        
        if self._bm25_index is None or self._bm25_docs is None:
            return []
        
        bm25_results = self._bm25_index.search(query, top_k=n_results)
        
        results = []
        for doc_idx, score in bm25_results:
            doc_id = self._bm25_docs['ids'][doc_idx]
            # Normalize score to 0-1 range (approximately)
            normalized_score = min(score / 10, 1.0)
            results.append((doc_id, normalized_score))
        
        return results
    
    def _hybrid_search(
        self,
        queries: List[str],
        n_results: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword methods.
        
        Args:
            queries: List of query variations
            n_results: Number of results per query
            
        Returns:
            Dictionary of chunk_id -> chunk data with combined scores
        """
        all_semantic_results = []
        all_keyword_results = []
        
        for query in queries:
            # Semantic search
            semantic_results = self._semantic_search(query, n_results)
            all_semantic_results.append(semantic_results)
            
            # Keyword search (if hybrid enabled)
            if self.use_hybrid:
                keyword_results = self._keyword_search(query, n_results)
                all_keyword_results.append(keyword_results)
        
        # Combine results using RRF
        if self.use_hybrid and all_keyword_results:
            # Combine all semantic results
            semantic_combined = self._reciprocal_rank_fusion(all_semantic_results)
            # Combine all keyword results
            keyword_combined = self._reciprocal_rank_fusion(all_keyword_results)
            
            # Final fusion with weights
            final_scores = defaultdict(float)
            
            for doc_id, score in semantic_combined:
                final_scores[doc_id] += score * self.semantic_weight
            
            for doc_id, score in keyword_combined:
                final_scores[doc_id] += score * self.keyword_weight
            
            ranked_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # Semantic only
            ranked_ids = self._reciprocal_rank_fusion(all_semantic_results)
        
        # Fetch full chunk data for top results
        results = {}
        chunk_ids_to_fetch = [doc_id for doc_id, _ in ranked_ids[:n_results * 2]]
        
        if chunk_ids_to_fetch:
            try:
                fetched = self.vector_store.report_collection.get(
                    ids=chunk_ids_to_fetch,
                    include=["documents", "metadatas"]
                )
                
                for i, chunk_id in enumerate(fetched['ids']):
                    results[chunk_id] = {
                        'chunk_id': chunk_id,
                        'content': fetched['documents'][i],
                        'metadata': fetched['metadatas'][i],
                        'similarity': dict(ranked_ids).get(chunk_id, 0)
                    }
            except Exception as e:
                logger.warning(f"Error fetching chunks: {e}")
        
        return results
    
    def retrieve_for_criterion(
        self,
        criterion_text: str,
        criterion_id: str = ""
    ) -> RetrievalResult:
        """
        Retrieve relevant report chunks for a single criterion.
        
        Uses hybrid search with query expansion for comprehensive retrieval.
        """
        # Expand query
        queries = self.query_expander.expand_query(
            criterion_text[:500],
            criterion_id
        )
        
        # Add key concept queries
        concepts = self.query_expander.extract_key_concepts(criterion_text)
        queries.extend(concepts)
        
        # Perform hybrid search
        all_results = self._hybrid_search(queries, self.report_top_k)
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )
        
        # Filter by threshold and token budget
        selected_chunks = []
        total_tokens = 0
        has_figure_evidence = False
        
        for result in sorted_results:
            if result.get('similarity', 0) < self.similarity_threshold:
                continue
            
            chunk_tokens = result['metadata'].get('token_count', 100)
            
            if total_tokens + chunk_tokens <= self.max_context_tokens:
                selected_chunks.append(result)
                total_tokens += chunk_tokens
                
                if result['metadata'].get('has_figure_reference', False):
                    has_figure_evidence = True
            
            if len(selected_chunks) >= self.report_top_k:
                break
        
        search_strategy = "hybrid" if self.use_hybrid else "semantic"
        
        logger.info(
            f"Retrieved for '{criterion_id}': "
            f"{len(queries)} query variations, "
            f"{len(all_results)} unique chunks, "
            f"{len(selected_chunks)} selected ({search_strategy})"
        )
        
        return RetrievalResult(
            criterion_id=criterion_id,
            criterion_text=criterion_text,
            retrieved_chunks=selected_chunks,
            total_tokens=total_tokens,
            has_figure_evidence=has_figure_evidence,
            search_strategy=search_strategy,
            query_variations=queries
        )
    
    def retrieve_all(self) -> RetrievalContext:
        """Retrieve evidence for all criteria."""
        criteria_list = self.extract_criteria_list()
        
        if not criteria_list:
            logger.warning("No criteria extracted")
            all_criteria = self.vector_store.get_all_criteria()
            combined_text = '\n'.join([c['content'] for c in all_criteria])
            
            result = self.retrieve_for_criterion(combined_text[:1000], "all")
            return RetrievalContext(
                criteria_results=[result],
                report_summary="",
                total_criteria=1,
                total_tokens=result.total_tokens
            )
        
        results = []
        total_tokens = 0
        
        for criterion in criteria_list:
            result = self.retrieve_for_criterion(
                criterion['text'],
                criterion['id']
            )
            results.append(result)
            total_tokens += result.total_tokens
        
        report_summary = self._generate_report_summary()
        
        return RetrievalContext(
            criteria_results=results,
            report_summary=report_summary,
            total_criteria=len(criteria_list),
            total_tokens=total_tokens
        )
    
    def _generate_report_summary(self) -> str:
        """Generate a brief summary of the report structure."""
        general_query = "introduction methodology results discussion conclusion"
        query_embedding = self.embedder.embed_query(general_query)
        
        results = self.vector_store.query_report(
            query_embedding=query_embedding,
            n_results=10
        )
        
        sections = set()
        for result in results:
            section = result['metadata'].get('section_title', '')
            if section:
                sections.add(section)
        
        if sections:
            return "Report sections: " + ", ".join(sorted(sections))
        
        return "Report structure analysis not available"
    
    def format_context_for_llm(
        self,
        result: RetrievalResult,
        include_page_refs: bool = True
    ) -> str:
        """Format retrieved chunks into context string for LLM."""
        if not result.retrieved_chunks:
            return "No relevant evidence found in the report for this criterion."
        
        context_parts = []
        
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Build location reference
            location_parts = []
            
            if include_page_refs:
                page_start = metadata.get('page_start', '?')
                page_end = metadata.get('page_end', page_start)
                if page_start == page_end:
                    location_parts.append(f"page {page_start}")
                else:
                    location_parts.append(f"pages {page_start}-{page_end}")
            
            section_number = metadata.get('section_number', '')
            section_title = metadata.get('section_title', '')
            
            if section_number:
                location_parts.append(f"Section {section_number}")
            elif section_title:
                num_match = re.match(r'^(\d+(?:\.\d+)*)', section_title)
                if num_match:
                    location_parts.append(f"Section {num_match.group(1)}")
                else:
                    location_parts.append(f"Section: {section_title[:50]}")
            
            location_str = " / ".join(location_parts) if location_parts else ""
            location_display = f" ({location_str})" if location_str else ""
            
            # Add relevance indicator
            similarity = chunk.get('similarity', 0)
            relevance = "HIGH" if similarity > 0.5 else "MEDIUM" if similarity > 0.3 else "LOW"
            
            context_parts.append(
                f"Evidence {i}{location_display} [Relevance: {relevance}]:\n{content}"
            )
        
        header = (
            "IMPORTANT: Use ONLY the page numbers and section numbers shown below "
            "when citing evidence. Do not invent section numbers.\n\n"
            f"Search strategy: {result.search_strategy} | "
            f"Query variations used: {len(result.query_variations)}\n\n"
        )
        
        return header + "\n\n---\n\n".join(context_parts)