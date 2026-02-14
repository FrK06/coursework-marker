"""
Retriever - Advanced retrieval with hybrid search and query expansion.
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
    search_strategy: str = "hybrid"
    query_variations: List[str] = field(default_factory=list)


class BM25:
    """BM25 implementation for keyword-based search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = defaultdict(int)
        self.term_freqs = []
        self.num_docs = 0
        self.documents = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        return re.findall(r'\b[a-z0-9]+\b', text)
    
    def fit(self, documents: List[str]):
        """Fit BM25 on a corpus of documents."""
        self.documents = documents
        self.num_docs = len(documents)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.term_freqs.append(dict(tf))
            
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
        """Calculate BM25 score for a query against a document."""
        query_tokens = self._tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0
        for token in query_tokens:
            if token not in doc_tf:
                continue
            
            tf = doc_tf[token]
            idf = self._idf(token)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents matching query."""
        scores = []
        for idx in range(self.num_docs):
            score = self.score(query, idx)
            if score > 0:
                scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class QueryExpander:
    """Expands queries for better retrieval coverage."""
    
    TERM_EXPANSIONS = {
        # General ML/AI terms
        'ml': ['machine learning', 'model', 'algorithm'],
        'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
        'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'deployment'],
        'gpu': ['graphics processing unit', 'cuda', 'accelerator', 'compute'],
        'cpu': ['processor', 'compute', 'central processing unit'],
        'api': ['interface', 'endpoint', 'service'],
        'gdpr': ['data protection', 'privacy', 'compliance', 'personal data'],
        'data': ['dataset', 'information', 'records'],
        'storage': ['database', 's3', 'blob', 'warehouse', 'lake'],
        'etl': ['pipeline', 'extraction', 'transformation', 'loading'],
        'training': ['learning', 'fitting', 'optimization'],
        'inference': ['prediction', 'serving', 'deployment'],
        'model': ['algorithm', 'classifier', 'neural network'],
        'accuracy': ['performance', 'metrics', 'evaluation', 'precision', 'recall'],
        'benchmark': ['comparison', 'evaluation', 'performance test'],
        'architecture': ['design', 'structure', 'system', 'infrastructure'],
        'scalability': ['scaling', 'elasticity', 'auto-scaling'],
        'security': ['encryption', 'authentication', 'access control', 'iam'],
        'requirements': ['specifications', 'needs', 'criteria'],
        'stakeholder': ['user', 'customer', 'client', 'business'],
        'feasibility': ['viability', 'assessment', 'evaluation'],
        'cost': ['pricing', 'budget', 'expense', 'economics'],

        # DSP-specific terms (CRITICAL: missing these caused 12 REFERRALs on strong report!)
        'hypothesis': ['null hypothesis', 'alternative hypothesis', 'H0', 'H1', 'statistical test', 'significance'],
        'eda': ['exploratory data analysis', 'visualization', 'summary statistics', 'distribution'],
        'confidence': ['confidence interval', 'CI', 'margin of error', 'effect size'],
        'effect': ['effect size', 'Cohen d', 'practical significance', 'magnitude'],
        'p-value': ['significance', 'alpha', 'threshold', 'reject null'],
        'anonymisation': ['pseudonymisation', 'de-identification', 'data masking', 'synthetic data'],
        'dpia': ['data protection impact assessment', 'risk assessment', 'privacy'],
        'dashboard': ['Power BI', 'visualization', 'report', 'chart', 'graph', 'figure'],
        'lakehouse': ['data lake', 'warehouse', 'medallion', 'bronze silver gold', 'delta lake'],
        'telemetry': ['event log', 'instrumentation', 'monitoring', 'observability'],
        'bias': ['selection bias', 'measurement bias', 'confounding', 'fairness'],
        'retention': ['data retention', 'storage limitation', 'deletion policy', 'lifecycle'],
    }
    
    KSB_CONTEXT = {
        # ═══════════════════════════════════════════════════════════════════════
        # MLCC Module (Machine Learning with Cloud Computing)
        # ═══════════════════════════════════════════════════════════════════════
        'K1': ['business objective', 'ml methodology', 'problem framing', 'AI methodology', 'business value'],
        'K2': ['storage', 'processing', 'organizational impact', 'data pipeline', 'infrastructure'],
        'K16': ['high performance', 'gpu', 'cpu', 'architecture', 'compute'],
        'K18': ['programming', 'data engineering', 'pipeline', 'code'],
        'K19': ['statistical', 'ml principles', 'evaluation', 'overfitting'],
        'K25': ['ml libraries', 'pytorch', 'tensorflow', 'keras'],
        'S15': ['deployment', 'service', 'platform', 'poc', 'engineering', 'artefact', 'build'],
        'S16': ['requirements', 'data management', 'cloud', 'governance'],
        'S19': ['scalable', 'infrastructure', 'benchmarking'],
        'S23': ['best practice', 'dissemination', 'documentation'],
        'B5': ['cpd', 'professional development', 'learning'],

        # ═══════════════════════════════════════════════════════════════════════
        # DSP Module (Data Science Principles) - CRITICAL: Was 100% missing!
        # ═══════════════════════════════════════════════════════════════════════

        # DSP Knowledge KSBs
        # Note: K2 is shared with MLCC but extended above
        'K5': ['analysis', 'research', 'EDA', 'exploratory', 'visualization', 'insight', 'methodology', 'business need'],
        'K15': ['engineering', 'artefact', 'build', 'dashboard', 'reproducibility', 'versioning', 'modular', 'screenshot'],
        'K20': ['data landscape', 'data types', 'collection', 'sources', 'schema', 'metadata', 'instrumentation', 'telemetry'],
        'K22': ['hypothesis', 'statistical test', 't-test', 'p-value', 'null hypothesis', 'alternative hypothesis', 'significance', 'confidence interval', 'Welch', 'ANOVA', 'chi-squared'],
        'K24': ['GDPR', 'data protection', 'privacy', 'ethics', 'compliance', 'regulation', 'legal'],
        'K26': ['scientific method', 'experimental design', 'variable', 'control', 'bias', 'confounding', 'validity'],
        'K27': ['data product', 'specification', 'requirements', 'stakeholder', 'design', 'lifecycle'],

        # DSP Skills KSBs
        'S1': ['storage architecture', 'data architecture', 'design', 'scalability', 'security', 'modelling'],
        'S9': ['data manipulation', 'analysis', 'visualisation', 'filtering', 'aggregation', 'transformation', 'chart', 'graph', 'figure', 'dashboard', 'Power BI', 'matplotlib'],
        'S10': ['dataset selection', 'methodology', 'business problem', 'rationale', 'justification', 'constraints'],
        'S13': ['resource', 'architecture', 'platform', 'tool selection', 'Power BI', 'Python', 'SQL', 'cloud'],
        'S17': ['data curation', 'data quality', 'validation', 'cleaning', 'duplicate', 'missing values', 'schema check', 'data dictionary'],
        'S18': ['monitoring', 'performance', 'dashboard', 'infrastructure diagram', 'pipeline health', 'freshness', 'data lineage'],
        'S21': ['uncertainty', 'confidence interval', 'error bars', 'effect size', 'Cohen', 'variability', 'sampling error', 'measurement error', 'sensitivity', 'limitation'],
        'S22': ['EDA', 'exploratory data analysis', 'hypothesis testing', 'business decision', 'statistical test', 'strategy', 'recommendation', 'quantify impact'],
        'S26': ['AI technique', 'DS technique', 'visual analytics', 'segmentation', 'forecasting', 'anomaly detection', 'evaluation metric'],

        # DSP Behaviours KSBs
        'B3': ['integrity', 'ethics', 'GDPR', 'data protection', 'anonymisation', 'pseudonymisation', 'access control', 'retention', 'DPIA', 'compliance', 'lawful basis'],
        'B7': ['best practice', 'sharing', 'documentation', 'reflection', 'lessons learned', 'template', 'show-and-tell', 'dissemination'],

        # ═══════════════════════════════════════════════════════════════════════
        # AIDI Module (AI-Driven Innovation) - CRITICAL: Was 100% missing!
        # ═══════════════════════════════════════════════════════════════════════

        # AIDI Knowledge KSBs
        # 'K1': ['business objective', 'AI methodology', 'ML methodology', 'problem framing', 'business value'],  # Duplicate K1 - shared with MLCC
        'K4': ['data extraction', 'data linkage', 'multiple systems', 'integration', 'schema', 'join', 'API', 'data lineage'],
        # 'K5': ['analysis', 'research', 'design', 'deploy', 'user needs', 'domain research'],  # Duplicate K5 - shared with DSP
        'K6': ['iterative', 'incremental', 'agile', 'MVP', 'delivery', 'sprint', 'feedback loop'],
        'K8': ['policy', 'standard', 'guideline', 'organisational', 'compliance', 'SDLC'],
        'K9': ['legal', 'ethical', 'regulatory', 'GDPR', 'DPIA', 'lawful basis', 'privacy', 'IP', 'licensing'],
        'K11': ['AI roles', 'DS roles', 'DE roles', 'industry', 'society', 'lifecycle', 'governance'],
        'K12': ['social context', 'ethical issues', 'automation', 'misuse', 'harm', 'bias', 'fairness', 'transparency', 'HITL'],
        'K21': ['team support', 'collaboration', 'stakeholder', 'handover', 'runbook', 'adoption'],
        # 'K24': ['GDPR', 'data protection', 'privacy', 'ethics', 'compliance', 'regulation'],  # Duplicate K24 - shared with DSP
        'K29': ['accessibility', 'inclusive design', 'WCAG', 'usability', 'assistive technology'],

        # AIDI Skills KSBs
        'S3': ['requirements', 'elicitation', 'user stories', 'acceptance criteria', 'functional', 'non-functional'],
        'S5': ['stakeholder', 'communication', 'presentation', 'reporting', 'engagement'],
        'S6': ['direction', 'roadmap', 'innovation', 'trends', 'strategy'],
        'S25': ['engineering practice', 'version control', 'testing', 'modular code', 'CI/CD', 'reproducibility'],
        # 'S26': ['AI technique', 'DS technique', 'complex problem', 'evaluation', 'benchmark', 'metric'],  # Duplicate S26 - shared with DSP

        # AIDI Behaviours KSBs
        # 'B3': ['integrity', 'ethics', 'GDPR', 'data protection', 'anonymisation', 'pseudonymisation', 'access control', 'DPIA'],  # Duplicate B3 - shared with DSP
        'B4': ['initiative', 'responsibility', 'challenge', 'ownership', 'problem-solving', 'iteration'],
        'B8': ['trends', 'innovation', 'literature', 'references', 'research', 'industry'],
    }
    
    def expand_query(self, query: str, ksb_code: str = "") -> List[str]:
        """Expand a query into multiple variations."""
        queries = [query]
        
        query_lower = query.lower()
        words = set(re.findall(r'\b[a-z]+\b', query_lower))
        
        expansion_terms = set()
        for word in words:
            if word in self.TERM_EXPANSIONS:
                expansion_terms.update(self.TERM_EXPANSIONS[word])
        
        if expansion_terms:
            for term in list(expansion_terms)[:5]:
                if term not in query_lower:
                    queries.append(term)
            
            combined = f"{query} {' '.join(list(expansion_terms)[:3])}"
            queries.append(combined)
        
        if ksb_code:
            ksb_upper = ksb_code.upper()
            for code in ksb_upper.split(','):
                code = code.strip()
                if code in self.KSB_CONTEXT:
                    for context_term in self.KSB_CONTEXT[code][:3]:
                        if context_term not in query_lower:
                            queries.append(context_term)
        
        quoted = re.findall(r'"([^"]+)"', query)
        queries.extend(quoted)
        
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 2:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:10]
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for focused retrieval."""
        concepts = []
        
        patterns = [
            r'\b(?:using|with|via|through)\s+([A-Z][a-zA-Z0-9\s]+)',
            r'\b([A-Z]{2,})\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:GB|MB|TB|ms|seconds|hours)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            concepts.extend([m.strip() for m in matches if len(m.strip()) > 1])
        
        return list(set(concepts))[:5]


class Retriever:
    """Advanced retriever with hybrid search capabilities."""
    
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
        keyword_weight: float = 0.4,
        hybrid_threshold: float = 0.01
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.criteria_top_k = criteria_top_k
        self.report_top_k = report_top_k
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold
        self.hybrid_threshold = hybrid_threshold
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        self.query_expander = QueryExpander()
        
        self._bm25_index = None
        self._bm25_docs = None
    
    def _build_bm25_index(self) -> Optional[BM25]:
        """Build BM25 index from report chunks."""
        try:
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
        self, result_lists: List[List[Tuple[str, float]]], k: int = 60
    ) -> List[Tuple[str, float]]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        
        for result_list in result_lists:
            for rank, (doc_id, _) in enumerate(result_list):
                scores[doc_id] += 1 / (k + rank + 1)
        
        combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return combined
    
    def extract_criteria_list(self) -> List[Dict[str, str]]:
        """Extract the list of criteria from stored documents."""
        all_criteria = self.vector_store.get_all_criteria()
        
        if not all_criteria:
            logger.warning("No criteria found in vector store")
            return []
        
        criteria_map = {}
        
        for chunk in all_criteria:
            criterion_id = chunk['metadata'].get('criterion_id', '')
            
            if criterion_id:
                if criterion_id not in criteria_map:
                    criteria_map[criterion_id] = {'id': criterion_id, 'chunks': []}
                criteria_map[criterion_id]['chunks'].append(chunk['content'])
            else:
                if 'general' not in criteria_map:
                    criteria_map['general'] = {'id': 'general', 'chunks': []}
                criteria_map['general']['chunks'].append(chunk['content'])
        
        criteria_list = []
        for _, data in sorted(criteria_map.items()):
            criteria_list.append({
                'id': data['id'],
                'text': '\n\n'.join(data['chunks'])
            })
        
        return criteria_list
    
    def _semantic_search(self, query: str, n_results: int) -> List[Tuple[str, float]]:
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
    
    def _keyword_search(self, query: str, n_results: int) -> List[Tuple[str, float]]:
        """Perform keyword search using BM25."""
        if self._bm25_index is None:
            self._bm25_index = self._build_bm25_index()
        
        if self._bm25_index is None or self._bm25_docs is None:
            return []
        
        bm25_results = self._bm25_index.search(query, top_k=n_results)
        
        results = []
        for doc_idx, score in bm25_results:
            doc_id = self._bm25_docs['ids'][doc_idx]
            normalized_score = min(score / 10, 1.0)
            results.append((doc_id, normalized_score))
        
        return results
    
    def _hybrid_search(
        self, queries: List[str], n_results: int
    ) -> Dict[str, Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword methods."""
        all_semantic_results = []
        all_keyword_results = []
        
        for query in queries:
            semantic_results = self._semantic_search(query, n_results)
            all_semantic_results.append(semantic_results)
            
            if self.use_hybrid:
                keyword_results = self._keyword_search(query, n_results)
                all_keyword_results.append(keyword_results)
        
        if self.use_hybrid and all_keyword_results:
            semantic_combined = self._reciprocal_rank_fusion(all_semantic_results)
            keyword_combined = self._reciprocal_rank_fusion(all_keyword_results)
            
            final_scores = defaultdict(float)
            
            for doc_id, score in semantic_combined:
                final_scores[doc_id] += score * self.semantic_weight
            
            for doc_id, score in keyword_combined:
                final_scores[doc_id] += score * self.keyword_weight
            
            ranked_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked_ids = self._reciprocal_rank_fusion(all_semantic_results)
        
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
        self, criterion_text: str, criterion_id: str = ""
    ) -> RetrievalResult:
        """Retrieve relevant report chunks for a single criterion."""
        queries = self.query_expander.expand_query(criterion_text[:500], criterion_id)
        concepts = self.query_expander.extract_key_concepts(criterion_text)
        queries.extend(concepts)
        
        all_results = self._hybrid_search(queries, self.report_top_k)
        
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )
        
        effective_threshold = self.hybrid_threshold if self.use_hybrid else self.similarity_threshold
        
        selected_chunks = []
        total_tokens = 0
        has_figure_evidence = False
        
        for result in sorted_results:
            if result.get('similarity', 0) < effective_threshold:
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
    
    def format_context_for_llm(
        self, result: RetrievalResult, include_page_refs: bool = True, pages_are_accurate: bool = False
    ) -> str:
        """Format retrieved chunks into context string for LLM.
        
        Args:
            result: RetrievalResult containing chunks
            include_page_refs: Whether to include page references
            pages_are_accurate: If True (PDF), show page numbers. If False (DOCX), hide them.
        """
        if not result.retrieved_chunks:
            return "No relevant evidence found in the report for this criterion."
        
        context_parts = []
        
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Build location reference - PRIORITIZE SECTIONS
            location_parts = []
            
            # First: Section number (reliable for both DOCX and PDF)
            section_number = metadata.get('section_number', '')
            section_title = metadata.get('section_title', '')
            
            if section_number:
                location_parts.append(f"Section {section_number}")
            elif section_title:
                # Try to extract section number from title
                num_match = re.match(r'^(\d+(?:\.\d+)*)', section_title)
                if num_match:
                    location_parts.append(f"Section {num_match.group(1)}")
                else:
                    # Use truncated title as fallback
                    clean_title = section_title[:40].strip()
                    if clean_title:
                        location_parts.append(f"Section: {clean_title}")
            
            # Second: Page numbers - ONLY if accurate (PDF)
            if include_page_refs and pages_are_accurate:
                page_start = metadata.get('page_start', 0)
                page_end = metadata.get('page_end', page_start)
                if page_start and page_start > 0:
                    if page_start == page_end:
                        location_parts.append(f"page {page_start}")
                    else:
                        location_parts.append(f"pages {page_start}-{page_end}")
            
            # Add chunk index as backup reference
            chunk_index = metadata.get('chunk_index', i-1)
            if not location_parts:
                location_parts.append(f"Chunk {chunk_index + 1}")
            
            location_str = " / ".join(location_parts)
            location_display = f" ({location_str})"
            
            similarity = chunk.get('similarity', 0)
            relevance = "HIGH" if similarity > 0.5 else "MEDIUM" if similarity > 0.3 else "LOW"
            
            context_parts.append(
                f"Evidence {i}{location_display} [Relevance: {relevance}]:\n{content}"
            )
        
        # Build header based on document type
        if pages_are_accurate:
            header = (
                "CITATION RULES:\n"
                "- Cite using Section AND page numbers shown below\n"
                "- Only cite locations shown in the evidence headers\n"
                "- Do NOT invent section or page numbers\n\n"
                f"Search strategy: {result.search_strategy} | "
                f"Query variations used: {len(result.query_variations)}\n\n"
            )
        else:
            header = (
                "CITATION RULES:\n"
                "- Cite using SECTION numbers only (e.g., 'Section 3', 'Section 5')\n"
                "- Do NOT cite page numbers - they are not available for this document\n"
                "- Only cite sections shown in the evidence headers below\n"
                "- Do NOT invent section numbers\n\n"
                f"Search strategy: {result.search_strategy} | "
                f"Query variations used: {len(result.query_variations)}\n\n"
            )
        
        return header + "\n\n---\n\n".join(context_parts)
