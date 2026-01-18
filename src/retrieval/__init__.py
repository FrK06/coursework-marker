"""
Retrieval Module - Advanced hybrid search with query expansion.

Features:
- Hybrid search (BM25 + semantic)
- Multi-query expansion
- Reciprocal Rank Fusion
- Section-aware retrieval
"""
from .retriever import Retriever, RetrievalResult, RetrievalContext, BM25, QueryExpander

__all__ = ["Retriever", "RetrievalResult", "RetrievalContext", "BM25", "QueryExpander"]