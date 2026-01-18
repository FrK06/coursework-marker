"""
Vector Store Module - Enhanced ChromaDB storage with keyword support.

Features:
- Keyword metadata for hybrid search
- Section-aware filtering
- Batch operations
- Improved error handling
"""
from .chroma_store import ChromaStore

__all__ = ["ChromaStore"]