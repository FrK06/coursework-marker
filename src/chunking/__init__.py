"""
Chunking Module - Smart document chunking with semantic boundaries.

Features:
- Semantic boundary detection
- Sliding window with smart overlap
- Keyword extraction for hybrid search
- Section-aware chunking
"""
from .chunker import SmartChunker, TextChunk

__all__ = ["SmartChunker", "TextChunk"]