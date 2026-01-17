"""
Coursework Marker Assistant - Source Package
"""
from .document_processing import DocxProcessor, PDFProcessor, ImageExtractor
from .chunking import SmartChunker
from .embeddings import Embedder
from .vector_store import ChromaStore
from .retrieval import Retriever
from .llm import OllamaClient
from .prompts import PromptTemplates
from .criteria import (
    KSBRubricParser, 
    KSBCriterion, 
    get_default_ksb_criteria,
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)

__all__ = [
    "DocxProcessor",
    "PDFProcessor", 
    "ImageExtractor",
    "SmartChunker",
    "Embedder",
    "ChromaStore",
    "Retriever",
    "OllamaClient",
    "PromptTemplates",
    "KSBRubricParser",
    "KSBCriterion",
    "get_default_ksb_criteria",
    "get_module_criteria",
    "get_available_modules",
    "AVAILABLE_MODULES",
]
