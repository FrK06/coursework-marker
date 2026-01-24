"""
Coursework Marker Assistant - Source Package
"""
from .document_processing import DocxProcessor, PDFProcessor, ImageProcessor, ProcessedImage
from .chunking import SmartChunker
from .embeddings import Embedder
from .vector_store import ChromaStore
from .retrieval import Retriever
from .llm import OllamaClient
from .prompts import KSBPromptTemplates, extract_grade_from_evaluation
from .criteria import (
    KSBCriterion,
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)

__all__ = [
    "DocxProcessor",
    "PDFProcessor",
    "ImageProcessor",
    "ProcessedImage",
    "SmartChunker",
    "Embedder",
    "ChromaStore",
    "Retriever",
    "OllamaClient",
    "KSBPromptTemplates",
    "extract_grade_from_evaluation",
    "KSBCriterion",
    "get_module_criteria",
    "get_available_modules",
    "AVAILABLE_MODULES",
]
