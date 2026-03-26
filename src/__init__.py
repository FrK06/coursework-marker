"""
Coursework Marker Assistant - Source Package.

Includes the current LangGraph assessment pipeline and the legacy three-agent system.
"""
from .document_processing import DocxProcessor, PDFProcessor, ImageProcessor, ProcessedImage
from .chunking import SmartChunker
from .embeddings import Embedder
from .vector_store import ChromaStore
from .llm import OllamaClient
from .criteria import (
    KSBCriterion,
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)

# Legacy three-agent system (still exported for compatibility)
from .agents import (
    AgentOrchestrator,
    AnalysisAgent,
    ScoringAgent,
    FeedbackAgent,
    create_agent_system
)

__all__ = [
    # Document processing
    "DocxProcessor",
    "PDFProcessor",
    "ImageProcessor",
    "ProcessedImage",

    # Processing
    "SmartChunker",
    "Embedder",
    "ChromaStore",
    "OllamaClient",

    # Criteria
    "KSBCriterion",
    "get_module_criteria",
    "get_available_modules",
    "AVAILABLE_MODULES",

    # Agents
    "AgentOrchestrator",
    "AnalysisAgent",
    "ScoringAgent",
    "FeedbackAgent",
    "create_agent_system",
]
