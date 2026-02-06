"""
Coursework Marker Assistant - Source Package

Now with three-agent agentic assessment system.
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

# NEW: Agentic system
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
    "Retriever",
    "OllamaClient",
    
    # Prompts
    "KSBPromptTemplates",
    "extract_grade_from_evaluation",
    
    # Criteria
    "KSBCriterion",
    "get_module_criteria",
    "get_available_modules",
    "AVAILABLE_MODULES",
    
    # NEW: Agents
    "AgentOrchestrator",
    "AnalysisAgent",
    "ScoringAgent",
    "FeedbackAgent",
    "create_agent_system",
]
