"""
Configuration settings for the Coursework Marker Assistant.

Enhanced with hybrid search and improved retrieval settings.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CRITERIA_DIR = DATA_DIR / "criteria"
REPORTS_DIR = DATA_DIR / "reports"
INDEX_DIR = DATA_DIR / "indexes"

# Ensure directories exist
for dir_path in [CRITERIA_DIR, REPORTS_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = 120  # seconds - longer timeout for CPU inference

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# Chunking settings (IMPROVED)
class ChunkingConfig:
    # Criteria document chunking
    CRITERIA_CHUNK_SIZE = 400  # tokens
    CRITERIA_CHUNK_OVERLAP = 50
    
    # Student report chunking (larger for better context)
    REPORT_CHUNK_SIZE = 600  # tokens
    REPORT_CHUNK_OVERLAP = 120  # More overlap for continuity
    
    # Size limits
    MIN_CHUNK_SIZE = 100  # Avoid tiny chunks
    MAX_CHUNK_SIZE = 1000  # Hard limit
    
    # Separators for splitting (in priority order)
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Vector store settings
CHROMA_PERSIST_DIR = str(INDEX_DIR)
CRITERIA_COLLECTION_NAME = "criteria_collection"
REPORT_COLLECTION_NAME = "report_collection"

# Retrieval settings (IMPROVED)
class RetrievalConfig:
    # Number of chunks to retrieve per query
    CRITERIA_TOP_K = 5
    REPORT_TOP_K = 8  # Balanced for quality vs context size
    
    # Maximum context tokens per criterion evaluation
    MAX_CONTEXT_TOKENS = 3000
    
    # Similarity threshold (lower to capture more relevant content)
    SIMILARITY_THRESHOLD = 0.2
    
    # === NEW: Hybrid Search Settings ===
    USE_HYBRID_SEARCH = True  # Enable hybrid (BM25 + semantic)
    SEMANTIC_WEIGHT = 0.6     # Weight for semantic similarity
    KEYWORD_WEIGHT = 0.4      # Weight for BM25 keyword matching
    
    # Query expansion
    USE_QUERY_EXPANSION = True
    MAX_QUERY_VARIATIONS = 10
    
    # BM25 parameters
    BM25_K1 = 1.5  # Term frequency saturation
    BM25_B = 0.75  # Document length normalization
    
    # Reciprocal Rank Fusion
    RRF_K = 60  # Ranking constant (higher = less weight to top results)

# LLM settings
class LLMConfig:
    # Temperature for different tasks
    EXTRACTION_TEMPERATURE = 0.1  # Low for structured extraction
    EVALUATION_TEMPERATURE = 0.3  # Slightly higher for nuanced feedback
    SUMMARY_TEMPERATURE = 0.4     # Higher for creative synthesis
    
    # Token limits
    MAX_OUTPUT_TOKENS = 1500  # Increased for detailed feedback
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

# Vision settings (for figure analysis)
class VisionConfig:
    ENABLE_VISION = True
    MAX_IMAGES_PER_EVALUATION = 3
    IMAGE_MAX_SIZE = (800, 800)  # Resize large images for performance

# Output formatting
class OutputConfig:
    # Sections to include in feedback
    INCLUDE_EVIDENCE = True
    INCLUDE_GAPS = True
    INCLUDE_IMPROVEMENTS = True
    INCLUDE_EXAMPLE_REWRITES = True
    
    # Tone settings
    FEEDBACK_TONE = "constructive"  # Options: constructive, critical, balanced

# === NEW: Search Strategy Presets ===
class SearchPresets:
    """Pre-configured search strategies for different use cases."""
    
    BALANCED = {
        'use_hybrid': True,
        'semantic_weight': 0.6,
        'keyword_weight': 0.4,
        'similarity_threshold': 0.2,
        'report_top_k': 8
    }
    
    SEMANTIC_HEAVY = {
        'use_hybrid': True,
        'semantic_weight': 0.8,
        'keyword_weight': 0.2,
        'similarity_threshold': 0.25,
        'report_top_k': 6
    }
    
    KEYWORD_HEAVY = {
        'use_hybrid': True,
        'semantic_weight': 0.4,
        'keyword_weight': 0.6,
        'similarity_threshold': 0.15,
        'report_top_k': 10
    }
    
    SEMANTIC_ONLY = {
        'use_hybrid': False,
        'semantic_weight': 1.0,
        'keyword_weight': 0.0,
        'similarity_threshold': 0.3,
        'report_top_k': 6
    }
    
    HIGH_RECALL = {
        'use_hybrid': True,
        'semantic_weight': 0.5,
        'keyword_weight': 0.5,
        'similarity_threshold': 0.1,
        'report_top_k': 12
    }