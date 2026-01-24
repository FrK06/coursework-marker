"""
Configuration settings for the Coursework Marker Assistant.
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
OLLAMA_TIMEOUT = 120

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# Vector store settings
CHROMA_PERSIST_DIR = str(INDEX_DIR)


class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    CRITERIA_TOP_K = 5
    REPORT_TOP_K = 8
    MAX_CONTEXT_TOKENS = 3000
    SIMILARITY_THRESHOLD = 0.2
    USE_HYBRID_SEARCH = True
    SEMANTIC_WEIGHT = 0.6
    KEYWORD_WEIGHT = 0.4
    USE_QUERY_EXPANSION = True
    MAX_QUERY_VARIATIONS = 10
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60


class LLMConfig:
    """Configuration for LLM generation."""
    EXTRACTION_TEMPERATURE = 0.1
    EVALUATION_TEMPERATURE = 0.3
    SUMMARY_TEMPERATURE = 0.4
    MAX_OUTPUT_TOKENS = 1500
    MAX_RETRIES = 3
    RETRY_DELAY = 2
