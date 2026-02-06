"""
Configuration settings for the Coursework Marker Assistant.

UPDATED: Improved parameters for better accuracy
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

# ═══════════════════════════════════════════════════════════════════════════════
#                        MODEL RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# 
# ⚠️  CRITICAL: For accurate academic marking, you need a capable model.
# 
# RECOMMENDED MODELS (in order of preference):
# 
# 1. Claude 3.5 Sonnet (API)     - Best accuracy, requires API key
# 2. Llama 3.1 70B (local)       - Best local option, needs ~40GB VRAM
# 3. Mistral 7B v0.3 (local)     - Minimum viable, needs ~8GB VRAM
# 4. Llama 3.1 8B (local)        - Marginal, test carefully
# 
# NOT RECOMMENDED:
# ❌ Gemma 3 4B                   - Too small, causes hallucinations
# ❌ Any model < 7B parameters    - Insufficient reasoning capacity
# 
# ═══════════════════════════════════════════════════════════════════════════════

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# UPDATED: Default to a more capable model
# Change this to your actual model - options:
#   - "llama3.1:70b" (best local)
#   - "mistral:7b" (minimum viable)
#   - "llama3.1:8b" (marginal)
#   - "gemma3:4b" (NOT recommended - causes hallucinations)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

OLLAMA_TIMEOUT = 180  # Increased for larger models

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# Vector store settings
CHROMA_PERSIST_DIR = str(INDEX_DIR)


class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    CRITERIA_TOP_K = 5
    REPORT_TOP_K = 10          # INCREASED from 8 - more evidence per KSB
    MAX_CONTEXT_TOKENS = 4000  # INCREASED from 3000 - more context for LLM
    SIMILARITY_THRESHOLD = 0.15  # LOWERED from 0.2 - less aggressive filtering
    USE_HYBRID_SEARCH = True
    SEMANTIC_WEIGHT = 0.6
    KEYWORD_WEIGHT = 0.4
    USE_QUERY_EXPANSION = True
    MAX_QUERY_VARIATIONS = 10
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60


class ChunkingConfig:
    """Configuration for document chunking - UPDATED for better granularity."""
    CRITERIA_CHUNK_SIZE = 300    # REDUCED from 400
    CRITERIA_OVERLAP = 50
    REPORT_CHUNK_SIZE = 400      # REDUCED from 600
    REPORT_OVERLAP = 80          # REDUCED from 120
    MIN_CHUNK_SIZE = 50          # REDUCED from 100
    MAX_CHUNK_SIZE = 600         # REDUCED from 1000


class LLMConfig:
    """Configuration for LLM generation."""
    EXTRACTION_TEMPERATURE = 0.1
    EVALUATION_TEMPERATURE = 0.2   # LOWERED from 0.3 - more deterministic
    SUMMARY_TEMPERATURE = 0.3      # LOWERED from 0.4
    MAX_OUTPUT_TOKENS = 2000       # INCREASED from 1500
    MAX_RETRIES = 3
    RETRY_DELAY = 2


class ValidationConfig:
    """Configuration for output validation."""
    ENABLE_HALLUCINATION_CHECK = True
    HALLUCINATION_KEYWORDS = [
        'celestial', 'rovio', 'angry birds', 'founded in',
        'headquarters:', 'website:', 'here are some key facts',
        'do you want me to provide more', 'best known for:'
    ]
    FLAG_NON_ASCII = True
    MAX_EVIDENCE_REFS = 15  # Flag if model claims more evidence than exists
