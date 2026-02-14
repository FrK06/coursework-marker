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


# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# UPDATED: Default to working 7B baseline model (4B models are BELOW MINIMUM and unreliable)
# Your models: mistral:7b (✅ RECOMMENDED - working baseline), gpt-oss:20b (⚠️ too strict), qwen3-vl:4b (⚠️ small), gemma3:4b (❌ not recommended)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")  # ✅ Recommended 7B baseline

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


class ModelConfig:
    """Model-specific configuration for grading behavior."""

    # Model profiles with optimized parameters
    # Format: "model_name": {"temperature": float, "max_tokens": int, "strictness_adjustment": str}
    MODEL_PROFILES = {
        # ═══════════════════════════════════════════════════════════════════════
        # YOUR MODELS (based on ollama list)
        # ═══════════════════════════════════════════════════════════════════════

        "mistral:7b": {
            "temperature": 0.1,
            "max_tokens": 1500,
            "strictness_adjustment": "balanced",  # ✅ WORKS WELL - Your baseline
            "notes": "7B params - well-tested, produces fair grades"
        },

        "gpt-oss:20b": {
            "temperature": 0.25,  # Higher temp - large model being too strict
            "max_tokens": 2000,   # Can handle longer outputs
            "strictness_adjustment": "lenient",  # ⚠️ TOO STRICT despite 20B params
            "notes": "20B params - paradoxically strict, needs lenient calibration"
        },

        "qwen3-vl:4b": {
            "temperature": 0.3,   # Higher temp for small model
            "max_tokens": 1500,
            "strictness_adjustment": "lenient",  # ⚠️ 4B is below 7B minimum
            "notes": "4B vision model - below recommended size, may be unreliable"
        },

        "gemma3:4b": {
            "temperature": 0.3,   # Higher temp for small model
            "max_tokens": 1500,
            "strictness_adjustment": "lenient",  
            "notes": "4B params - BELOW MINIMUM, known to hallucinate"
        },

        # ═══════════════════════════════════════════════════════════════════════
        # Other common models (for reference)
        # ═══════════════════════════════════════════════════════════════════════

        "mistral": {
            "temperature": 0.1,
            "max_tokens": 1500,
            "strictness_adjustment": "balanced",
        },
        "llama3:8b": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "strictness_adjustment": "lenient",
        },
        "llama3": {
            "temperature": 0.2,
            "max_tokens": 1500,
            "strictness_adjustment": "lenient",
        },
    }

    # Default fallback for unknown models
    DEFAULT_PROFILE = {
        "temperature": 0.15,  # Slightly higher than Mistral default
        "max_tokens": 1500,
        "strictness_adjustment": "lenient",  # Err on side of leniency for unknowns
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> dict:
        """Get configuration for a specific model."""
        # Try exact match first
        if model_name in cls.MODEL_PROFILES:
            return cls.MODEL_PROFILES[model_name]

        # Try prefix match (e.g., "mistral:7b-instruct" matches "mistral")
        for profile_name, config in cls.MODEL_PROFILES.items():
            if model_name.startswith(profile_name):
                return config

        # Return default
        return cls.DEFAULT_PROFILE
