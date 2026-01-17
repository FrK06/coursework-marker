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
OLLAMA_TIMEOUT = 120  # seconds - longer timeout for CPU inference

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# Chunking settings
class ChunkingConfig:
    # Criteria document chunking (smaller chunks for precise matching)
    CRITERIA_CHUNK_SIZE = 400  # tokens
    CRITERIA_CHUNK_OVERLAP = 50
    
    # Student report chunking (larger chunks for context preservation)
    REPORT_CHUNK_SIZE = 700  # tokens
    REPORT_CHUNK_OVERLAP = 100
    
    # Separators for splitting (in priority order)
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Vector store settings
CHROMA_PERSIST_DIR = str(INDEX_DIR)
CRITERIA_COLLECTION_NAME = "criteria_collection"
REPORT_COLLECTION_NAME = "report_collection"

# Retrieval settings
class RetrievalConfig:
    # Number of chunks to retrieve per query
    CRITERIA_TOP_K = 5
    REPORT_TOP_K = 12  # Increased to capture content spread across sections
    
    # Maximum context tokens per criterion evaluation
    MAX_CONTEXT_TOKENS = 3500  # More context for better assessment
    
    # Similarity threshold (lower to avoid missing relevant content)
    SIMILARITY_THRESHOLD = 0.15
    
    # Multi-query expansion for better recall
    USE_MULTI_QUERY = True

# LLM settings
class LLMConfig:
    # Temperature for different tasks
    EXTRACTION_TEMPERATURE = 0.1  # Low for structured extraction
    EVALUATION_TEMPERATURE = 0.3  # Slightly higher for nuanced feedback
    SUMMARY_TEMPERATURE = 0.4     # Higher for creative synthesis
    
    # Token limits
    MAX_OUTPUT_TOKENS = 1024
    
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
