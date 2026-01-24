"""
Embedder - Local embedding generation using sentence-transformers.

Uses all-MiniLM-L6-v2 by default:
- 384 dimensions
- ~80MB model size
- Optimized for CPU
"""
from typing import List, Union
import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates embeddings for text using local sentence-transformers models.
    
    Default model: all-MiniLM-L6-v2
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar for large batches
            normalize: L2 normalize embeddings (recommended for similarity)
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Embed a list of document chunks."""
        return self.embed(documents, show_progress=show_progress)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Returns 1D array of shape (embedding_dim,)
        """
        embedding = self.embed([query], show_progress=False)
        return embedding[0]
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Shape (embedding_dim,)
            document_embeddings: Shape (n_docs, embedding_dim)
            
        Returns:
            Similarity scores of shape (n_docs,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        return similarities
