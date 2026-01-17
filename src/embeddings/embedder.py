"""
Embedder - Local embedding generation using sentence-transformers.

Uses all-MiniLM-L6-v2 by default:
- 384 dimensions
- ~80MB model size
- Optimized for CPU
- Excellent for semantic similarity
"""
from typing import List, Union, Optional                                   #"Optional" is not accessed
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
    - Fast inference on CPU
    - Good quality for semantic search
    - 384-dimensional embeddings
    
    Alternative models for different trade-offs:
    - all-mpnet-base-v2: Higher quality, slower (768 dims)
    - paraphrase-MiniLM-L3-v2: Faster, lower quality (384 dims)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for embedding generation
        """
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
        
        # Generate embeddings
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
        """
        Embed a list of document chunks.
        
        Convenience method with progress bar enabled by default.
        """
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
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def most_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar documents to a query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return
            
        Returns:
            List of (index, score, text) tuples, sorted by similarity
        """
        query_emb = self.embed_query(query)
        doc_embs = self.embed_documents(documents, show_progress=False)
        
        similarities = self.similarity(query_emb, doc_embs)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), float(similarities[idx]), documents[idx])
            for idx in top_indices
        ]
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_sequence_length': self.model.max_seq_length
        }


class CachedEmbedder(Embedder):
    """
    Embedder with caching to avoid re-computing embeddings.
    
    Useful when the same documents are queried multiple times.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        normalize: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """Embed with optional caching."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not use_cache:
            return super().embed(texts, show_progress, normalize)
        
        # Check cache
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = super().embed(
                uncached_texts, 
                show_progress, 
                normalize
            )
            
            # Update cache and results
            for j, (text, emb) in enumerate(zip(uncached_texts, new_embeddings)):
                cache_key = hash(text)
                self._cache[cache_key] = emb
                results.append((uncached_indices[j], emb))
        
        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in results])
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
    
    def cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)
