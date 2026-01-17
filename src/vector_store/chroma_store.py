"""
ChromaDB Vector Store - Persistent vector storage with metadata filtering.

Features:
- Separate collections for criteria and report
- Metadata-based filtering (section, page, criterion_id)
- Persistent storage across sessions
- Efficient similarity search
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

import numpy as np

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB-based vector store for document chunks.
    
    Maintains separate collections for:
    - Criteria/rubric documents
    - Student reports
    
    Supports:
    - Metadata filtering (by section, page, criterion)
    - Similarity threshold filtering
    - Persistent storage
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/indexes",
        criteria_collection: str = "criteria",
        report_collection: str = "report"
    ):
        """
        Initialize the ChromaDB store.
        
        Args:
            persist_directory: Directory for persistent storage
            criteria_collection: Name for criteria collection
            report_collection: Name for report collection
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.criteria_collection_name = criteria_collection
        self.report_collection_name = report_collection
        
        # Initialize client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collections will be lazily initialized
        self._criteria_collection = None
        self._report_collection = None
        
        logger.info(f"ChromaDB initialized at {self.persist_directory}")
    
    @property
    def criteria_collection(self):
        """Get or create criteria collection."""
        if self._criteria_collection is None:
            self._criteria_collection = self.client.get_or_create_collection(
                name=self.criteria_collection_name,
                metadata={"description": "Marking criteria and rubrics"}
            )
        return self._criteria_collection
    
    @property
    def report_collection(self):
        """Get or create report collection."""
        if self._report_collection is None:
            self._report_collection = self.client.get_or_create_collection(
                name=self.report_collection_name,
                metadata={"description": "Student report chunks"}
            )
        return self._report_collection
    
    def add_criteria(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> int:
        """
        Add criteria chunks to the store.
        
        Args:
            chunks: List of chunk dicts with 'content', 'chunk_id', metadata
            embeddings: numpy array of embeddings
            
        Returns:
            Number of chunks added
        """
        return self._add_to_collection(
            self.criteria_collection,
            chunks,
            embeddings
        )
    
    def add_report(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> int:
        """
        Add report chunks to the store.
        
        Args:
            chunks: List of chunk dicts with 'content', 'chunk_id', metadata
            embeddings: numpy array of embeddings
            
        Returns:
            Number of chunks added
        """
        return self._add_to_collection(
            self.report_collection,
            chunks,
            embeddings
        )
    
    def _add_to_collection(
        self,
        collection,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> int:
        """Add chunks to a collection."""
        if not chunks:
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', str(hash(chunk.get('content', ''))))
            ids.append(chunk_id)
            documents.append(chunk.get('content', ''))
            
            # Build metadata (ChromaDB requires flat dict with simple types)
            metadata = {
                'document_type': str(chunk.get('document_type', '')),
                'page_start': int(chunk.get('page_start', 1)),
                'page_end': int(chunk.get('page_end', 1)),
                'section_title': str(chunk.get('section_title', '')),
                'chunk_type': str(chunk.get('chunk_type', 'text')),
                'has_figure_reference': bool(chunk.get('has_figure_reference', False)),
                'criterion_id': str(chunk.get('criterion_id', '')),
                'rubric_level': str(chunk.get('rubric_level', '')),
                'token_count': int(chunk.get('token_count', 0))
            }
            metadatas.append(metadata)
        
        # Convert embeddings to list
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        logger.info(f"Added {len(chunks)} chunks to {collection.name}")
        return len(chunks)
    
    def query_criteria(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filter_criterion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the criteria collection.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_criterion: Optional criterion ID to filter by
            
        Returns:
            List of matching chunks with scores
        """
        where_filter = None
        if filter_criterion:
            where_filter = {"criterion_id": filter_criterion}
        
        return self._query_collection(
            self.criteria_collection,
            query_embedding,
            n_results,
            where_filter
        )
    
    def query_report(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        filter_section: Optional[str] = None,
        filter_has_figures: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the report collection.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_section: Optional section title to filter by
            filter_has_figures: Optional filter for chunks with figure references
            
        Returns:
            List of matching chunks with scores
        """
        where_filter = {}
        
        if filter_section:
            where_filter["section_title"] = filter_section
        
        if filter_has_figures is not None:
            where_filter["has_figure_reference"] = filter_has_figures
        
        where_filter = where_filter if where_filter else None
        
        return self._query_collection(
            self.report_collection,
            query_embedding,
            n_results,
            where_filter
        )
    
    def _query_collection(
        self,
        collection,
        query_embedding: np.ndarray,
        n_results: int,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query a collection and return results."""
        # Ensure embedding is a list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Query
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    # Convert distance to similarity (ChromaDB uses L2 by default)
                    'similarity': 1 / (1 + results['distances'][0][i])
                })
        
        return formatted
    
    def get_all_criteria(self) -> List[Dict[str, Any]]:
        """Get all criteria chunks (for overview)."""
        result = self.criteria_collection.get(
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if result['ids']:
            for i in range(len(result['ids'])):
                chunks.append({
                    'chunk_id': result['ids'][i],
                    'content': result['documents'][i],
                    'metadata': result['metadatas'][i]
                })
        
        return chunks
    
    def clear_criteria(self):
        """Clear all criteria data."""
        try:
            # Check if collection exists before deleting
            existing = [c.name for c in self.client.list_collections()]
            if self.criteria_collection_name in existing:
                self.client.delete_collection(self.criteria_collection_name)
                logger.info("Criteria collection cleared")
            self._criteria_collection = None
        except Exception as e:
            logger.warning(f"Could not clear criteria collection: {e}")
            self._criteria_collection = None
    
    def clear_report(self):
        """Clear all report data."""
        try:
            # Check if collection exists before deleting
            existing = [c.name for c in self.client.list_collections()]
            if self.report_collection_name in existing:
                self.client.delete_collection(self.report_collection_name)
                logger.info("Report collection cleared")
            self._report_collection = None
        except Exception as e:
            logger.warning(f"Could not clear report collection: {e}")
            self._report_collection = None
    
    def clear_all(self):
        """Clear all data."""
        self.clear_criteria()
        self.clear_report()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        return {
            'criteria_count': self.criteria_collection.count(),
            'report_count': self.report_collection.count(),
            'persist_directory': str(self.persist_directory)
        }
