"""
ChromaDB Vector Store - Storage for document embeddings.
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
    ChromaDB-based vector store.
    
    Features:
    - Separate collections for criteria and reports
    - Keyword metadata for hybrid search
    - Section-aware filtering
    - Batch operations
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/indexes_e5-base-v2",
        criteria_collection: str = "criteria",
        report_collection: str = "report"
    ):
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.criteria_collection_name = criteria_collection
        self.report_collection_name = report_collection
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
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
    
    def _prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage."""
        return {
            'document_type': str(chunk.get('document_type', '')),
            'page_start': int(chunk.get('page_start', 1)),
            'page_end': int(chunk.get('page_end', 1)),
            'section_title': str(chunk.get('section_title', ''))[:500],
            'section_number': str(chunk.get('section_number', '')),
            'chunk_type': str(chunk.get('chunk_type', 'text')),
            'has_figure_reference': bool(chunk.get('has_figure_reference', False)),
            'criterion_id': str(chunk.get('criterion_id', '')),
            'rubric_level': str(chunk.get('rubric_level', '')),
            'token_count': int(chunk.get('token_count', 0)),
            'keywords': str(chunk.get('keywords', ''))[:1000],
            'parent_section': str(chunk.get('parent_section', ''))[:200],
            'chunk_index': int(chunk.get('chunk_index', 0)),
        }
    
    def add_criteria(
        self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, batch_size: int = 100
    ) -> int:
        """Add criteria chunks to the store."""
        return self._add_to_collection(self.criteria_collection, chunks, embeddings, batch_size)
    
    def add_report(
        self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, batch_size: int = 100
    ) -> int:
        """Add report chunks to the store."""
        return self._add_to_collection(self.report_collection, chunks, embeddings, batch_size)
    
    def _add_to_collection(
        self, collection, chunks: List[Dict[str, Any]],
        embeddings: np.ndarray, batch_size: int = 100
    ) -> int:
        """Add chunks to a collection in batches."""
        if not chunks:
            return 0
        
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            ids = []
            documents = []
            metadatas = []
            
            for j, chunk in enumerate(batch_chunks):
                chunk_id = chunk.get('chunk_id', f"chunk_{i+j}_{hash(chunk.get('content', ''))}")
                ids.append(chunk_id)
                documents.append(chunk.get('content', ''))
                metadatas.append(self._prepare_metadata(chunk))
            
            embeddings_list = batch_embeddings.tolist()
            
            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings_list
                )
                total_added += len(batch_chunks)
            except Exception as e:
                logger.error(f"Error adding batch to collection: {e}")
                for j, (id_, doc, meta, emb) in enumerate(
                    zip(ids, documents, metadatas, embeddings_list)
                ):
                    try:
                        collection.add(
                            ids=[id_], documents=[doc],
                            metadatas=[meta], embeddings=[emb]
                        )
                        total_added += 1
                    except Exception as e2:
                        logger.error(f"Error adding chunk {id_}: {e2}")
        
        logger.info(f"Added {total_added} chunks to {collection.name}")
        return total_added
    
    def query_criteria(
        self, query_embedding: np.ndarray, n_results: int = 5,
        filter_criterion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the criteria collection."""
        where_filter = None
        if filter_criterion:
            where_filter = {"criterion_id": filter_criterion}
        
        return self._query_collection(self.criteria_collection, query_embedding, n_results, where_filter)
    
    def query_report(
        self, query_embedding: np.ndarray, n_results: int = 5,
        filter_section: Optional[str] = None,
        filter_has_figures: Optional[bool] = None,
        filter_page_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Query the report collection with optional filters."""
        where_clauses = []
        
        if filter_section:
            where_clauses.append({"section_title": {"$eq": filter_section}})
        if filter_has_figures is not None:
            where_clauses.append({"has_figure_reference": {"$eq": filter_has_figures}})
        if filter_page_range:
            start_page, end_page = filter_page_range
            where_clauses.append({"page_start": {"$gte": start_page}})
            where_clauses.append({"page_end": {"$lte": end_page}})
        
        where_filter = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}
        
        return self._query_collection(self.report_collection, query_embedding, n_results, where_filter)
    
    def _query_collection(
        self, collection, query_embedding: np.ndarray,
        n_results: int, where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Query a collection and return formatted results."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        collection_count = collection.count()
        if collection_count == 0:
            return []
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return []
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into a standard format."""
        formatted = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results.get('distances') else 0
                
                formatted.append({
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': distance,
                    'similarity': 1 / (1 + distance)
                })
        
        return formatted
    
    def get_all_criteria(self) -> List[Dict[str, Any]]:
        """Get all criteria chunks."""
        try:
            result = self.criteria_collection.get(include=["documents", "metadatas"])
            
            chunks = []
            if result['ids']:
                for i in range(len(result['ids'])):
                    chunks.append({
                        'chunk_id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i] if result.get('metadatas') else {}
                    })
            return chunks
        except Exception as e:
            logger.error(f"Error getting criteria: {e}")
            return []
    
    def get_all_report_chunks(self) -> List[Dict[str, Any]]:
        """Get all report chunks (for BM25 indexing)."""
        try:
            result = self.report_collection.get(include=["documents", "metadatas"])
            
            chunks = []
            if result['ids']:
                for i in range(len(result['ids'])):
                    chunks.append({
                        'chunk_id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i] if result.get('metadatas') else {}
                    })
            return chunks
        except Exception as e:
            logger.error(f"Error getting report chunks: {e}")
            return []
    
    def clear_criteria(self):
        """Clear all criteria data."""
        try:
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
        try:
            criteria_count = self.criteria_collection.count()
            report_count = self.report_collection.count()
        except Exception:
            criteria_count = 0
            report_count = 0
        
        return {
            'criteria_count': criteria_count,
            'report_count': report_count,
            'persist_directory': str(self.persist_directory),
            'collections': [c.name for c in self.client.list_collections()]
        }
