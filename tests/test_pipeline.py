"""
Basic tests for the Coursework Marker Assistant pipeline.

Run with: pytest tests/test_pipeline.py -v
"""
import pytest
import tempfile
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChunker:
    """Tests for the SmartChunker class."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes correctly."""
        from src.chunking import SmartChunker
        
        chunker = SmartChunker(
            criteria_chunk_size=400,
            report_chunk_size=700
        )
        
        assert chunker.criteria_chunk_size == 400
        assert chunker.report_chunk_size == 700
    
    def test_token_counting(self):
        """Test token counting functionality."""
        from src.chunking import SmartChunker
        
        chunker = SmartChunker()
        
        # Simple text should have some tokens
        text = "This is a simple test sentence."
        tokens = chunker.count_tokens(text)
        
        assert tokens > 0
        assert tokens < 20  # Should be around 7-8 tokens


class TestEmbedder:
    """Tests for the Embedder class."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        from src.embeddings import Embedder
        return Embedder()
    
    def test_embedder_initialization(self, embedder):
        """Test embedder initializes with correct dimensions."""
        assert embedder.embedding_dim == 384
    
    def test_single_embedding(self, embedder):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedder.embed_query(text)
        
        assert embedding.shape == (384,)
    
    def test_batch_embedding(self, embedder):
        """Test embedding multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedder.embed(texts)
        
        assert embeddings.shape == (3, 384)
    
    def test_similarity(self, embedder):
        """Test that similar texts have higher similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "Quantum physics is complex."
        
        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)
        emb3 = embedder.embed_query(text3)
        
        import numpy as np
        
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13


class TestVectorStore:
    """Tests for the ChromaStore class."""
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary vector store."""
        from src.vector_store import ChromaStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaStore(
                persist_directory=tmpdir,
                criteria_collection="test_criteria",
                report_collection="test_report"
            )
            yield store
    
    def test_store_initialization(self, temp_store):
        """Test store initializes correctly."""
        stats = temp_store.get_stats()
        
        assert stats['criteria_count'] == 0
        assert stats['report_count'] == 0
    
    def test_add_and_query(self, temp_store):
        """Test adding and querying documents."""
        from src.embeddings import Embedder
        import numpy as np
        
        embedder = Embedder()
        
        # Add some test criteria
        chunks = [
            {
                'content': 'Criterion 1: Critical analysis of sources',
                'chunk_id': 'crit_1',
                'document_type': 'criteria',
                'criterion_id': '1',
                'token_count': 10
            },
            {
                'content': 'Criterion 2: Clear written communication',
                'chunk_id': 'crit_2',
                'document_type': 'criteria',
                'criterion_id': '2',
                'token_count': 10
            }
        ]
        
        texts = [c['content'] for c in chunks]
        embeddings = embedder.embed(texts)
        
        temp_store.add_criteria(chunks, embeddings)
        
        # Query
        query_emb = embedder.embed_query("analysis of academic sources")
        results = temp_store.query_criteria(query_emb, n_results=2)
        
        assert len(results) == 2
        # First result should be about analysis
        assert 'analysis' in results[0]['content'].lower()


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_system_prompts_exist(self):
        """Test that all system prompts are defined."""
        from src.prompts import PromptTemplates
        
        assert len(PromptTemplates.SYSTEM_PROMPT_MARKER) > 100
        assert len(PromptTemplates.SYSTEM_PROMPT_EXTRACTOR) > 50
        assert len(PromptTemplates.SYSTEM_PROMPT_SUMMARIZER) > 50
    
    def test_criterion_evaluation_format(self):
        """Test criterion evaluation prompt formatting."""
        from src.prompts import PromptTemplates
        
        prompt = PromptTemplates.format_criterion_evaluation(
            criterion_text="Test criterion",
            evidence_text="Test evidence"
        )
        
        assert "Test criterion" in prompt
        assert "Test evidence" in prompt
        assert "<criterion>" in prompt
        assert "<evidence_from_report>" in prompt
    
    def test_overall_summary_format(self):
        """Test overall summary prompt formatting."""
        from src.prompts import PromptTemplates
        
        prompt = PromptTemplates.format_overall_summary(
            evaluations_text="Test evaluations"
        )
        
        assert "Test evaluations" in prompt
        assert "<individual_evaluations>" in prompt


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_ollama_connection(self):
        """Test connection to Ollama."""
        from src.llm import OllamaClient
        
        client = OllamaClient()
        assert client.is_available()
    
    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_simple_generation(self):
        """Test simple text generation."""
        from src.llm import OllamaClient
        
        client = OllamaClient()
        response = client.generate(
            prompt="Say 'Hello' in exactly one word.",
            temperature=0.1,
            max_tokens=10
        )
        
        assert len(response) > 0
        assert 'hello' in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
