"""
Basic tests for the Coursework Marker Assistant pipeline.

Run with: pytest tests/test_pipeline.py -v
"""
import pytest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChunker:
    """Tests for the SmartChunker class."""
    
    def test_chunker_initialization(self):
        from src.chunking import SmartChunker
        
        chunker = SmartChunker(criteria_chunk_size=400, report_chunk_size=700)
        
        assert chunker.criteria_chunk_size == 400
        assert chunker.report_chunk_size == 700
    
    def test_token_counting(self):
        from src.chunking import SmartChunker
        
        chunker = SmartChunker()
        text = "This is a simple test sentence."
        tokens = chunker.count_tokens(text)
        
        assert tokens > 0
        assert tokens < 20


class TestEmbedder:
    """Tests for the Embedder class."""
    
    @pytest.fixture
    def embedder(self):
        from src.embeddings import Embedder
        return Embedder()
    
    def test_embedder_initialization(self, embedder):
        assert embedder.embedding_dim == 768  # E5-base-v2 dimension

    def test_single_embedding(self, embedder):
        text = "This is a test sentence."
        embedding = embedder.embed_query(text)
        assert embedding.shape == (768,)

    def test_batch_embedding(self, embedder):
        texts = ["First test sentence.", "Second test sentence.", "Third test sentence."]
        embeddings = embedder.embed_documents(texts)  # Use embed_documents for batches
        assert embeddings.shape == (3, 768)
    
    def test_similarity(self, embedder):
        import numpy as np

        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "Quantum physics is complex."

        emb1 = embedder.embed_query(text1)
        emb2 = embedder.embed_query(text2)
        emb3 = embedder.embed_query(text3)

        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13

    def test_e5_query_prefix(self, embedder):
        """Test that embed_query adds 'query: ' prefix for E5 models."""
        import numpy as np

        # For E5 models, query and document embeddings of same text should differ
        text = "This is a test sentence."

        query_emb = embedder.embed_query(text)
        doc_emb = embedder.embed_documents([text])[0]

        # Should be different vectors due to different prefixes
        if "e5" in embedder.model_name.lower():
            assert not np.allclose(query_emb, doc_emb, atol=0.01), \
                "E5 query and document embeddings should differ due to prefixes"
        else:
            # For non-E5 models, they should be the same
            assert np.allclose(query_emb, doc_emb, atol=0.01), \
                "Non-E5 models should produce same embedding regardless of prefix"

    def test_e5_document_prefix(self, embedder):
        """Test that embed_documents adds 'passage: ' prefix for E5 models."""
        # This is implicitly tested in test_e5_query_prefix
        # Just verify embed_documents accepts list and returns correct shape
        texts = ["First doc", "Second doc"]
        embeddings = embedder.embed_documents(texts)
        assert embeddings.shape == (2, embedder.embedding_dim)


class TestVectorStore:
    """Tests for the ChromaStore class."""
    
    @pytest.fixture
    def temp_store(self):
        from src.vector_store import ChromaStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaStore(
                persist_directory=tmpdir,
                criteria_collection="test_criteria",
                report_collection="test_report"
            )
            yield store
    
    def test_store_initialization(self, temp_store):
        stats = temp_store.get_stats()
        assert stats['criteria_count'] == 0
        assert stats['report_count'] == 0
    
    def test_add_and_query(self, temp_store):
        from src.embeddings import Embedder
        
        embedder = Embedder()
        
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
        embeddings = embedder.embed_documents(texts)  # Use embed_documents for document chunks

        temp_store.add_criteria(chunks, embeddings)
        
        query_emb = embedder.embed_query("analysis of academic sources")
        results = temp_store.query_criteria(query_emb, n_results=2)
        
        assert len(results) == 2
        assert 'analysis' in results[0]['content'].lower()


class TestKSBPromptTemplates:
    """Tests for KSB prompt templates."""
    
    def test_system_prompt_exists(self):
        from src.prompts import KSBPromptTemplates
        
        assert len(KSBPromptTemplates.SYSTEM_PROMPT_KSB_MARKER) > 100
    
    def test_ksb_evaluation_format(self):
        from src.prompts import KSBPromptTemplates
        
        prompt = KSBPromptTemplates.format_ksb_evaluation(
            ksb_code="K1",
            ksb_title="Test KSB",
            pass_criteria="Pass test",
            merit_criteria="Merit test",
            referral_criteria="Referral test",
            evidence_text="Test evidence"
        )
        
        assert "K1" in prompt
        assert "Test KSB" in prompt
        assert "Test evidence" in prompt
    
    def test_overall_summary_format(self):
        from src.prompts import KSBPromptTemplates
        
        prompt = KSBPromptTemplates.format_overall_summary("Test evaluations")
        assert "Test evaluations" in prompt


class TestGradeExtraction:
    """Tests for grade extraction from evaluations."""
    
    def test_extract_grade_from_json(self):
        from src.prompts import extract_grade_from_evaluation
        
        evaluation = '''
        Some text here.
        ```json
        {
            "ksb_code": "K1",
            "grade": "PASS",
            "confidence": "HIGH"
        }
        ```
        More text.
        '''
        
        result = extract_grade_from_evaluation(evaluation)
        assert result['grade'] == 'PASS'
        assert result['confidence'] == 'HIGH'
        assert result['method'] == 'json'
    
    def test_extract_grade_from_regex(self):
        from src.prompts import extract_grade_from_evaluation
        
        evaluation = "The **Grade**: MERIT based on strong evidence."
        
        result = extract_grade_from_evaluation(evaluation)
        assert result['grade'] == 'MERIT'


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_ollama_connection(self):
        from src.llm import OllamaClient
        
        client = OllamaClient()
        assert client.is_available()
    
    @pytest.mark.skip(reason="Requires Ollama to be running")
    def test_simple_generation(self):
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
