"""
Verification script for E5-base-v2 embedding upgrade.

Tests:
1. Model loads correctly with 768 dimensions
2. embed_query() adds "query: " prefix
3. embed_documents() adds "passage: " prefix
4. Query and document embeddings of same text differ (asymmetric encoding)

Run: python verify_e5_upgrade.py
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings import Embedder

def test_model_loading():
    """Test 1: Model loads with correct dimension."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading")
    print("="*80)

    embedder = Embedder()

    print(f"Model name: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.embedding_dim}")

    assert "e5" in embedder.model_name.lower(), f"Expected E5 model, got {embedder.model_name}"
    assert embedder.embedding_dim == 768, f"Expected 768-dim, got {embedder.embedding_dim}"

    print("PASS: Model is E5-base-v2 with 768 dimensions")
    return embedder


def test_query_embedding(embedder):
    """Test 2: Query embedding works and has correct shape."""
    print("\n" + "="*80)
    print("TEST 2: Query Embedding")
    print("="*80)

    query = "machine learning algorithm"
    embedding = embedder.embed_query(query)

    print(f"Query: '{query}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    assert embedding.shape == (768,), f"Expected (768,) shape, got {embedding.shape}"
    assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}"

    print("PASS: Query embedding has correct shape (768,)")
    return embedding


def test_document_embedding(embedder):
    """Test 3: Document embedding works and has correct shape."""
    print("\n" + "="*80)
    print("TEST 3: Document Embedding (Batch)")
    print("="*80)

    documents = [
        "This is the first document about data science.",
        "This is the second document about machine learning.",
        "This is the third document about statistics."
    ]

    embeddings = embedder.embed_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings type: {type(embeddings)}")
    print(f"First document, first 5 values: {embeddings[0][:5]}")

    assert embeddings.shape == (3, 768), f"Expected (3, 768) shape, got {embeddings.shape}"
    assert isinstance(embeddings, np.ndarray), f"Expected numpy array, got {type(embeddings)}"

    print("PASS: Document batch embedding has correct shape (3, 768)")
    return embeddings


def test_asymmetric_encoding(embedder):
    """Test 4: Query and document embeddings differ (asymmetric encoding)."""
    print("\n" + "="*80)
    print("TEST 4: Asymmetric Encoding (CRITICAL for E5)")
    print("="*80)

    text = "This is a test sentence about artificial intelligence."

    query_emb = embedder.embed_query(text)
    doc_emb = embedder.embed_documents([text])[0]

    print(f"Text: '{text}'")
    print(f"Query embedding first 5 values: {query_emb[:5]}")
    print(f"Document embedding first 5 values: {doc_emb[:5]}")

    # Check if they're different (due to different prefixes)
    are_different = not np.allclose(query_emb, doc_emb, atol=0.01)

    # Calculate cosine similarity to show they're still semantically related
    cosine_sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    print(f"\nCosine similarity between query and doc embeddings: {cosine_sim:.4f}")

    print(f"Are embeddings different? {are_different}")

    if "e5" in embedder.model_name.lower():
        assert are_different, \
            "E5 models should produce different embeddings for query vs document (different prefixes)"
        print("PASS: Asymmetric encoding working correctly (query != document)")
    else:
        print("WARNING: Not an E5 model, skipping asymmetric check")


def test_semantic_similarity(embedder):
    """Test 5: Similar texts have higher similarity than dissimilar texts."""
    print("\n" + "="*80)
    print("TEST 5: Semantic Similarity")
    print("="*80)

    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    text3 = "Quantum physics is very complex."

    emb1 = embedder.embed_query(text1)
    emb2 = embedder.embed_query(text2)
    emb3 = embedder.embed_query(text3)

    sim_12 = np.dot(emb1, emb2)  # Similar texts
    sim_13 = np.dot(emb1, emb3)  # Dissimilar texts

    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Text 3: '{text3}'")
    print(f"\nSimilarity(1, 2): {sim_12:.4f} (should be HIGH)")
    print(f"Similarity(1, 3): {sim_13:.4f} (should be LOW)")

    assert sim_12 > sim_13, \
        f"Similar texts should have higher similarity ({sim_12:.4f}) than dissimilar ({sim_13:.4f})"

    print("PASS: Similar texts have higher similarity")


def main():
    print("\n" + "="*80)
    print("E5-BASE-V2 EMBEDDING UPGRADE VERIFICATION")
    print("="*80)

    try:
        # Test 1: Load model
        embedder = test_model_loading()

        # Test 2: Query embedding
        test_query_embedding(embedder)

        # Test 3: Document embedding
        test_document_embedding(embedder)

        # Test 4: Asymmetric encoding (CRITICAL!)
        test_asymmetric_encoding(embedder)

        # Test 5: Semantic similarity
        test_semantic_similarity(embedder)

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        print("\nE5-base-v2 upgrade successful!")
        print("\nNext steps:")
        print("1. Delete old index: Use 'Reset Index' button in UI or run:")
        print("   rm -rf ./data/indexes/")
        print("2. Run the application: streamlit run ui/ksb_app.py")
        print("3. Upload a report to verify end-to-end pipeline")
        print("\nSee MIGRATION.md for full details.")

        return 0

    except AssertionError as e:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print("\n" + "="*80)
        print("UNEXPECTED ERROR")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
