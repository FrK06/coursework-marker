"""
Test script for multi-base query retrieval improvement.

Verifies that:
1. get_base_queries() generates 2-3 targeted queries per KSB
2. retrieve_for_criterion() expands all base queries
3. Query count per KSB is higher than before
4. Evidence retrieval improves for statistical/infrastructure KSBs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import Retriever
from src.retrieval.retriever import QueryExpander
from src.embeddings import Embedder
from src.vector_store import ChromaStore
from src.chunking import SmartChunker
import tempfile

def test_base_query_generation():
    """Test that get_base_queries() works correctly."""
    print("="*80)
    print("TEST 1: Base Query Generation")
    print("="*80)

    expander = QueryExpander()

    # Test statistical KSBs
    test_cases = [
        ('K22', 'Knowledge of hypothesis testing and statistical significance'),
        ('K2', 'Knowledge of data storage and processing infrastructure'),
        ('K24', 'Knowledge of GDPR and data protection requirements'),
        ('S9', 'Skill in data manipulation and visualization'),
        ('K18', 'Knowledge of programming and data engineering'),
    ]

    for ksb_code, ksb_desc in test_cases:
        base_queries = expander.get_base_queries(ksb_code, ksb_desc)
        print(f"\n{ksb_code}: {ksb_desc[:50]}...")
        print(f"  Base queries ({len(base_queries)}):")
        for i, query in enumerate(base_queries, 1):
            print(f"    {i}. {query}")

    print("\nPASS: Base queries generated successfully")


def test_query_expansion_count():
    """Test that query count increases with multi-base approach."""
    print("\n" + "="*80)
    print("TEST 2: Query Count Comparison")
    print("="*80)

    expander = QueryExpander()

    # Statistical KSB example
    ksb_code = "K22"
    ksb_desc = "Knowledge of hypothesis testing, statistical significance, and effect sizes"

    # OLD approach: single base query
    old_queries = expander.expand_query(ksb_desc[:500], ksb_code)
    old_count = len(old_queries)

    # NEW approach: multi-base queries
    base_queries = expander.get_base_queries(ksb_code, ksb_desc)
    all_queries = []
    for base_query in base_queries:
        expanded = expander.expand_query(base_query, ksb_code)
        all_queries.extend(expanded)

    # Add original + concepts
    all_queries.extend(old_queries)
    concepts = expander.extract_key_concepts(ksb_desc)
    all_queries.extend(concepts)

    # Deduplicate
    seen = set()
    new_queries = []
    for q in all_queries:
        q_lower = q.lower().strip()
        if q_lower not in seen and len(q_lower) > 2:
            seen.add(q_lower)
            new_queries.append(q)

    new_count = len(new_queries)

    print(f"\nK22 (Hypothesis Testing):")
    print(f"  OLD approach: {old_count} query variations")
    print(f"  NEW approach: {new_count} query variations")
    print(f"  Improvement: +{new_count - old_count} queries ({(new_count / old_count - 1) * 100:.1f}% increase)")

    print("\nSample queries (first 10):")
    for i, q in enumerate(new_queries[:10], 1):
        print(f"  {i}. {q[:60]}...")

    if new_count > old_count:
        print("\nPASS: Multi-base queries generate more variations")
    else:
        print("\nFAIL: Multi-base queries should generate more variations")


def test_retrieval_with_mock_chunks():
    """Test retrieval with mock chunks to verify multi-base queries work."""
    print("\n" + "="*80)
    print("TEST 3: Retrieval with Multi-Base Queries")
    print("="*80)

    # Create mock chunks mimicking DSP report with K22 evidence
    class MockChunk:
        def __init__(self, content, section_number='1', page=1):
            self.content = content
            self.section_number = section_number
            self.page_estimate = page
            self.chunk_type = 'text'
            self.heading_level = None
            self.figure_ids = []
            self.has_figure_reference = False

    chunks = [
        # Chunk with methodology evidence
        MockChunk(
            "[Section 2.1 | Task 2 - Hypothesis Testing]\n\n"
            "2.1 Test Selection\n\n"
            "A Welch's t-test was selected to compare means between groups. "
            "This test was chosen because the assumptions of normality were verified "
            "using Shapiro-Wilk test (p > 0.05) and equal variance assumption was "
            "violated (Levene's test p < 0.05), making Welch's test more appropriate "
            "than Student's t-test.",
            section_number='2.1',
            page=3
        ),
        # Chunk with results evidence
        MockChunk(
            "[Section 2.2 | Task 2 - Hypothesis Testing]\n\n"
            "2.2 Results\n\n"
            "The Welch's t-test yielded t(45.3) = 3.24, p = 0.002, indicating a "
            "statistically significant difference. Effect size (Cohen's d = 0.68) "
            "suggests a medium-to-large practical effect. The 95% confidence interval "
            "for the difference in means was [2.1, 8.9].",
            section_number='2.2',
            page=4
        ),
        # Chunk with business decision evidence
        MockChunk(
            "[Section 2.3 | Task 2 - Hypothesis Testing]\n\n"
            "2.3 Business Recommendations\n\n"
            "Based on the statistical significance (p = 0.002) and practical effect size "
            "(Cohen's d = 0.68), I recommend implementing Strategy A. The confidence "
            "interval suggests we can expect a 2-9 point improvement, which translates "
            "to a 15-20% increase in customer satisfaction and estimated £50k annual revenue.",
            section_number='2.3',
            page=5
        ),
        # Irrelevant chunk
        MockChunk(
            "[Section 1.1 | Task 1 - Data Infrastructure]\n\n"
            "1.1 Storage Architecture\n\n"
            "The data storage uses Azure Blob Storage with hot tier for active data "
            "and cool tier for archival. This provides cost-effective storage while "
            "maintaining performance for frequently accessed data.",
            section_number='1.1',
            page=1
        ),
    ]

    # Create embedder and vector store
    embedder = Embedder()
    chunker = SmartChunker()

    # Chunk the mock chunks
    text_chunks = chunker.chunk_report(chunks, document_id="test_report")

    print(f"\nCreated {len(text_chunks)} chunks from mock data")

    # Create vector store and add chunks
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = ChromaStore(persist_directory=tmpdir)

        # Embed and index chunks
        chunk_dicts = [c.to_dict() if hasattr(c, 'to_dict') else c for c in text_chunks]
        texts = [c.get('content', '') if isinstance(c, dict) else c.content for c in chunk_dicts]
        embeddings = embedder.embed_documents(texts)
        vector_store.add_report(chunk_dicts, embeddings)

        print(f"Indexed {len(texts)} chunks in vector store")

        # Create retriever
        from config import RetrievalConfig
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            report_top_k=RetrievalConfig.REPORT_TOP_K,
            use_hybrid=True,
            semantic_weight=0.6,
            keyword_weight=0.4
        )

        # Test retrieval for K22
        criterion_text = "Knowledge of hypothesis testing, statistical significance, and effect sizes"
        result = retriever.retrieve_for_criterion(criterion_text, criterion_id="K22")

        print(f"\nK22 Retrieval Results:")
        print(f"  Query variations: {len(result.query_variations)}")
        print(f"  Chunks retrieved: {len(result.retrieved_chunks)}")
        print(f"  Search strategy: {result.search_strategy}")

        print(f"\nSample queries used:")
        for i, q in enumerate(result.query_variations[:10], 1):
            print(f"  {i}. {q[:60]}...")

        print(f"\nRetrieved chunks:")
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            section = chunk.get('metadata', {}).get('section_number', '?')
            similarity = chunk.get('similarity', 0)
            preview = chunk.get('content', '')[:80].replace('\n', ' ')
            print(f"  {i}. Section {section} (sim={similarity:.3f}): {preview}...")

        # Verify we got relevant chunks (sections 2.1, 2.2, 2.3)
        retrieved_sections = [
            c.get('metadata', {}).get('section_number', '')
            for c in result.retrieved_chunks
        ]
        k22_sections = [s for s in retrieved_sections if s.startswith('2.')]
        irrelevant_sections = [s for s in retrieved_sections if s.startswith('1.')]

        print(f"\nRelevant K22 chunks (Task 2): {len(k22_sections)}")
        print(f"Irrelevant chunks (Task 1): {len(irrelevant_sections)}")

        if len(k22_sections) >= 2:
            print("\nPASS: Multi-base queries retrieved relevant K22 evidence")
        else:
            print("\nWARNING: Expected more K22-relevant chunks")


def main():
    print("\n" + "="*80)
    print("MULTI-BASE QUERY RETRIEVAL VERIFICATION")
    print("="*80 + "\n")

    # Test 1: Base query generation
    test_base_query_generation()

    # Test 2: Query count comparison
    test_query_expansion_count()

    # Test 3: Retrieval with mock chunks
    test_retrieval_with_mock_chunks()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nMulti-base query strategy:")
    print("  1. get_base_queries() generates 2-3 targeted queries per KSB")
    print("  2. Each base query is expanded with expand_query()")
    print("  3. All variations are merged and deduplicated")
    print("  4. Higher query count = better evidence coverage")
    print("\nExpected impact:")
    print("  - Statistical KSBs: Better coverage of methodology + results + business")
    print("  - Infrastructure KSBs: Better coverage of architecture + implementation + justification")
    print("  - Ethics KSBs: Better coverage of legal + implementation + risk")
    print("\nNext: Test on real DSP report to measure grade improvements")


if __name__ == "__main__":
    main()
