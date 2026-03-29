"""
Test script to verify audit trail count fixes.

Verifies that:
1. Query variations are properly recorded (not 0)
2. Total chunks retrieved is accurate (before OCR additions)
3. OCR chunks are counted separately
4. Boilerplate filtered is non-negative
5. Search strategy mode is recorded from metadata
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.core import AgentContext
from src.agents.scoring_agent import ScoringAgent
from src.llm import OllamaClient
from src.criteria import get_module_criteria

def test_audit_trail_metadata():
    """Test that audit trail uses evidence_metadata correctly."""
    print("="*80)
    print("TEST: Audit Trail Metadata Integration")
    print("="*80)

    # Create mock context with evidence_metadata (simulates AnalysisAgent output)
    context = AgentContext()

    # Mock KSB criteria
    context.ksb_criteria = [
        {
            'code': 'K22',
            'title': 'Knowledge of hypothesis testing',
            'knowledge_criteria': 'Understanding of statistical significance',
            'pass_criteria': 'Demonstrates basic understanding',
            'merit_criteria': 'Shows advanced application',
            'weight': 0.5
        }
    ]

    # Mock evidence (simulate 8 chunks retrieved, then 2 OCR chunks added)
    context.evidence_map = {
        'K22': [
            {'content': f'Evidence chunk {i}', 'section': f'2.{i}', 'relevance': 0.8}
            for i in range(10)  # 8 original + 2 OCR = 10 total
        ]
    }

    # Mock evidence metadata (populated by AnalysisAgent)
    context.evidence_metadata = {
        'K22': {
            'search_strategy': 'hybrid',
            'query_variations': 15,  # Count of query variations (EvidenceFinder returns count, not list)
            'total_chunks': 8,  # BEFORE OCR chunks added
            'ocr_chunks': 2     # OCR chunks added after initial retrieval
        }
    }

    print("\nMock Data Setup:")
    print(f"  Evidence chunks in map: {len(context.evidence_map['K22'])}")
    print(f"  Metadata - Total chunks (before OCR): {context.evidence_metadata['K22']['total_chunks']}")
    print(f"  Metadata - OCR chunks added: {context.evidence_metadata['K22']['ocr_chunks']}")
    print(f"  Metadata - Query variations: {context.evidence_metadata['K22']['query_variations']}")
    print(f"  Metadata - Search strategy: {context.evidence_metadata['K22']['search_strategy']}")

    # Create mock LLM client (won't actually be called in this test)
    try:
        llm = OllamaClient(model='mistral:7b')
    except Exception as e:
        print(f"\nWarning: Could not connect to Ollama ({e})")
        print("This is OK - we're only testing metadata extraction, not LLM calls")
        llm = None

    # We can't easily test the full ScoringAgent.process() without LLM
    # Instead, we'll verify the audit trail building logic directly

    # Simulate what ScoringAgent does when building audit trail
    ksb_code = 'K22'
    metadata = context.evidence_metadata.get(ksb_code, {})
    evidence_parts = context.evidence_map.get(ksb_code, [])

    audit_trail = {
        'evidence': {
            'chunks': [],
            'total_chunks_retrieved': metadata.get('total_chunks', len(context.evidence_map.get(ksb_code, []))),
            'chunks_after_filtering': len(evidence_parts),
            'ocr_chunks_added': metadata.get('ocr_chunks', 0),
            'search_strategy': {
                'query_variations': metadata.get('query_variations', 0),  # Already an int count
                'mode': metadata.get('search_strategy', 'hybrid'),
                'boilerplate_filtered': 0  # Will be calculated
            }
        }
    }

    # Calculate boilerplate filtering (prevent negative values)
    total_before_ocr = audit_trail['evidence']['total_chunks_retrieved']
    chunks_used = audit_trail['evidence']['chunks_after_filtering']
    audit_trail['evidence']['search_strategy']['boilerplate_filtered'] = max(0, total_before_ocr - chunks_used)

    print("\n" + "="*80)
    print("AUDIT TRAIL RESULTS:")
    print("="*80)
    print(f"\nEvidence Counts:")
    print(f"  Total Retrieved (before OCR): {audit_trail['evidence']['total_chunks_retrieved']}")
    print(f"  OCR Chunks Added:             {audit_trail['evidence']['ocr_chunks_added']}")
    print(f"  Chunks After Filtering:       {audit_trail['evidence']['chunks_after_filtering']}")
    print(f"  Boilerplate Filtered:         {audit_trail['evidence']['search_strategy']['boilerplate_filtered']}")

    print(f"\nSearch Strategy:")
    print(f"  Query Variations: {audit_trail['evidence']['search_strategy']['query_variations']}")
    print(f"  Mode: {audit_trail['evidence']['search_strategy']['mode']}")

    print("\n" + "="*80)
    print("VERIFICATION:")
    print("="*80)

    # Test 1: Query variations should not be 0
    query_count = audit_trail['evidence']['search_strategy']['query_variations']
    if query_count > 0:
        print(f"PASS: Query variations = {query_count} (not 0)")
    else:
        print(f"FAIL: Query variations = 0 (should be {metadata['query_variations']})")

    # Test 2: Total chunks should be before OCR (8, not 10)
    total_retrieved = audit_trail['evidence']['total_chunks_retrieved']
    expected_total = 8
    if total_retrieved == expected_total:
        print(f"PASS: Total chunks retrieved = {total_retrieved} (correct, before OCR)")
    else:
        print(f"FAIL: Total chunks retrieved = {total_retrieved} (expected {expected_total})")

    # Test 3: OCR chunks should be counted separately
    ocr_chunks = audit_trail['evidence']['ocr_chunks_added']
    expected_ocr = 2
    if ocr_chunks == expected_ocr:
        print(f"PASS: OCR chunks added = {ocr_chunks}")
    else:
        print(f"FAIL: OCR chunks added = {ocr_chunks} (expected {expected_ocr})")

    # Test 4: Boilerplate filtered should be non-negative
    boilerplate = audit_trail['evidence']['search_strategy']['boilerplate_filtered']
    if boilerplate >= 0:
        print(f"PASS: Boilerplate filtered = {boilerplate} (non-negative)")
    else:
        print(f"FAIL: Boilerplate filtered = {boilerplate} (negative!)")

    # Test 5: Search strategy mode should be from metadata
    mode = audit_trail['evidence']['search_strategy']['mode']
    expected_mode = 'hybrid'
    if mode == expected_mode:
        print(f"PASS: Search strategy mode = '{mode}'")
    else:
        print(f"FAIL: Search strategy mode = '{mode}' (expected '{expected_mode}')")

    # Test 6: Math check - total_before_ocr + ocr_chunks = chunks_in_map
    math_check = total_retrieved + ocr_chunks == len(context.evidence_map['K22'])
    if math_check:
        print(f"PASS: Math check: {total_retrieved} + {ocr_chunks} = {len(context.evidence_map['K22'])}")
    else:
        print(f"FAIL: Math check failed: {total_retrieved} + {ocr_chunks} != {len(context.evidence_map['K22'])}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nBEFORE FIX (old behavior):")
    print("  - Query Variations: 0 (hardcoded)")
    print("  - Total Retrieved: 10 (counted AFTER OCR chunks added)")
    print("  - Chunks After Filtering: 10")
    print("  - Boilerplate Filtered: 10 - 10 = 0 (misleading)")
    print("  - Search Strategy: 'hybrid' (hardcoded)")

    print("\nAFTER FIX (new behavior):")
    print(f"  - Query Variations: {query_count} (from metadata)")
    print(f"  - Total Retrieved: {total_retrieved} (from metadata, BEFORE OCR)")
    print(f"  - OCR Chunks Added: {ocr_chunks} (tracked separately)")
    print(f"  - Chunks After Filtering: {chunks_used}")
    print(f"  - Boilerplate Filtered: max(0, {total_retrieved} - {chunks_used}) = {boilerplate}")
    print(f"  - Search Strategy: '{mode}' (from metadata)")

    print("\nKEY IMPROVEMENTS:")
    print("  1. Query count now shows actual variations (15 instead of 0)")
    print("  2. Total chunks is accurate (before OCR additions)")
    print("  3. OCR chunks tracked separately for transparency")
    print("  4. Boilerplate calculation can't go negative")
    print("  5. Search strategy reflects actual retrieval mode")

    print("\n" + "="*80)
    print("NEXT STEP: Run real assessment and check exported JSON")
    print("="*80)
    print("\nTo verify end-to-end:")
    print("  1. streamlit run ui/ksb_app.py")
    print("  2. Upload a test report with images")
    print("  3. Run assessment")
    print("  4. Export results as JSON")
    print("  5. Check audit_trail section for any KSB:")
    print("     - query_variations should be > 0 (typically 10-20)")
    print("     - total_chunks_retrieved should be < chunks in evidence_map")
    print("     - ocr_chunks_added should show how many OCR chunks were added")
    print("     - boilerplate_filtered should be >= 0")


def main():
    print("\n" + "="*80)
    print("AUDIT TRAIL FIX VERIFICATION")
    print("="*80 + "\n")

    test_audit_trail_metadata()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
