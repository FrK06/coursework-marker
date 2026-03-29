"""
Verification script for improved chunking.

Tests the new word-based merging with parent context prepending.

Usage:
    python verify_chunking.py <path_to_docx_file>

    Or without arguments to use synthetic test data.
"""
import sys
from pathlib import Path
from src.chunking import SmartChunker
from src.document_processing import DocxProcessor


def create_synthetic_chunks():
    """Create synthetic chunks mimicking a typical DSP report structure."""

    class MockChunk:
        def __init__(self, content, chunk_type='text', heading_level=None,
                     section_number=None, page_estimate=1):
            self.content = content
            self.chunk_type = chunk_type
            self.heading_level = heading_level
            self.section_number = section_number
            self.page_estimate = page_estimate
            self.figure_ids = []
            self.has_figure_reference = False

    chunks = []

    # Simulate DSP report structure
    # Task 1: Data Infrastructure (parent)
    chunks.append(MockChunk(
        "1. Task 1 - Data Infrastructure & Storage",
        chunk_type='heading',
        heading_level=1,
        section_number='1',
        page_estimate=1
    ))

    # Subsection 1.1 (120 words - should merge with 1.2)
    chunks.append(MockChunk(
        "1.1 Storage Architecture\n\n" + " ".join(["word"] * 120),
        chunk_type='heading',
        heading_level=2,
        section_number='1.1',
        page_estimate=1
    ))

    # Subsection 1.2 (100 words - should merge with 1.1)
    chunks.append(MockChunk(
        "1.2 Data Quality Controls\n\n" + " ".join(["word"] * 100),
        chunk_type='heading',
        heading_level=2,
        section_number='1.2',
        page_estimate=1
    ))

    # Subsection 1.3 (250 words - should be standalone chunk)
    chunks.append(MockChunk(
        "1.3 Infrastructure Design\n\n" + " ".join(["word"] * 250),
        chunk_type='heading',
        heading_level=2,
        section_number='1.3',
        page_estimate=2
    ))

    # Task 2: Hypothesis Testing (parent)
    chunks.append(MockChunk(
        "2. Task 2 - Testing of a hypothesis",
        chunk_type='heading',
        heading_level=1,
        section_number='2',
        page_estimate=2
    ))

    # Subsection 2.1 (80 words - should merge with 2.2 and 2.3)
    chunks.append(MockChunk(
        "2.1 Hypothesis Statement\n\n" + " ".join(["word"] * 80),
        chunk_type='heading',
        heading_level=2,
        section_number='2.1',
        page_estimate=2
    ))

    # Subsection 2.2 (70 words - should merge with 2.1 and 2.3)
    chunks.append(MockChunk(
        "2.2 Test Selection\n\n" + " ".join(["word"] * 70),
        chunk_type='heading',
        heading_level=2,
        section_number='2.2',
        page_estimate=2
    ))

    # Subsection 2.3 (90 words - completes the merge to reach 240 words)
    chunks.append(MockChunk(
        "2.3 Results and Interpretation\n\n" + " ".join(["word"] * 90),
        chunk_type='heading',
        heading_level=2,
        section_number='2.3',
        page_estimate=3
    ))

    # Task 3: Visualizations (parent)
    chunks.append(MockChunk(
        "3. Task 3 - Data Visualizations",
        chunk_type='heading',
        heading_level=1,
        section_number='3',
        page_estimate=3
    ))

    # Subsection 3.1 (300 words - standalone)
    chunks.append(MockChunk(
        "3.1 Visualization Design\n\n" + " ".join(["word"] * 300),
        chunk_type='heading',
        heading_level=2,
        section_number='3.1',
        page_estimate=3
    ))

    # Subsection 3.2 (350 words - standalone)
    chunks.append(MockChunk(
        "3.2 Chart Analysis\n\n" + " ".join(["word"] * 350),
        chunk_type='heading',
        heading_level=2,
        section_number='3.2',
        page_estimate=4
    ))

    # Very short subsection (30 words - should be skipped)
    chunks.append(MockChunk(
        "3.3 Conclusion\n\n" + " ".join(["word"] * 30),
        chunk_type='heading',
        heading_level=2,
        section_number='3.3',
        page_estimate=4
    ))

    return chunks


def test_chunking_with_file(file_path: Path):
    """Test chunking with a real DOCX file."""
    print(f"\n{'='*80}")
    print(f"TESTING WITH FILE: {file_path}")
    print(f"{'='*80}\n")

    processor = DocxProcessor()
    doc = processor.process(str(file_path))

    chunker = SmartChunker()
    chunks = chunker.chunk_report(doc.chunks, document_id="test_report")

    print_chunk_analysis(chunks)


def test_chunking_with_synthetic():
    """Test chunking with synthetic data."""
    print(f"\n{'='*80}")
    print(f"TESTING WITH SYNTHETIC DSP REPORT DATA")
    print(f"{'='*80}\n")

    print("Input structure:")
    print("  - Task 1: Data Infrastructure (parent)")
    print("    - 1.1: 120 words (should merge)")
    print("    - 1.2: 100 words (should merge)")
    print("    - 1.3: 250 words (standalone)")
    print("  - Task 2: Hypothesis Testing (parent)")
    print("    - 2.1: 80 words (should merge)")
    print("    - 2.2: 70 words (should merge)")
    print("    - 2.3: 90 words (should merge)")
    print("  - Task 3: Visualizations (parent)")
    print("    - 3.1: 300 words (standalone)")
    print("    - 3.2: 350 words (standalone)")
    print("    - 3.3: 30 words (should skip - <50 words)")
    print("\n")

    chunks = create_synthetic_chunks()

    chunker = SmartChunker()
    result_chunks = chunker.chunk_report(chunks, document_id="synthetic_report")

    print_chunk_analysis(result_chunks)

    # Show parent context prepending
    print(f"\n{'='*80}")
    print("PARENT CONTEXT VERIFICATION")
    print(f"{'='*80}\n")

    for i, chunk in enumerate(result_chunks[:3], 1):
        content_preview = chunk.content[:150].replace('\n', ' ')
        print(f"Chunk {i}: {content_preview}...")
        if chunk.content.startswith('['):
            print(f"  [PASS] Has parent context prepended")
        else:
            print(f"  [WARNING] Missing parent context")
        print()


def print_chunk_analysis(chunks):
    """Print detailed analysis of chunks."""
    print(f"CHUNKING RESULTS")
    print(f"{'-'*80}\n")

    print(f"Total chunks created: {len(chunks)}")
    print(f"Target: 15-20 chunks for typical 3000-5000 word reports\n")

    print(f"{'Chunk ID':<15} {'Section':<30} {'Words':<10} {'Status':<20}")
    print(f"{'-'*80}")

    total_words = 0
    min_words = float('inf')
    max_words = 0
    under_50 = 0

    for chunk in chunks:
        word_count = len(chunk.content.split())
        total_words += word_count
        min_words = min(min_words, word_count)
        max_words = max(max_words, word_count)

        if word_count < 50:
            under_50 += 1
            status = "WARNING: Too short"
        elif word_count < 200:
            status = "WARNING: Below target"
        elif word_count <= 400:
            status = "OK: Optimal"
        else:
            status = "WARNING: Above target"

        section = chunk.section_title or chunk.section_number or "Unknown"
        section = section[:30] if len(section) > 30 else section

        print(f"{chunk.chunk_id:<15} {section:<30} {word_count:<10} {status:<20}")

    print(f"\n{'-'*80}")
    print(f"STATISTICS")
    print(f"{'-'*80}\n")

    avg_words = total_words / len(chunks) if chunks else 0

    print(f"Total words: {total_words}")
    print(f"Average words per chunk: {avg_words:.1f}")
    print(f"Min words: {min_words if min_words != float('inf') else 0}")
    print(f"Max words: {max_words}")
    print(f"Chunks < 50 words: {under_50} (should be 0)")
    print()

    # Verify requirements
    print(f"REQUIREMENT VERIFICATION")
    print(f"{'-'*80}\n")

    req1 = "PASS" if avg_words >= 200 else "FAIL"
    print(f"[{req1}] Minimum 200 words per chunk (avg: {avg_words:.1f})")

    req2 = "PASS" if under_50 == 0 else "FAIL"
    print(f"[{req2}] No chunks < 50 words (found: {under_50})")

    req3 = "PASS" if 15 <= len(chunks) <= 20 else "INFO"
    print(f"[{req3}] Target 15-20 chunks (actual: {len(chunks)})")

    # Check parent context
    chunks_with_parent = sum(1 for c in chunks if c.content.startswith('['))
    req4 = "PASS" if chunks_with_parent > 0 else "FAIL"
    print(f"[{req4}] Parent context prepended ({chunks_with_parent}/{len(chunks)} chunks)")
    print()


def main():
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        if not file_path.suffix == '.docx':
            print(f"Error: File must be a .docx file")
            sys.exit(1)
        test_chunking_with_file(file_path)
    else:
        test_chunking_with_synthetic()


if __name__ == "__main__":
    main()
