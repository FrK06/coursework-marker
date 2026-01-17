#!/usr/bin/env python3
"""
Coursework Marker Assistant - Main Entry Point

This script provides a command-line interface to run the coursework marker
either as a Streamlit app or as a programmatic pipeline.

Usage:
    # Run the Streamlit UI
    python main.py ui
    
    # Or directly
    streamlit run ui/streamlit_app.py
    
    # Run programmatic evaluation
    python main.py evaluate --criteria path/to/criteria.pdf --report path/to/report.docx
"""
import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ui():
    """Launch the Streamlit UI."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    subprocess.run(["streamlit", "run", str(ui_path)])


def run_evaluation(
    criteria_path: str,
    report_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False
):
    """
    Run programmatic evaluation.
    
    Args:
        criteria_path: Path to criteria document
        report_path: Path to student report
        output_path: Optional path for output JSON
        verbose: Enable verbose logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import modules
    from src.document_processing import DocxProcessor, PDFProcessor
    from src.chunking import SmartChunker
    from src.embeddings import Embedder
    from src.vector_store import ChromaStore
    from src.retrieval import Retriever
    from src.llm import OllamaClient
    from src.prompts import PromptTemplates
    import tempfile
    
    logger.info("Starting coursework evaluation...")
    
    # Process criteria
    logger.info(f"Processing criteria: {criteria_path}")
    criteria_path = Path(criteria_path)
    
    if criteria_path.suffix.lower() == '.pdf':
        processor = PDFProcessor()
    else:
        processor = DocxProcessor()
    
    criteria_doc = processor.process(str(criteria_path))
    
    # Process report
    logger.info(f"Processing report: {report_path}")
    report_processor = DocxProcessor()
    report_doc = report_processor.process(report_path)
    
    # Chunk documents
    logger.info("Chunking documents...")
    chunker = SmartChunker()
    criteria_chunks = chunker.chunk_criteria(criteria_doc.chunks)
    report_chunks = chunker.chunk_report(report_doc.chunks)
    
    logger.info(f"Criteria: {len(criteria_chunks)} chunks")
    logger.info(f"Report: {len(report_chunks)} chunks")
    
    # Initialize components
    logger.info("Loading embedding model...")
    embedder = Embedder()
    
    logger.info("Connecting to Ollama...")
    llm = OllamaClient()
    
    # Create temporary vector store
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = ChromaStore(persist_directory=tmpdir)
        
        # Embed and index
        logger.info("Embedding and indexing documents...")
        
        criteria_texts = [c.content for c in criteria_chunks]
        criteria_embeddings = embedder.embed_documents(criteria_texts)
        criteria_dicts = [c.to_dict() for c in criteria_chunks]
        vector_store.add_criteria(criteria_dicts, criteria_embeddings)
        
        report_texts = [c.content for c in report_chunks]
        report_embeddings = embedder.embed_documents(report_texts)
        report_dicts = [c.to_dict() for c in report_chunks]
        vector_store.add_report(report_dicts, report_embeddings)
        
        # Create retriever
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        # Get criteria list
        criteria_list = retriever.extract_criteria_list()
        
        if not criteria_list:
            logger.warning("Could not extract criteria, using general evaluation")
            criteria_list = [{'id': 'general', 'text': criteria_doc.raw_text[:2000]}]
        
        # Evaluate each criterion
        results = {
            'criteria_evaluations': [],
            'overall_summary': None
        }
        
        all_evaluations = []
        
        for criterion in criteria_list:
            logger.info(f"Evaluating criterion {criterion['id']}...")
            
            # Retrieve evidence
            retrieval_result = retriever.retrieve_for_criterion(
                criterion['text'],
                criterion['id']
            )
            
            evidence_text = retriever.format_context_for_llm(retrieval_result)
            
            # Generate evaluation
            prompt = PromptTemplates.format_criterion_evaluation(
                criterion_text=criterion['text'],
                evidence_text=evidence_text
            )
            
            evaluation = llm.generate(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT_MARKER,
                temperature=0.3
            )
            
            results['criteria_evaluations'].append({
                'criterion_id': criterion['id'],
                'criterion_text': criterion['text'][:500],
                'evidence_count': len(retrieval_result.retrieved_chunks),
                'evaluation': evaluation
            })
            
            all_evaluations.append(f"### Criterion {criterion['id']}\n{evaluation}")
        
        # Generate summary
        logger.info("Generating overall summary...")
        
        summary_prompt = PromptTemplates.format_overall_summary(
            '\n\n---\n\n'.join(all_evaluations)
        )
        
        results['overall_summary'] = llm.generate(
            prompt=summary_prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT_SUMMARIZER,
            temperature=0.4
        )
    
    # Output results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    else:
        # Print to console
        print("\n" + "="*60)
        print("COURSEWORK FEEDBACK")
        print("="*60)
        
        print("\n## Overall Assessment\n")
        print(results['overall_summary'])
        
        print("\n" + "-"*60)
        
        for eval_data in results['criteria_evaluations']:
            print(f"\n## Criterion {eval_data['criterion_id']}")
            print(f"Evidence chunks: {eval_data['evidence_count']}")
            print("-"*40)
            print(eval_data['evaluation'])
    
    logger.info("Evaluation complete!")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Coursework Marker Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run the web UI
    python main.py ui
    
    # Evaluate from command line
    python main.py evaluate --criteria rubric.pdf --report student_work.docx
    
    # Save output to file
    python main.py evaluate --criteria rubric.pdf --report report.docx --output feedback.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch Streamlit UI')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation from command line')
    eval_parser.add_argument(
        '--criteria', '-c',
        required=True,
        help='Path to criteria/rubric document (PDF or DOCX)'
    )
    eval_parser.add_argument(
        '--report', '-r',
        required=True,
        help='Path to student report (DOCX)'
    )
    eval_parser.add_argument(
        '--output', '-o',
        help='Path for output JSON file (optional)'
    )
    eval_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.command == 'ui':
        run_ui()
    elif args.command == 'evaluate':
        run_evaluation(
            criteria_path=args.criteria,
            report_path=args.report,
            output_path=args.output,
            verbose=args.verbose
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
