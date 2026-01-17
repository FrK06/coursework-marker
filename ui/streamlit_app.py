"""
Streamlit UI for the Coursework Marker Assistant.

A clean, functional interface for:
- Uploading marking criteria and student reports
- Generating structured feedback
- Displaying results in an organized manner
"""
import streamlit as st
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processing import DocxProcessor, PDFProcessor, ImageExtractor
from src.chunking import SmartChunker
from src.embeddings import Embedder
from src.vector_store import ChromaStore
from src.retrieval import Retriever
from src.llm import OllamaClient
from src.prompts import PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Coursework Marker Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .criterion-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .evidence-quote {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-style: italic;
        margin: 0.5rem 0;
    }
    .gap-item {
        color: #d63384;
    }
    .improvement-item {
        color: #198754;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'criteria_processed' not in st.session_state:
        st.session_state.criteria_processed = False
    if 'report_processed' not in st.session_state:
        st.session_state.report_processed = False
    if 'feedback_generated' not in st.session_state:
        st.session_state.feedback_generated = False
    if 'feedback_results' not in st.session_state:
        st.session_state.feedback_results = None
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None


@st.cache_resource
def load_embedder():
    """Load the embedding model (cached)."""
    return Embedder()


@st.cache_resource
def load_llm_client():
    """Load the Ollama client (cached)."""
    try:
        return OllamaClient()
    except ConnectionError as e:
        st.error(str(e))
        return None


def process_criteria_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process an uploaded criteria file."""
    try:
        # Save to temp file
        suffix = Path(uploaded_file.name).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Process based on file type
        if suffix == '.docx':
            processor = DocxProcessor()
            doc = processor.process(tmp_path)
            chunks = doc.chunks
            raw_text = doc.raw_text
        elif suffix == '.pdf':
            processor = PDFProcessor()
            doc = processor.process(tmp_path)
            chunks = doc.chunks
            raw_text = doc.raw_text
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None
        
        # Chunk the document
        chunker = SmartChunker()
        text_chunks = chunker.chunk_criteria(chunks, "criteria")
        
        return {
            'chunks': text_chunks,
            'raw_text': raw_text,
            'stats': chunker.get_chunking_stats(text_chunks)
        }
        
    except Exception as e:
        logger.exception("Error processing criteria file")
        st.error(f"Error processing criteria: {str(e)}")
        return None


def process_report_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process an uploaded student report file."""
    try:
        # Save to temp file
        suffix = Path(uploaded_file.name).suffix.lower()
        
        if suffix != '.docx':
            st.error("Student reports must be .docx files")
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Process document
        processor = DocxProcessor()
        doc = processor.process(tmp_path)
        
        # Chunk the document
        chunker = SmartChunker()
        text_chunks = chunker.chunk_report(doc.chunks, "report")
        
        # Extract images if any
        image_extractor = ImageExtractor()
        images = image_extractor.process_images(
            doc.figures,
            doc.figure_captions,
            doc.chunks
        )
        
        return {
            'chunks': text_chunks,
            'raw_text': doc.raw_text,
            'images': images,
            'title': doc.title,
            'stats': chunker.get_chunking_stats(text_chunks)
        }
        
    except Exception as e:
        logger.exception("Error processing report file")
        st.error(f"Error processing report: {str(e)}")
        return None


def index_documents(
    criteria_data: Dict[str, Any],
    report_data: Dict[str, Any],
    embedder: Embedder,
    vector_store: ChromaStore
):
    """Index processed documents into vector store."""
    progress = st.progress(0, text="Indexing documents...")
    
    try:
        # Clear existing data
        vector_store.clear_all()
        progress.progress(10, text="Cleared existing index...")
        
        # Embed criteria chunks
        criteria_texts = [c.content for c in criteria_data['chunks']]
        criteria_embeddings = embedder.embed_documents(criteria_texts)
        progress.progress(40, text="Embedded criteria...")
        
        # Index criteria
        criteria_dicts = [c.to_dict() for c in criteria_data['chunks']]
        vector_store.add_criteria(criteria_dicts, criteria_embeddings)
        progress.progress(50, text="Indexed criteria...")
        
        # Embed report chunks
        report_texts = [c.content for c in report_data['chunks']]
        report_embeddings = embedder.embed_documents(report_texts)
        progress.progress(80, text="Embedded report...")
        
        # Index report
        report_dicts = [c.to_dict() for c in report_data['chunks']]
        vector_store.add_report(report_dicts, report_embeddings)
        progress.progress(100, text="Indexing complete!")
        
        time.sleep(0.5)
        progress.empty()
        
        return True
        
    except Exception as e:
        logger.exception("Error indexing documents")
        st.error(f"Error indexing: {str(e)}")
        return False


def generate_feedback(
    embedder: Embedder,
    vector_store: ChromaStore,
    llm: OllamaClient
) -> Optional[Dict[str, Any]]:
    """Generate feedback using RAG pipeline."""
    
    # Create retriever
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        report_top_k=5,
        max_context_tokens=2000
    )
    
    # Get criteria list
    criteria_list = retriever.extract_criteria_list()
    
    if not criteria_list:
        st.warning("Could not extract individual criteria. Using general evaluation.")
        criteria_list = [{'id': 'general', 'text': 'Overall assessment'}]
    
    results = {
        'criteria_evaluations': [],
        'overall_summary': None
    }
    
    # Progress tracking
    total_steps = len(criteria_list) + 1  # +1 for summary
    progress = st.progress(0, text="Generating feedback...")
    
    # Evaluate each criterion
    all_evaluations = []
    
    for i, criterion in enumerate(criteria_list):
        progress.progress(
            (i + 1) / total_steps,
            text=f"Evaluating criterion {criterion['id']}..."
        )
        
        # Retrieve evidence
        retrieval_result = retriever.retrieve_for_criterion(
            criterion['text'],
            criterion['id']
        )
        
        # Format context
        evidence_text = retriever.format_context_for_llm(retrieval_result)
        
        # Generate evaluation
        prompt = PromptTemplates.format_criterion_evaluation(
            criterion_text=criterion['text'],
            evidence_text=evidence_text
        )
        
        try:
            evaluation = llm.generate(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT_MARKER,
                temperature=0.3,
                max_tokens=1024
            )
            
            results['criteria_evaluations'].append({
                'criterion_id': criterion['id'],
                'criterion_text': criterion['text'],
                'evidence_count': len(retrieval_result.retrieved_chunks),
                'evaluation': evaluation
            })
            
            all_evaluations.append(
                f"### Criterion {criterion['id']}\n{evaluation}"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating criterion {criterion['id']}: {e}")
            results['criteria_evaluations'].append({
                'criterion_id': criterion['id'],
                'criterion_text': criterion['text'],
                'evidence_count': 0,
                'evaluation': f"Error generating evaluation: {str(e)}"
            })
    
    # Generate overall summary
    progress.progress(1.0, text="Generating overall summary...")
    
    try:
        summary_prompt = PromptTemplates.format_overall_summary(
            '\n\n---\n\n'.join(all_evaluations)
        )
        
        results['overall_summary'] = llm.generate(
            prompt=summary_prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT_SUMMARIZER,
            temperature=0.4,
            max_tokens=1024
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        results['overall_summary'] = f"Error generating summary: {str(e)}"
    
    progress.empty()
    
    return results


def display_feedback(results: Dict[str, Any]):
    """Display the generated feedback in a structured format."""
    
    # Overall Summary
    st.markdown("## 📊 Overall Assessment")
    st.markdown(results.get('overall_summary', 'No summary available'))
    
    st.divider()
    
    # Per-criterion evaluations
    st.markdown("## 📋 Criterion-by-Criterion Evaluation")
    
    for eval_data in results.get('criteria_evaluations', []):
        # Format criterion ID nicely (handle KSBs like "K2,K16,S16")
        criterion_id = eval_data['criterion_id']
        if ',' in criterion_id:
            criterion_display = f"KSBs: {criterion_id}"
        elif criterion_id == 'general':
            criterion_display = "General Requirements"
        else:
            criterion_display = f"Criterion {criterion_id}"
        
        with st.expander(
            f"**{criterion_display}** "
            f"({eval_data['evidence_count']} evidence chunks)",
            expanded=False
        ):
            # Show full criterion text (use text_area for long content)
            st.markdown("**Criterion Requirements:**")
            criterion_text = eval_data['criterion_text']
            if len(criterion_text) > 1000:
                st.text_area(
                    "Full requirements (scroll to read)",
                    criterion_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.info(criterion_text)
            
            st.markdown("---")
            
            # Show evaluation
            st.markdown(eval_data['evaluation'])


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📝 Coursework Marker Assistant</p>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered feedback generation for student coursework</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check Ollama status
        llm = load_llm_client()
        if llm and llm.is_available():
            st.success("✅ Ollama connected")
        else:
            st.error("❌ Ollama not available")
            st.markdown("""
            Please ensure Ollama is running:
            ```bash
            ollama serve
            ollama pull gemma3:4b
            ```
            """)
        
        st.divider()
        
        # Load embedder
        with st.spinner("Loading embedding model..."):
            embedder = load_embedder()
        st.success(f"✅ Embedder loaded ({embedder.embedding_dim}d)")
        
        st.divider()
        
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Upload your marking criteria (PDF or DOCX)
        2. Upload the student's report (DOCX)
        3. Click "Generate Feedback"
        4. Review the structured feedback
        """)
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset All", type="secondary"):
            for key in ['criteria_processed', 'report_processed', 
                       'feedback_generated', 'feedback_results',
                       'criteria_data', 'report_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📑 Marking Criteria")
        criteria_file = st.file_uploader(
            "Upload criteria document",
            type=['pdf', 'docx'],
            key="criteria_upload",
            help="Upload your marking rubric or criteria document"
        )
        
        if criteria_file:
            if 'criteria_data' not in st.session_state:
                with st.spinner("Processing criteria..."):
                    criteria_data = process_criteria_file(criteria_file)
                    if criteria_data:
                        st.session_state.criteria_data = criteria_data
                        st.session_state.criteria_processed = True
            
            if st.session_state.get('criteria_processed'):
                stats = st.session_state.criteria_data['stats']
                st.success(
                    f"✅ Processed: {stats['total_chunks']} chunks, "
                    f"~{stats['total_tokens']} tokens"
                )
    
    with col2:
        st.markdown("### 📄 Student Report")
        report_file = st.file_uploader(
            "Upload student report",
            type=['docx'],
            key="report_upload",
            help="Upload the student's coursework (DOCX format)"
        )
        
        if report_file:
            if 'report_data' not in st.session_state:
                with st.spinner("Processing report..."):
                    report_data = process_report_file(report_file)
                    if report_data:
                        st.session_state.report_data = report_data
                        st.session_state.report_processed = True
            
            if st.session_state.get('report_processed'):
                stats = st.session_state.report_data['stats']
                images = st.session_state.report_data.get('images', {})
                st.success(
                    f"✅ Processed: {stats['total_chunks']} chunks, "
                    f"~{stats['total_tokens']} tokens, "
                    f"{len(images)} figures"
                )
    
    st.divider()
    
    # Generate feedback button
    can_generate = (
        st.session_state.get('criteria_processed') and 
        st.session_state.get('report_processed') and
        llm is not None
    )
    
    if st.button(
        "🚀 Generate Feedback",
        type="primary",
        disabled=not can_generate,
        use_container_width=True
    ):
        # Initialize vector store with a temp directory (no auto-cleanup to avoid Windows file locking)
        tmpdir = tempfile.mkdtemp()
        vector_store = ChromaStore(persist_directory=tmpdir)
        
        # Index documents
        st.markdown("### 📥 Indexing Documents")
        success = index_documents(
            st.session_state.criteria_data,
            st.session_state.report_data,
            embedder,
            vector_store
        )
        
        if success:
            st.markdown("### 🤖 Generating Feedback")
            st.info(
                "This may take several minutes on CPU. "
                "Each criterion is evaluated separately."
            )
            
            results = generate_feedback(embedder, vector_store, llm)
            
            if results:
                st.session_state.feedback_results = results
                st.session_state.feedback_generated = True
                st.rerun()
    
    # Display results if available
    if st.session_state.get('feedback_generated') and st.session_state.get('feedback_results'):
        st.divider()
        display_feedback(st.session_state.feedback_results)
        
        # Download option
        st.divider()
        st.markdown("### 💾 Export Feedback")
        
        # Create downloadable text
        export_text = "# Coursework Feedback Report\n\n"
        export_text += "## Overall Assessment\n\n"
        export_text += st.session_state.feedback_results.get('overall_summary', '') + "\n\n"
        
        for eval_data in st.session_state.feedback_results.get('criteria_evaluations', []):
            export_text += f"---\n\n## Criterion {eval_data['criterion_id']}\n\n"
            export_text += eval_data.get('evaluation', '') + "\n\n"
        
        st.download_button(
            label="📥 Download Feedback (Markdown)",
            data=export_text,
            file_name="coursework_feedback.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
