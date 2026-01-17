"""
KSB Coursework Marker - Streamlit UI

Evaluates student coursework against KSB criteria with Pass/Merit/Referral grading.
"""
import streamlit as st
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processing import DocxProcessor, PDFProcessor
from src.chunking import SmartChunker
from src.embeddings import Embedder
from src.vector_store import ChromaStore
from src.retrieval import Retriever
from src.llm import OllamaClient
from src.criteria.ksb_parser import (
    KSBRubricParser,                                                                                      # "KSBRubricParser" is not accessed
    KSBCriterion, 
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)
from src.prompts.ksb_templates import KSBPromptTemplates
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, 
    EMBEDDING_MODEL,
    RetrievalConfig, LLMConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="KSB Coursework Marker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .grade-pass {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-weight: 600;
    }
    .grade-merit {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-weight: 600;
    }
    .grade-referral {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-weight: 600;
    }
    .ksb-card {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'report_data': None,
        'ksb_criteria': None,
        'feedback_results': None,
        'feedback_generated': False,
        'ollama_connected': False,
        'embedder_loaded': False,
        'selected_module': 'MLCC',  # Default module
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_ollama_client():
    """Load and cache the Ollama client."""
    try:
        client = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout=120
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return None


@st.cache_resource
def load_embedder():
    """Load and cache the embedding model."""
    try:
        return Embedder(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        return None


def process_report(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process uploaded student report."""
    if uploaded_file is None:
        return None
    
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
        elif suffix == '.pdf':
            processor = PDFProcessor()
            doc = processor.process(tmp_path)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None
        
        # Chunk the document
        chunker = SmartChunker()
        chunks = chunker.chunk_report(doc.chunks, document_id="report")
        
        return {
            'chunks': chunks,
            'title': doc.title or uploaded_file.name,
            'filename': uploaded_file.name,  # Store original filename for comparison
            'total_pages': doc.total_pages_estimate,
            'figures': getattr(doc, 'figures', {}),
            'raw_text': doc.raw_text
        }
        
    except Exception as e:
        logger.exception("Error processing report")
        st.error(f"Error processing file: {str(e)}")
        return None


def index_report(
    report_data: Dict[str, Any],
    embedder: Embedder,
    vector_store: ChromaStore
) -> bool:
    """Index the report into vector store."""
    progress = st.progress(0, text="Indexing report...")
    
    try:
        # Clear existing data
        vector_store.clear_report()
        progress.progress(20, text="Embedding report chunks...")
        
        # Embed report chunks
        report_texts = [c.content for c in report_data['chunks']]
        report_embeddings = embedder.embed_documents(report_texts)
        progress.progress(70, text="Storing in vector database...")
        
        # Index report
        report_dicts = [c.to_dict() for c in report_data['chunks']]
        vector_store.add_report(report_dicts, report_embeddings)
        progress.progress(100, text="Indexing complete!")
        
        time.sleep(0.3)
        progress.empty()
        return True
        
    except Exception as e:
        logger.exception("Error indexing report")
        st.error(f"Error indexing: {str(e)}")
        return False


def evaluate_ksb(
    ksb: KSBCriterion,
    embedder: Embedder,
    vector_store: ChromaStore,
    llm: OllamaClient
) -> Dict[str, Any]:
    """Evaluate student work against a single KSB."""
    
    # Create retriever
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        report_top_k=RetrievalConfig.REPORT_TOP_K,
        max_context_tokens=RetrievalConfig.MAX_CONTEXT_TOKENS,
        similarity_threshold=RetrievalConfig.SIMILARITY_THRESHOLD
    )
    
    # Build query from KSB content
    query = f"{ksb.code} {ksb.title} {ksb.pass_criteria}"
    
    # Retrieve relevant evidence
    result = retriever.retrieve_for_criterion(query, ksb.code)
    evidence_text = retriever.format_context_for_llm(result)
    
    # Format prompt
    prompt = KSBPromptTemplates.format_ksb_evaluation(
        ksb_code=ksb.code,
        ksb_title=ksb.title,
        pass_criteria=ksb.pass_criteria,
        merit_criteria=ksb.merit_criteria,
        referral_criteria=ksb.referral_criteria,
        evidence_text=evidence_text
    )
    
    # Generate evaluation
    system_prompt = KSBPromptTemplates.get_system_prompt()
    evaluation = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=LLMConfig.EVALUATION_TEMPERATURE,
        max_tokens=1500
    )
    
    # Extract recommended grade from evaluation
    grade = extract_grade_from_evaluation(evaluation)
    
    return {
        'ksb_code': ksb.code,
        'ksb_title': ksb.title,
        'ksb_category': ksb.category,
        'pass_criteria': ksb.pass_criteria,
        'merit_criteria': ksb.merit_criteria,
        'referral_criteria': ksb.referral_criteria,
        'evaluation': evaluation,
        'grade': grade,
        'evidence_count': len(result.retrieved_chunks)
    }


def extract_grade_from_evaluation(evaluation: str) -> str:
    """Extract the recommended grade from evaluation text."""
    import re
    
    # Look for "Recommended Grade: PASS/MERIT/REFERRAL"
    match = re.search(
        r'Recommended Grade[:\s]+\*?\*?(PASS|MERIT|REFERRAL)\*?\*?',
        evaluation,
        re.IGNORECASE
    )
    
    if match:
        return match.group(1).upper()
    
    # Fallback: look for grade mentions
    eval_upper = evaluation.upper()
    if 'REFERRAL' in eval_upper and 'NOT MET' in eval_upper:
        return 'REFERRAL'
    elif 'MERIT' in eval_upper and 'EXCEEDS' in eval_upper:
        return 'MERIT'
    else:
        return 'PASS'


def generate_overall_summary(
    ksb_evaluations: List[Dict[str, Any]],
    llm: OllamaClient
) -> str:
    """Generate overall summary from KSB evaluations."""
    
    # Compile evaluations text
    evals_text = ""
    for eval_data in ksb_evaluations:
        evals_text += f"\n\n{'='*60}\n"
        evals_text += f"## {eval_data['ksb_code']} - {eval_data['ksb_title']}\n"
        evals_text += f"**Recommended Grade: {eval_data['grade']}**\n\n"
        evals_text += eval_data['evaluation']
    
    # Format prompt
    prompt = KSBPromptTemplates.format_overall_summary(evals_text)
    system_prompt = KSBPromptTemplates.get_system_prompt()
    
    # Generate summary
    summary = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=LLMConfig.SUMMARY_TEMPERATURE,
        max_tokens=2000
    )
    
    return summary


def generate_feedback(
    ksb_criteria: List[KSBCriterion],
    embedder: Embedder,
    vector_store: ChromaStore,
    llm: OllamaClient
) -> Optional[Dict[str, Any]]:
    """Generate feedback for all KSBs."""
    
    ksb_evaluations = []
    total_ksbs = len(ksb_criteria)
    
    progress_bar = st.progress(0, text="Evaluating KSBs...")
    status_text = st.empty()
    
    for i, ksb in enumerate(ksb_criteria):
        status_text.markdown(f"**Evaluating {ksb.code}:** {ksb.title[:50]}...")
        
        try:
            eval_result = evaluate_ksb(ksb, embedder, vector_store, llm)
            ksb_evaluations.append(eval_result)
        except Exception as e:
            logger.exception(f"Error evaluating {ksb.code}")
            ksb_evaluations.append({
                'ksb_code': ksb.code,
                'ksb_title': ksb.title,
                'ksb_category': ksb.category,
                'evaluation': f"Error during evaluation: {str(e)}",
                'grade': 'ERROR',
                'evidence_count': 0
            })
        
        progress_bar.progress((i + 1) / total_ksbs, 
                             text=f"Evaluated {i + 1}/{total_ksbs} KSBs")
    
    # Generate overall summary
    status_text.markdown("**Generating overall summary...**")
    overall_summary = generate_overall_summary(ksb_evaluations, llm)
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'ksb_evaluations': ksb_evaluations,
        'overall_summary': overall_summary
    }


def display_grade_badge(grade: str):
    """Display a colored grade badge."""
    if grade == 'MERIT':
        st.markdown(f'<span class="grade-merit">MERIT</span>', unsafe_allow_html=True)
    elif grade == 'PASS':
        st.markdown(f'<span class="grade-pass">PASS</span>', unsafe_allow_html=True)
    elif grade == 'REFERRAL':
        st.markdown(f'<span class="grade-referral">REFERRAL</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**{grade}**')


def display_ksb_summary_table(evaluations: List[Dict[str, Any]]):
    """Display a summary table of all KSB grades."""
    st.markdown("### 📊 KSB Grade Summary")
    
    # Group by category
    knowledge = [e for e in evaluations if e.get('ksb_category') == 'Knowledge']
    skills = [e for e in evaluations if e.get('ksb_category') == 'Skill']
    behaviours = [e for e in evaluations if e.get('ksb_category') == 'Behaviour']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Knowledge (K)**")
        for e in knowledge:
            grade_emoji = "🟢" if e['grade'] == 'MERIT' else "🟡" if e['grade'] == 'PASS' else "🔴"
            st.markdown(f"{grade_emoji} **{e['ksb_code']}**: {e['grade']}")
    
    with col2:
        st.markdown("**Skills (S)**")
        for e in skills:
            grade_emoji = "🟢" if e['grade'] == 'MERIT' else "🟡" if e['grade'] == 'PASS' else "🔴"
            st.markdown(f"{grade_emoji} **{e['ksb_code']}**: {e['grade']}")
    
    with col3:
        st.markdown("**Behaviours (B)**")
        for e in behaviours:
            grade_emoji = "🟢" if e['grade'] == 'MERIT' else "🟡" if e['grade'] == 'PASS' else "🔴"
            st.markdown(f"{grade_emoji} **{e['ksb_code']}**: {e['grade']}")
    
    # Summary stats
    st.markdown("---")
    total = len(evaluations)
    merits = sum(1 for e in evaluations if e['grade'] == 'MERIT')
    passes = sum(1 for e in evaluations if e['grade'] == 'PASS')
    referrals = sum(1 for e in evaluations if e['grade'] == 'REFERRAL')
    
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total KSBs", total)
    stat_cols[1].metric("Merit", merits, delta=None)
    stat_cols[2].metric("Pass", passes, delta=None)
    stat_cols[3].metric("Referral", referrals, delta=None if referrals == 0 else f"-{referrals}")


def display_feedback(results: Dict[str, Any]):
    """Display the generated feedback."""
    
    # Display KSB summary table first
    display_ksb_summary_table(results.get('ksb_evaluations', []))
    
    st.divider()
    
    # Display overall summary
    st.markdown("## 📊 Overall Assessment")
    st.markdown(results.get('overall_summary', 'No summary available.'))
    
    st.divider()
    
    # Display per-KSB evaluations
    st.markdown("## 📋 KSB-by-KSB Evaluation")
    
    for eval_data in results.get('ksb_evaluations', []):
        grade = eval_data.get('grade', 'N/A')
        grade_color = (
            "🟢" if grade == 'MERIT' 
            else "🟡" if grade == 'PASS' 
            else "🔴"
        )
        
        with st.expander(
            f"{grade_color} **{eval_data['ksb_code']}** - {eval_data['ksb_title'][:50]}... [{grade}]",
            expanded=False
        ):
            # Show criteria
            st.markdown("**KSB Criteria:**")
            
            criteria_tabs = st.tabs(["Pass", "Merit", "Referral"])
            with criteria_tabs[0]:
                st.info(eval_data.get('pass_criteria', 'N/A'))
            with criteria_tabs[1]:
                st.success(eval_data.get('merit_criteria', 'N/A'))
            with criteria_tabs[2]:
                st.error(eval_data.get('referral_criteria', 'N/A'))
            
            st.markdown("---")
            st.markdown(f"**Evidence chunks found:** {eval_data.get('evidence_count', 0)}")
            st.markdown("---")
            
            # Show evaluation
            st.markdown("**Evaluation:**")
            st.markdown(eval_data.get('evaluation', 'No evaluation available.'))


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📝 KSB Coursework Marker</p>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered KSB assessment with Pass/Merit/Referral grading</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Load components
        llm = load_ollama_client()
        embedder = load_embedder()
        
        if llm:
            st.success("✅ Ollama connected")
            st.session_state.ollama_connected = True
        else:
            st.error("❌ Ollama not connected")
            st.info("Make sure Ollama is running with: `ollama serve`")
        
        if embedder:
            st.success(f"✅ Embedder loaded ({embedder.embedding_dim}d)")
            st.session_state.embedder_loaded = True
        else:
            st.error("❌ Embedder failed to load")
        
        st.divider()
        
        # Module Selection
        st.markdown("## 📚 Module Selection")
        
        # Get available modules
        modules = get_available_modules()
        module_options = {code: info['name'] for code, info in modules.items()}
        
        # Module selector
        selected_module = st.selectbox(
            "Select Module",
            options=list(module_options.keys()),
            format_func=lambda x: module_options[x],
            index=list(module_options.keys()).index(st.session_state.selected_module),
            help="Choose which module's KSB rubric to use for assessment"
        )
        
        # Update if module changed
        if selected_module != st.session_state.selected_module:
            st.session_state.selected_module = selected_module
            st.session_state.ksb_criteria = None  # Reset criteria
            st.session_state.feedback_results = None
            st.session_state.feedback_generated = False
            st.rerun()
        
        # Show module info
        module_info = modules[selected_module]
        st.info(f"**{module_info['description']}**\n\n{module_info['ksb_count']} KSBs to assess")
        
        # Load criteria for selected module
        if st.session_state.ksb_criteria is None:
            st.session_state.ksb_criteria = get_module_criteria(selected_module)
        
        if st.session_state.ksb_criteria:
            st.success(f"✅ {len(st.session_state.ksb_criteria)} KSBs loaded")
            
            # Show KSB list grouped by category
            with st.expander("View KSB List"):
                knowledge = [k for k in st.session_state.ksb_criteria if k.code.startswith('K')]
                skills = [k for k in st.session_state.ksb_criteria if k.code.startswith('S')]
                behaviours = [k for k in st.session_state.ksb_criteria if k.code.startswith('B')]
                
                if knowledge:
                    st.markdown("**Knowledge:**")
                    for ksb in knowledge:
                        st.markdown(f"- {ksb.code}: {ksb.title[:35]}...")
                
                if skills:
                    st.markdown("**Skills:**")
                    for ksb in skills:
                        st.markdown(f"- {ksb.code}: {ksb.title[:35]}...")
                
                if behaviours:
                    st.markdown("**Behaviours:**")
                    for ksb in behaviours:
                        st.markdown(f"- {ksb.code}: {ksb.title[:35]}...")
        
        st.divider()
        
        # Instructions
        st.markdown("## 📖 How to Use")
        st.markdown("""
        1. **Select module** (MLCC or AIDI)
        2. Upload the student's report (DOCX/PDF)
        3. Click "Generate KSB Assessment"
        4. Review grades and feedback for each KSB
        5. Download the complete assessment
        """)
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ['report_data', 'feedback_results', 'feedback_generated', 'ksb_criteria']:
                st.session_state[key] = None if key != 'feedback_generated' else False
            st.rerun()
    
    # Main content
    st.markdown("## 📄 Student Report")
    
    uploaded_file = st.file_uploader(
        "Upload the student's coursework report",
        type=['docx', 'pdf'],
        help="Upload the student's submission in DOCX or PDF format"
    )
    
    if uploaded_file:
        # Process if new file (compare by filename, not document title)
        if (st.session_state.report_data is None or 
            st.session_state.report_data.get('filename') != uploaded_file.name):
            
            with st.spinner("Processing report..."):
                report_data = process_report(uploaded_file)
                
                if report_data:
                    st.session_state.report_data = report_data
                    st.session_state.feedback_generated = False
                    st.session_state.feedback_results = None
        
        # Show report info
        if st.session_state.report_data:
            report = st.session_state.report_data
            col1, col2, col3 = st.columns(3)
            col1.metric("Document", report['title'][:30] + "...")
            col2.metric("Pages (est.)", report['total_pages'])
            col3.metric("Chunks", len(report['chunks']))
    
    st.divider()
    
    # Generate button
    can_generate = (
        st.session_state.report_data is not None and
        st.session_state.ksb_criteria is not None and
        st.session_state.ollama_connected and
        st.session_state.embedder_loaded
    )
    
    if not can_generate:
        st.warning("Upload a report and ensure Ollama is connected to generate assessment.")
    
    if st.button(
        "🚀 Generate KSB Assessment",
        type="primary",
        disabled=not can_generate,
        use_container_width=True
    ):
        # Create temp directory for vector store
        tmpdir = tempfile.mkdtemp()
        vector_store = ChromaStore(persist_directory=tmpdir)
        
        # Index report
        st.markdown("### 📥 Indexing Report")
        success = index_report(
            st.session_state.report_data,
            embedder,
            vector_store
        )
        
        if success:
            st.markdown("### 🤖 Evaluating Against KSBs")
            st.info(
                f"Evaluating {len(st.session_state.ksb_criteria)} KSBs. "
                "This may take several minutes on CPU."
            )
            
            results = generate_feedback(
                st.session_state.ksb_criteria,
                embedder,
                vector_store,
                llm
            )
            
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
        st.markdown("### 💾 Export Assessment")
        
        # Get module info
        module_code = st.session_state.selected_module
        module_info = AVAILABLE_MODULES.get(module_code, {})
        module_name = module_info.get('name', module_code)
        
        # Create downloadable text
        export_text = f"# KSB Coursework Assessment Report\n\n"
        export_text += f"**Module:** {module_name}\n\n"
        export_text += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Add summary table
        export_text += "---\n\n## KSB Grade Summary\n\n"
        export_text += "| KSB | Title | Grade |\n"
        export_text += "|-----|-------|-------|\n"
        for eval_data in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"| {eval_data['ksb_code']} | {eval_data['ksb_title'][:40]}... | {eval_data['grade']} |\n"
        
        export_text += "\n\n## Overall Assessment\n\n"
        export_text += st.session_state.feedback_results.get('overall_summary', '') + "\n\n"
        
        for eval_data in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"---\n\n## {eval_data['ksb_code']} - {eval_data['ksb_title']}\n\n"
            export_text += f"**Grade: {eval_data['grade']}**\n\n"
            export_text += eval_data.get('evaluation', '') + "\n\n"
        
        # Generate filename with module
        filename = f"ksb_assessment_{module_code.lower()}_{time.strftime('%Y%m%d')}.md"
        
        st.download_button(
            label="📥 Download Assessment (Markdown)",
            data=export_text,
            file_name=filename,
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()