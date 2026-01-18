"""
KSB Coursework Marker - Streamlit UI

Evaluates student coursework against KSB criteria with Pass/Merit/Referral grading.
Now with configurable hybrid search settings.
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
    KSBRubricParser, 
    KSBCriterion, 
    get_module_criteria,
    get_available_modules,
    AVAILABLE_MODULES
)
from src.prompts.ksb_templates import KSBPromptTemplates
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, 
    EMBEDDING_MODEL,
    RetrievalConfig, LLMConfig, SearchPresets
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
    /* Main app styling */
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #8b95a5;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Grade badges */
    .grade-pass {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .grade-merit {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .grade-referral {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151922 0%, #1a202c 100%);
        border-right: 1px solid #2d3748;
        min-width: 320px;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Module selector card */
    .module-card {
        background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%);
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .module-name {
        color: #667eea;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .module-desc {
        color: #8b95a5;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* KSB list styling - IMPROVED SIZES */
    .ksb-category {
        color: #667eea;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .ksb-item {
        color: #c9d1d9;
        font-size: 0.95rem;
        padding: 0.5rem 0;
        border-left: 3px solid #3d4756;
        padding-left: 0.75rem;
        margin: 0.4rem 0;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .ksb-item:hover {
        border-left-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    .ksb-code {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Status indicators */
    .status-ok {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        margin: 0.4rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        margin: 0.4rem 0;
    }
    
    /* Search settings card */
    .search-settings-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2530 100%);
        border: 1px solid #3d4756;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .search-preset-badge {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #1e2530;
        border: 2px dashed #3d4756;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:disabled {
        background: #2d3748;
        box-shadow: none;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #1e2530;
        border-radius: 8px;
        border: 1px solid #2d3748;
        font-size: 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }
    
    /* Info boxes */
    .stAlert {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #2d3748;
        margin: 1.5rem 0;
    }
    
    /* Summary stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2d3748;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .stat-number.merit { color: #3b82f6; }
    .stat-number.pass { color: #10b981; }
    .stat-number.referral { color: #ef4444; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* How to use list - IMPROVED */
    .how-to-item {
        color: #c9d1d9;
        font-size: 0.95rem;
        padding: 0.4rem 0;
        line-height: 1.5;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #667eea;
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
        'selected_module': 'DSP',
        # Search settings
        'search_preset': 'BALANCED',
        'use_hybrid': True,
        'semantic_weight': 0.6,
        'keyword_weight': 0.4,
        'similarity_threshold': 0.2,
        'report_top_k': 8,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_search_preset(preset_name: str):
    """Apply a search preset configuration."""
    presets = {
        'BALANCED': {'use_hybrid': True, 'semantic_weight': 0.6, 'keyword_weight': 0.4, 'similarity_threshold': 0.2, 'report_top_k': 8},
        'SEMANTIC_HEAVY': {'use_hybrid': True, 'semantic_weight': 0.8, 'keyword_weight': 0.2, 'similarity_threshold': 0.25, 'report_top_k': 6},
        'KEYWORD_HEAVY': {'use_hybrid': True, 'semantic_weight': 0.4, 'keyword_weight': 0.6, 'similarity_threshold': 0.15, 'report_top_k': 10},
        'SEMANTIC_ONLY': {'use_hybrid': False, 'semantic_weight': 1.0, 'keyword_weight': 0.0, 'similarity_threshold': 0.3, 'report_top_k': 6},
        'HIGH_RECALL': {'use_hybrid': True, 'semantic_weight': 0.5, 'keyword_weight': 0.5, 'similarity_threshold': 0.1, 'report_top_k': 12},
    }
    
    if preset_name in presets:
        preset = presets[preset_name]
        st.session_state.use_hybrid = preset['use_hybrid']
        st.session_state.semantic_weight = preset['semantic_weight']
        st.session_state.keyword_weight = preset['keyword_weight']
        st.session_state.similarity_threshold = preset['similarity_threshold']
        st.session_state.report_top_k = preset['report_top_k']


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
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        if suffix == '.docx':
            processor = DocxProcessor()
            doc = processor.process(tmp_path)
        elif suffix == '.pdf':
            processor = PDFProcessor()
            doc = processor.process(tmp_path)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None
        
        chunker = SmartChunker()
        chunks = chunker.chunk_report(doc.chunks, document_id="report")
        
        # Get chunking stats
        stats = chunker.get_chunking_stats(chunks)
        
        return {
            'chunks': chunks,
            'title': doc.title or uploaded_file.name,
            'filename': uploaded_file.name,
            'total_pages': doc.total_pages_estimate,
            'figures': getattr(doc, 'figures', {}),
            'raw_text': doc.raw_text,
            'chunking_stats': stats
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
        vector_store.clear_report()
        progress.progress(20, text="Embedding report chunks...")
        
        report_texts = [c.content for c in report_data['chunks']]
        report_embeddings = embedder.embed_documents(report_texts)
        progress.progress(70, text="Storing in vector database...")
        
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
    llm: OllamaClient,
    search_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate student work against a single KSB with configurable search."""
    
    # Create retriever with current search settings
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        report_top_k=search_settings['report_top_k'],
        max_context_tokens=RetrievalConfig.MAX_CONTEXT_TOKENS,
        similarity_threshold=search_settings['similarity_threshold'],
        use_hybrid=search_settings['use_hybrid'],
        semantic_weight=search_settings['semantic_weight'],
        keyword_weight=search_settings['keyword_weight']
    )
    
    query = f"{ksb.code} {ksb.title} {ksb.pass_criteria}"
    
    result = retriever.retrieve_for_criterion(query, ksb.code)
    evidence_text = retriever.format_context_for_llm(result)
    
    prompt = KSBPromptTemplates.format_ksb_evaluation(
        ksb_code=ksb.code,
        ksb_title=ksb.title,
        pass_criteria=ksb.pass_criteria,
        merit_criteria=ksb.merit_criteria,
        referral_criteria=ksb.referral_criteria,
        evidence_text=evidence_text
    )
    
    system_prompt = KSBPromptTemplates.get_system_prompt()
    evaluation = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=LLMConfig.EVALUATION_TEMPERATURE,
        max_tokens=1500
    )
    
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
        'evidence_count': len(result.retrieved_chunks),
        'search_strategy': result.search_strategy,
        'query_variations': len(result.query_variations)
    }


def extract_grade_from_evaluation(evaluation: str) -> str:
    """Extract the recommended grade from evaluation text."""
    import re
    
    match = re.search(
        r'Recommended Grade[:\s]+\*?\*?(PASS|MERIT|REFERRAL)\*?\*?',
        evaluation,
        re.IGNORECASE
    )
    
    if match:
        return match.group(1).upper()
    
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
    
    evals_text = ""
    for eval_data in ksb_evaluations:
        evals_text += f"\n\n{'='*60}\n"
        evals_text += f"## {eval_data['ksb_code']} - {eval_data['ksb_title']}\n"
        evals_text += f"**Recommended Grade: {eval_data['grade']}**\n\n"
        evals_text += eval_data['evaluation']
    
    prompt = KSBPromptTemplates.format_overall_summary(evals_text)
    system_prompt = KSBPromptTemplates.get_system_prompt()
    
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
    llm: OllamaClient,
    search_settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Generate feedback for all KSBs with configurable search."""
    
    ksb_evaluations = []
    total_ksbs = len(ksb_criteria)
    
    progress_bar = st.progress(0, text="Evaluating KSBs...")
    status_text = st.empty()
    
    for i, ksb in enumerate(ksb_criteria):
        status_text.markdown(f"**Evaluating {ksb.code}:** {ksb.title[:50]}...")
        
        try:
            eval_result = evaluate_ksb(ksb, embedder, vector_store, llm, search_settings)
            ksb_evaluations.append(eval_result)
        except Exception as e:
            logger.exception(f"Error evaluating {ksb.code}")
            ksb_evaluations.append({
                'ksb_code': ksb.code,
                'ksb_title': ksb.title,
                'ksb_category': ksb.category,
                'evaluation': f"Error during evaluation: {str(e)}",
                'grade': 'ERROR',
                'evidence_count': 0,
                'search_strategy': 'N/A',
                'query_variations': 0
            })
        
        progress_bar.progress((i + 1) / total_ksbs, 
                             text=f"Evaluated {i + 1}/{total_ksbs} KSBs")
    
    status_text.markdown("**Generating overall summary...**")
    overall_summary = generate_overall_summary(ksb_evaluations, llm)
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'ksb_evaluations': ksb_evaluations,
        'overall_summary': overall_summary,
        'search_settings': search_settings
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
    
    total = len(evaluations)
    merits = sum(1 for e in evaluations if e['grade'] == 'MERIT')
    passes = sum(1 for e in evaluations if e['grade'] == 'PASS')
    referrals = sum(1 for e in evaluations if e['grade'] == 'REFERRAL')
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); 
                border-radius: 16px; 
                padding: 1.5rem; 
                margin: 1rem 0;
                border: 1px solid #2d3748;">
        <h3 style="color: #e2e8f0; margin-bottom: 1rem; text-align: center;">📊 Assessment Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #667eea;">{total}</div>
            <div class="metric-label">Total KSBs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number merit">{merits}</div>
            <div class="metric-label">Merit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number pass">{passes}</div>
            <div class="metric-label">Pass</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number referral">{referrals}</div>
            <div class="metric-label">Referral</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    knowledge = [e for e in evaluations if e.get('ksb_category') == 'Knowledge']
    skills = [e for e in evaluations if e.get('ksb_category') == 'Skill']
    behaviours = [e for e in evaluations if e.get('ksb_category') == 'Behaviour']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #1e2530; border-radius: 10px; padding: 1rem; border: 1px solid #2d3748;">
            <p style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">📘 Knowledge</p>
        </div>
        """, unsafe_allow_html=True)
        for e in knowledge:
            grade_color = "#3b82f6" if e['grade'] == 'MERIT' else "#10b981" if e['grade'] == 'PASS' else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #2d3748;">
                <span style="color: #a0aec0;">{e['ksb_code']}</span>
                <span style="color: {grade_color}; font-weight: 600;">{e['grade']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1e2530; border-radius: 10px; padding: 1rem; border: 1px solid #2d3748;">
            <p style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">🔧 Skills</p>
        </div>
        """, unsafe_allow_html=True)
        for e in skills:
            grade_color = "#3b82f6" if e['grade'] == 'MERIT' else "#10b981" if e['grade'] == 'PASS' else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #2d3748;">
                <span style="color: #a0aec0;">{e['ksb_code']}</span>
                <span style="color: {grade_color}; font-weight: 600;">{e['grade']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #1e2530; border-radius: 10px; padding: 1rem; border: 1px solid #2d3748;">
            <p style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">💡 Behaviours</p>
        </div>
        """, unsafe_allow_html=True)
        for e in behaviours:
            grade_color = "#3b82f6" if e['grade'] == 'MERIT' else "#10b981" if e['grade'] == 'PASS' else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #2d3748;">
                <span style="color: #a0aec0;">{e['ksb_code']}</span>
                <span style="color: {grade_color}; font-weight: 600;">{e['grade']}</span>
            </div>
            """, unsafe_allow_html=True)


def display_feedback(results: Dict[str, Any]):
    """Display the generated feedback."""
    
    display_ksb_summary_table(results.get('ksb_evaluations', []))
    
    # Display search settings used
    if 'search_settings' in results:
        settings = results['search_settings']
        st.markdown(f"""
        <div style="background: #1a1f2e; border: 1px solid #3d4756; border-radius: 8px; padding: 0.75rem; margin: 1rem 0;">
            <span style="color: #8b95a5; font-size: 0.85rem;">
                🔍 Search: <b style="color: #667eea;">{'Hybrid' if settings['use_hybrid'] else 'Semantic Only'}</b> | 
                Semantic: <b>{int(settings['semantic_weight']*100)}%</b> | 
                Keyword: <b>{int(settings['keyword_weight']*100)}%</b> |
                Threshold: <b>{settings['similarity_threshold']}</b> |
                Top-K: <b>{settings['report_top_k']}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); 
                border-radius: 16px; 
                padding: 1.5rem; 
                margin: 1rem 0;
                border: 1px solid #2d3748;">
        <h3 style="color: #e2e8f0; margin-bottom: 0.5rem;">📝 Overall Assessment</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(results.get('overall_summary', 'No summary available.'))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); 
                border-radius: 16px; 
                padding: 1.5rem; 
                margin: 1rem 0;
                border: 1px solid #2d3748;">
        <h3 style="color: #e2e8f0; margin-bottom: 0.5rem;">📋 Detailed KSB Evaluations</h3>
        <p style="color: #8b95a5; font-size: 0.9rem;">Click on each KSB to view detailed feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    for eval_data in results.get('ksb_evaluations', []):
        grade = eval_data.get('grade', 'N/A')
        grade_color = (
            "🟢" if grade == 'MERIT' 
            else "🟡" if grade == 'PASS' 
            else "🔴"
        )
        
        evidence_info = f"{eval_data.get('evidence_count', 0)} chunks"
        if eval_data.get('query_variations'):
            evidence_info += f", {eval_data['query_variations']} queries"
        
        with st.expander(
            f"{grade_color} **{eval_data['ksb_code']}** - {eval_data['ksb_title'][:50]}... [{grade}]",
            expanded=False
        ):
            # Show search info
            st.markdown(f"""
            <div style="background: #1a1f2e; padding: 0.5rem; border-radius: 6px; margin-bottom: 1rem;">
                <span style="color: #8b95a5; font-size: 0.85rem;">
                    📊 Evidence: <b style="color: #667eea;">{evidence_info}</b> | 
                    Strategy: <b>{eval_data.get('search_strategy', 'N/A')}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**KSB Criteria:**")
            
            criteria_tabs = st.tabs(["Pass", "Merit", "Referral"])
            with criteria_tabs[0]:
                st.info(eval_data.get('pass_criteria', 'N/A'))
            with criteria_tabs[1]:
                st.success(eval_data.get('merit_criteria', 'N/A'))
            with criteria_tabs[2]:
                st.error(eval_data.get('referral_criteria', 'N/A'))
            
            st.markdown("---")
            st.markdown("**Evaluation:**")
            st.markdown(eval_data.get('evaluation', 'No evaluation available.'))


def render_search_settings():
    """Render the search settings panel in the sidebar."""
    st.markdown("## 🔍 Search Settings")
    
    # Preset selector
    preset_options = {
        'BALANCED': '⚖️ Balanced (60/40)',
        'SEMANTIC_HEAVY': '🧠 Semantic Heavy (80/20)',
        'KEYWORD_HEAVY': '🔤 Keyword Heavy (40/60)',
        'SEMANTIC_ONLY': '🎯 Semantic Only',
        'HIGH_RECALL': '📚 High Recall (50/50)',
        'CUSTOM': '⚙️ Custom'
    }
    
    selected_preset = st.selectbox(
        "Search Preset",
        options=list(preset_options.keys()),
        format_func=lambda x: preset_options[x],
        index=list(preset_options.keys()).index(st.session_state.get('search_preset', 'BALANCED')),
        help="Choose a pre-configured search strategy or customize your own"
    )
    
    # Apply preset if changed (and not custom)
    if selected_preset != st.session_state.search_preset:
        st.session_state.search_preset = selected_preset
        if selected_preset != 'CUSTOM':
            apply_search_preset(selected_preset)
            st.rerun()
    
    # Show current settings
    with st.expander("⚙️ Advanced Settings", expanded=(selected_preset == 'CUSTOM')):
        # Hybrid toggle
        use_hybrid = st.toggle(
            "Enable Hybrid Search",
            value=st.session_state.use_hybrid,
            help="Combine semantic (meaning) and keyword (BM25) search"
        )
        
        if use_hybrid != st.session_state.use_hybrid:
            st.session_state.use_hybrid = use_hybrid
            st.session_state.search_preset = 'CUSTOM'
        
        if use_hybrid:
            # Weight sliders
            st.markdown("**Search Weights**")
            
            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.semantic_weight,
                step=0.1,
                help="Weight for meaning-based similarity search"
            )
            
            if semantic_weight != st.session_state.semantic_weight:
                st.session_state.semantic_weight = semantic_weight
                st.session_state.keyword_weight = 1.0 - semantic_weight
                st.session_state.search_preset = 'CUSTOM'
            
            # Show keyword weight (auto-calculated)
            st.markdown(f"""
            <div style="background: #1a1f2e; padding: 0.5rem; border-radius: 6px; margin: 0.5rem 0;">
                <span style="color: #8b95a5; font-size: 0.85rem;">
                    Keyword Weight: <b style="color: #667eea;">{1.0 - semantic_weight:.1f}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="Minimum similarity score to include a chunk (lower = more results)"
        )
        
        if similarity_threshold != st.session_state.similarity_threshold:
            st.session_state.similarity_threshold = similarity_threshold
            st.session_state.search_preset = 'CUSTOM'
        
        # Top-K
        report_top_k = st.slider(
            "Results per KSB",
            min_value=3,
            max_value=15,
            value=st.session_state.report_top_k,
            step=1,
            help="Number of text chunks to retrieve for each KSB"
        )
        
        if report_top_k != st.session_state.report_top_k:
            st.session_state.report_top_k = report_top_k
            st.session_state.search_preset = 'CUSTOM'
    
    # Show current config summary
    st.markdown(f"""
    <div class="search-settings-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #8b95a5; font-size: 0.85rem;">Current Config</span>
            <span class="search-preset-badge">{st.session_state.search_preset}</span>
        </div>
        <div style="color: #c9d1d9; font-size: 0.85rem; line-height: 1.6;">
            {'🔀 Hybrid' if st.session_state.use_hybrid else '🎯 Semantic'} | 
            S:{int(st.session_state.semantic_weight*100)}% K:{int(st.session_state.keyword_weight*100)}%<br>
            Threshold: {st.session_state.similarity_threshold} | Top-K: {st.session_state.report_top_k}
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_current_search_settings() -> Dict[str, Any]:
    """Get current search settings from session state."""
    return {
        'use_hybrid': st.session_state.use_hybrid,
        'semantic_weight': st.session_state.semantic_weight,
        'keyword_weight': st.session_state.keyword_weight,
        'similarity_threshold': st.session_state.similarity_threshold,
        'report_top_k': st.session_state.report_top_k
    }


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📝 KSB Coursework Marker</p>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered KSB assessment with hybrid search</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        # Logo/Brand area
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🎓</span>
            <p style="color: #667eea; font-weight: 600; margin: 0.5rem 0 0 0; font-size: 1.2rem;">KSB Marker</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## ⚡ Status")
        
        # Load components
        llm = load_ollama_client()
        embedder = load_embedder()
        
        if llm:
            st.markdown('<div class="status-ok">✓ Ollama Connected</div>', unsafe_allow_html=True)
            st.session_state.ollama_connected = True
        else:
            st.markdown('<div class="status-error">✗ Ollama Disconnected</div>', unsafe_allow_html=True)
            st.caption("Run: `ollama serve`")
        
        if embedder:
            st.markdown(f'<div class="status-ok">✓ Embedder Ready ({embedder.embedding_dim}d)</div>', unsafe_allow_html=True)
            st.session_state.embedder_loaded = True
        else:
            st.markdown('<div class="status-error">✗ Embedder Error</div>', unsafe_allow_html=True)
        
        st.markdown("## 📚 Module")
        
        modules = get_available_modules()
        module_options = {code: info['name'] for code, info in modules.items()}
        
        selected_module = st.selectbox(
            "Select Module",
            options=list(module_options.keys()),
            format_func=lambda x: module_options[x],
            index=list(module_options.keys()).index(st.session_state.selected_module),
            help="Choose which module's KSB rubric to use",
            label_visibility="collapsed"
        )
        
        if selected_module != st.session_state.selected_module:
            st.session_state.selected_module = selected_module
            st.session_state.ksb_criteria = None
            st.session_state.feedback_results = None
            st.session_state.feedback_generated = False
            st.rerun()
        
        module_info = modules[selected_module]
        st.markdown(f"""
        <div class="module-card">
            <div class="module-name">{module_info['ksb_count']} KSBs to assess</div>
            <div class="module-desc">{module_info['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.ksb_criteria is None:
            st.session_state.ksb_criteria = get_module_criteria(selected_module)
        
        if st.session_state.ksb_criteria:
            with st.expander(f"📋 View {len(st.session_state.ksb_criteria)} KSBs", expanded=False):
                knowledge = [k for k in st.session_state.ksb_criteria if k.code.startswith('K')]
                skills = [k for k in st.session_state.ksb_criteria if k.code.startswith('S')]
                behaviours = [k for k in st.session_state.ksb_criteria if k.code.startswith('B')]
                
                if knowledge:
                    st.markdown('<p class="ksb-category">📘 Knowledge</p>', unsafe_allow_html=True)
                    for ksb in knowledge:
                        title_display = ksb.title if len(ksb.title) <= 60 else ksb.title[:57] + "..."
                        st.markdown(
                            f'<div class="ksb-item"><span class="ksb-code">{ksb.code}</span>: {title_display}</div>', 
                            unsafe_allow_html=True
                        )
                
                if skills:
                    st.markdown('<p class="ksb-category">🔧 Skills</p>', unsafe_allow_html=True)
                    for ksb in skills:
                        title_display = ksb.title if len(ksb.title) <= 60 else ksb.title[:57] + "..."
                        st.markdown(
                            f'<div class="ksb-item"><span class="ksb-code">{ksb.code}</span>: {title_display}</div>', 
                            unsafe_allow_html=True
                        )
                
                if behaviours:
                    st.markdown('<p class="ksb-category">💡 Behaviours</p>', unsafe_allow_html=True)
                    for ksb in behaviours:
                        title_display = ksb.title if len(ksb.title) <= 60 else ksb.title[:57] + "..."
                        st.markdown(
                            f'<div class="ksb-item"><span class="ksb-code">{ksb.code}</span>: {title_display}</div>', 
                            unsafe_allow_html=True
                        )
        
        # Search Settings Section
        render_search_settings()
        
        st.markdown("## 📖 How to Use")
        st.markdown("""
        <div class="how-to-item">1️⃣ Select module (DSP, MLCC, AIDI)</div>
        <div class="how-to-item">2️⃣ Configure search settings</div>
        <div class="how-to-item">3️⃣ Upload student report (DOCX/PDF)</div>
        <div class="how-to-item">4️⃣ Click Generate Assessment</div>
        <div class="how-to-item">5️⃣ Review KSB grades & feedback</div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ['report_data', 'feedback_results', 'feedback_generated', 'ksb_criteria']:
                st.session_state[key] = None if key != 'feedback_generated' else False
            st.rerun()
    
    # Main content area
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); 
                border-radius: 16px; 
                padding: 2rem; 
                margin: 1rem 0;
                border: 1px solid #2d3748;">
        <h2 style="color: #e2e8f0; margin-bottom: 0.5rem;">📄 Student Report</h2>
        <p style="color: #8b95a5; font-size: 0.9rem;">Upload the coursework document to begin assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload the student's coursework report",
        type=['docx', 'pdf'],
        help="Upload the student's submission in DOCX or PDF format",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        if (st.session_state.report_data is None or 
            st.session_state.report_data.get('filename') != uploaded_file.name):
            
            with st.spinner("Processing report..."):
                report_data = process_report(uploaded_file)
                
                if report_data:
                    st.session_state.report_data = report_data
                    st.session_state.feedback_generated = False
                    st.session_state.feedback_results = None
        
        if st.session_state.report_data:
            report = st.session_state.report_data
            
            st.markdown("""
            <div style="margin: 1.5rem 0;">
                <p style="color: #667eea; font-weight: 600; margin-bottom: 1rem;">📁 Document Loaded</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Document</div>
                    <div class="metric-value" style="font-size: 0.9rem; color: #e2e8f0;">{report['title'][:20]}...</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Pages</div>
                    <div class="metric-value">{report['total_pages']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Chunks</div>
                    <div class="metric-value">{len(report['chunks'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                stats = report.get('chunking_stats', {})
                keywords_count = stats.get('chunks_with_keywords', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">With Keywords</div>
                    <div class="metric-value">{keywords_count}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    can_generate = (
        st.session_state.report_data is not None and
        st.session_state.ksb_criteria is not None and
        st.session_state.ollama_connected and
        st.session_state.embedder_loaded
    )
    
    if not can_generate:
        st.markdown("""
        <div style="background: #2d3748; padding: 1rem; border-radius: 10px; text-align: center;">
            <p style="color: #a0aec0; margin: 0;">⚠️ Upload a report and ensure Ollama is connected to generate assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button(
        "🚀 Generate KSB Assessment",
        type="primary",
        disabled=not can_generate,
        use_container_width=True
    ):
        # Get current search settings
        search_settings = get_current_search_settings()
        
        tmpdir = tempfile.mkdtemp()
        vector_store = ChromaStore(persist_directory=tmpdir)
        
        st.markdown("### 📥 Indexing Report")
        success = index_report(
            st.session_state.report_data,
            embedder,
            vector_store
        )
        
        if success:
            st.markdown("### 🤖 Evaluating Against KSBs")
            
            # Show search config being used
            st.info(
                f"Using **{st.session_state.search_preset}** search: "
                f"{'Hybrid' if search_settings['use_hybrid'] else 'Semantic'} "
                f"(S:{int(search_settings['semantic_weight']*100)}% / K:{int(search_settings['keyword_weight']*100)}%) | "
                f"Threshold: {search_settings['similarity_threshold']} | "
                f"Top-K: {search_settings['report_top_k']}"
            )
            
            results = generate_feedback(
                st.session_state.ksb_criteria,
                embedder,
                vector_store,
                llm,
                search_settings
            )
            
            if results:
                st.session_state.feedback_results = results
                st.session_state.feedback_generated = True
                st.rerun()
    
    # Display results
    if st.session_state.get('feedback_generated') and st.session_state.get('feedback_results'):
        st.divider()
        display_feedback(st.session_state.feedback_results)
        
        st.divider()
        st.markdown("### 💾 Export Assessment")
        
        module_code = st.session_state.selected_module
        module_info = AVAILABLE_MODULES.get(module_code, {})
        module_name = module_info.get('name', module_code)
        
        # Include search settings in export
        search_settings = st.session_state.feedback_results.get('search_settings', {})
        
        export_text = f"# KSB Coursework Assessment Report\n\n"
        export_text += f"**Module:** {module_name}\n\n"
        export_text += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n"
        export_text += f"**Search Configuration:**\n"
        export_text += f"- Mode: {'Hybrid (BM25 + Semantic)' if search_settings.get('use_hybrid', True) else 'Semantic Only'}\n"
        export_text += f"- Semantic Weight: {search_settings.get('semantic_weight', 0.6)}\n"
        export_text += f"- Keyword Weight: {search_settings.get('keyword_weight', 0.4)}\n"
        export_text += f"- Similarity Threshold: {search_settings.get('similarity_threshold', 0.2)}\n"
        export_text += f"- Results per KSB: {search_settings.get('report_top_k', 8)}\n\n"
        
        export_text += "---\n\n## KSB Grade Summary\n\n"
        export_text += "| KSB | Title | Grade | Evidence |\n"
        export_text += "|-----|-------|-------|----------|\n"
        for eval_data in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"| {eval_data['ksb_code']} | {eval_data['ksb_title'][:35]}... | {eval_data['grade']} | {eval_data.get('evidence_count', 0)} chunks |\n"
        
        export_text += "\n\n## Overall Assessment\n\n"
        export_text += st.session_state.feedback_results.get('overall_summary', '') + "\n\n"
        
        for eval_data in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"---\n\n## {eval_data['ksb_code']} - {eval_data['ksb_title']}\n\n"
            export_text += f"**Grade: {eval_data['grade']}** | "
            export_text += f"Evidence: {eval_data.get('evidence_count', 0)} chunks | "
            export_text += f"Search: {eval_data.get('search_strategy', 'N/A')}\n\n"
            export_text += eval_data.get('evaluation', '') + "\n\n"
        
        filename = f"ksb_assessment_{module_code.lower()}_{time.strftime('%Y%m%d')}.md"
        
        st.download_button(
            label="📥 Download Assessment (Markdown)",
            data=export_text,
            file_name=filename,
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()