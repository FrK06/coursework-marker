"""
KSB Coursework Marker - Streamlit UI with Vision Support

Evaluates student coursework against KSB criteria with Pass/Merit/Referral grading.
Now includes vision analysis for charts, figures, and tables.
"""
import streamlit as st
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processing import DocxProcessor, PDFProcessor, ImageProcessor, ProcessedImage
from src.chunking import SmartChunker
from src.embeddings import Embedder
from src.vector_store import ChromaStore
from src.retrieval import Retriever
from src.llm import OllamaClient
from src.criteria import KSBCriterion, get_module_criteria, get_available_modules
from src.prompts import KSBPromptTemplates, extract_grade_from_evaluation
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL, RetrievalConfig, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="KSB Coursework Marker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: 700; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; text-align: center; }
    .sub-header { font-size: 1.1rem; color: #8b95a5; margin-bottom: 2rem; text-align: center; }
    .status-ok { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 0.6rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .status-error { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); color: white; padding: 0.6rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .stat-card { background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); border-radius: 10px; padding: 1rem; text-align: center; border: 1px solid #2d3748; }
    .stat-number { font-size: 2.5rem; font-weight: 700; }
    .stat-number.merit { color: #3b82f6; }
    .stat-number.pass { color: #10b981; }
    .stat-number.referral { color: #ef4444; }
    .metric-label { font-size: 0.85rem; color: #8b95a5; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'report_data': None,
        'processed_images': None,
        'ksb_criteria': None,
        'feedback_results': None,
        'feedback_generated': False,
        'ollama_connected': False,
        'embedder_loaded': False,
        'selected_module': 'DSP',
        'search_preset': 'BALANCED',
        'use_hybrid': True,
        'semantic_weight': 0.6,
        'keyword_weight': 0.4,
        'similarity_threshold': 0.2,
        'report_top_k': 8,
        'use_vision': True,
        'max_images_per_ksb': 5,
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
        return OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=180)
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


@st.cache_resource
def load_image_processor():
    """Load and cache the image processor."""
    try:
        return ImageProcessor(max_size=(1024, 1024))
    except Exception as e:
        logger.warning(f"Failed to load image processor: {e}")
        return None


def process_report(uploaded_file, image_processor: Optional[ImageProcessor] = None) -> Optional[Dict[str, Any]]:
    """Process uploaded student report including images."""
    if uploaded_file is None:
        return None
    
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        if suffix == '.docx':
            processor = DocxProcessor()
        elif suffix == '.pdf':
            processor = PDFProcessor()
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None
        
        doc = processor.process(tmp_path)
        
        if len(doc.chunks) == 0:
            st.error("Document processor returned no content!")
            return None
        
        chunker = SmartChunker()
        chunks = chunker.chunk_report(doc.chunks, document_id="report")
        
        if len(chunks) == 0:
            st.error("Chunker returned no content!")
            return None
        
        stats = chunker.get_chunking_stats(chunks)
        
        # Process images if available
        processed_images = []
        if image_processor:
            if suffix == '.docx' and hasattr(doc, 'figures') and doc.figures:
                captions = getattr(doc, 'figure_captions', {})
                processed_images = image_processor.process_docx_images(doc.figures, captions)
                logger.info(f"Processed {len(processed_images)} images from DOCX")
            elif suffix == '.pdf':
                processed_images = image_processor.process_pdf_images(tmp_path)
                logger.info(f"Processed {len(processed_images)} images from PDF")
        
        # Get page count - handle both DOCX (estimate) and PDF (accurate)
        total_pages = getattr(doc, 'total_pages', None) or getattr(doc, 'total_pages_estimate', 1)
        pages_are_accurate = getattr(doc, 'pages_are_accurate', suffix == '.pdf')
        source_type = getattr(doc, 'source_type', suffix.replace('.', ''))
        
        return {
            'chunks': chunks,
            'title': doc.title or uploaded_file.name,
            'filename': uploaded_file.name,
            'total_pages': total_pages,
            'pages_are_accurate': pages_are_accurate,  # NEW: track reliability
            'source_type': source_type,  # NEW: 'docx' or 'pdf'
            'figures': getattr(doc, 'figures', {}),
            'figure_captions': getattr(doc, 'figure_captions', {}),
            'processed_images': processed_images,
            'raw_text': doc.raw_text,
            'chunking_stats': stats,
            'tmp_path': tmp_path
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
        
        stats = vector_store.get_stats()
        if stats.get('report_count', 0) == 0:
            st.error("Indexing failed - no documents stored!")
            return False
        
        progress.progress(100, text="Indexing complete!")
        time.sleep(0.3)
        progress.empty()
        return True
        
    except Exception as e:
        logger.exception("Error indexing report")
        st.error(f"Error indexing: {str(e)}")
        return False


def format_image_context(images: List[ProcessedImage]) -> str:
    """Format image metadata as context for the LLM prompt."""
    if not images:
        return ""
    
    lines = [
        "\n## FIGURES/IMAGES IN DOCUMENT",
        f"The document contains {len(images)} figure(s)/image(s) provided for your analysis:",
        ""
    ]
    
    for i, img in enumerate(images, 1):
        desc = f"- **Figure {i}** ({img.image_id}): {img.width}x{img.height} {img.format}"
        if img.caption:
            desc += f'\n  Caption: "{img.caption}"'
        if img.page:
            desc += f" (Page {img.page})"
        lines.append(desc)
    
    lines.append("")
    lines.append("**IMPORTANT:** Examine these images carefully for charts, diagrams, tables, architecture diagrams, results plots, or other visual evidence relevant to the criterion.")
    lines.append("")
    
    return "\n".join(lines)


def evaluate_ksb(
    ksb: KSBCriterion,
    embedder: Embedder,
    vector_store: ChromaStore,
    llm: OllamaClient,
    search_settings: Dict[str, Any],
    images: Optional[List[ProcessedImage]] = None,
    use_vision: bool = True,
    pages_are_accurate: bool = False  # NEW: for PDF=True, DOCX=False
) -> Dict[str, Any]:
    """Evaluate student work against a single KSB with optional vision."""
    
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
    evidence_text = retriever.format_context_for_llm(result, pages_are_accurate=pages_are_accurate)
    
    # Prepare images for vision model
    image_base64_list = []
    image_context = ""
    
    if use_vision and images:
        selected_images = images[:search_settings.get('max_images_per_ksb', 5)]
        
        if selected_images:
            image_base64_list = [img.base64_data for img in selected_images]
            image_context = format_image_context(selected_images)
            logger.info(f"Including {len(selected_images)} images for {ksb.code} evaluation")
    
    # Format prompt with image context
    prompt = KSBPromptTemplates.format_ksb_evaluation(
        ksb_code=ksb.code,
        ksb_title=ksb.title,
        pass_criteria=ksb.pass_criteria,
        merit_criteria=ksb.merit_criteria,
        referral_criteria=ksb.referral_criteria,
        evidence_text=evidence_text,
        image_context=image_context
    )
    
    system_prompt = KSBPromptTemplates.get_system_prompt()
    
    # Generate evaluation WITH images if available
    if image_base64_list:
        logger.info(f"Calling LLM with {len(image_base64_list)} images for vision analysis")
        evaluation = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=LLMConfig.EVALUATION_TEMPERATURE,
            max_tokens=1500,
            images=image_base64_list
        )
    else:
        evaluation = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=LLMConfig.EVALUATION_TEMPERATURE,
            max_tokens=1500
        )
    
    grade_info = extract_grade_from_evaluation(evaluation)
    
    return {
        'ksb_code': ksb.code,
        'ksb_title': ksb.title,
        'ksb_category': ksb.category,
        'pass_criteria': ksb.pass_criteria,
        'merit_criteria': ksb.merit_criteria,
        'referral_criteria': ksb.referral_criteria,
        'evaluation': evaluation,
        'grade': grade_info['grade'],
        'evidence_count': len(result.retrieved_chunks),
        'images_analyzed': len(image_base64_list),
        'search_strategy': result.search_strategy,
        'query_variations': len(result.query_variations)
    }


def generate_overall_summary(ksb_evaluations: List[Dict[str, Any]], llm: OllamaClient) -> str:
    """Generate overall summary from KSB evaluations."""
    evals_text = ""
    for eval_data in ksb_evaluations:
        evals_text += f"\n\n{'='*60}\n"
        evals_text += f"## {eval_data['ksb_code']} - {eval_data['ksb_title']}\n"
        evals_text += f"**Grade: {eval_data['grade']}**\n\n"
        evals_text += eval_data['evaluation']
    
    prompt = KSBPromptTemplates.format_overall_summary(evals_text)
    system_prompt = KSBPromptTemplates.get_system_prompt()
    
    return llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=LLMConfig.SUMMARY_TEMPERATURE,
        max_tokens=2000
    )


def generate_feedback(
    ksb_criteria: List[KSBCriterion],
    embedder: Embedder,
    vector_store: ChromaStore,
    llm: OllamaClient,
    search_settings: Dict[str, Any],
    images: Optional[List[ProcessedImage]] = None,
    use_vision: bool = True,
    pages_are_accurate: bool = False  # NEW: for PDF=True, DOCX=False
) -> Optional[Dict[str, Any]]:
    """Generate feedback for all KSBs with optional vision."""
    stats = vector_store.get_stats()
    if stats.get('report_count', 0) == 0:
        st.error("Vector store is empty!")
        return None
    
    ksb_evaluations = []
    total_ksbs = len(ksb_criteria)
    
    progress_bar = st.progress(0, text="Evaluating KSBs...")
    status_text = st.empty()
    
    for i, ksb in enumerate(ksb_criteria):
        vision_status = f" (with {len(images)} images)" if use_vision and images else ""
        status_text.markdown(f"**Evaluating {ksb.code}:** {ksb.title[:50]}...{vision_status}")
        
        try:
            eval_result = evaluate_ksb(
                ksb, embedder, vector_store, llm, search_settings,
                images=images, use_vision=use_vision,
                pages_are_accurate=pages_are_accurate  # NEW: pass to evaluate_ksb
            )
            ksb_evaluations.append(eval_result)
        except Exception as e:
            logger.exception(f"Error evaluating {ksb.code}")
            ksb_evaluations.append({
                'ksb_code': ksb.code,
                'ksb_title': ksb.title,
                'ksb_category': ksb.category,
                'pass_criteria': ksb.pass_criteria,
                'merit_criteria': ksb.merit_criteria,
                'referral_criteria': ksb.referral_criteria,
                'evaluation': f"Error: {str(e)}",
                'grade': 'ERROR',
                'evidence_count': 0,
                'images_analyzed': 0,
                'search_strategy': 'N/A',
                'query_variations': 0
            })
        
        progress_bar.progress((i + 1) / total_ksbs, text=f"Evaluated {i + 1}/{total_ksbs} KSBs")
    
    status_text.markdown("**Generating overall summary...**")
    overall_summary = generate_overall_summary(ksb_evaluations, llm)
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'ksb_evaluations': ksb_evaluations,
        'overall_summary': overall_summary,
        'search_settings': search_settings,
        'vision_enabled': use_vision and images is not None and len(images) > 0,
        'total_images': len(images) if images else 0
    }


def display_ksb_summary_table(evaluations: List[Dict[str, Any]]):
    """Display summary table."""
    total = len(evaluations)
    merits = sum(1 for e in evaluations if e['grade'] == 'MERIT')
    passes = sum(1 for e in evaluations if e['grade'] == 'PASS')
    referrals = sum(1 for e in evaluations if e['grade'] == 'REFERRAL')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#667eea">{total}</div><div class="metric-label">Total KSBs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number merit">{merits}</div><div class="metric-label">Merit</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number pass">{passes}</div><div class="metric-label">Pass</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number referral">{referrals}</div><div class="metric-label">Referral</div></div>', unsafe_allow_html=True)


def display_feedback(results: Dict[str, Any]):
    """Display the generated feedback."""
    display_ksb_summary_table(results.get('ksb_evaluations', []))
    
    if 'search_settings' in results:
        settings = results['search_settings']
        vision_info = ""
        if results.get('vision_enabled'):
            vision_info = f" | 🖼️ Vision: {results.get('total_images', 0)} images"
        st.info(f"Search: {'Hybrid' if settings['use_hybrid'] else 'Semantic'} | "
                f"Weights: S:{int(settings['semantic_weight']*100)}%/K:{int(settings['keyword_weight']*100)}% | "
                f"Threshold: {settings['similarity_threshold']} | Top-K: {settings['report_top_k']}"
                f"{vision_info}")
    
    st.markdown("## 📝 Overall Assessment")
    st.markdown(results.get('overall_summary', 'No summary available.'))
    
    st.markdown("## 📋 Detailed KSB Evaluations")
    
    for eval_data in results.get('ksb_evaluations', []):
        grade = eval_data.get('grade', 'N/A')
        grade_icon = "🟢" if grade == 'MERIT' else "🟡" if grade == 'PASS' else "🔴"
        images_info = f", 🖼️{eval_data.get('images_analyzed', 0)}" if eval_data.get('images_analyzed', 0) > 0 else ""
        
        with st.expander(f"{grade_icon} **{eval_data['ksb_code']}** - {eval_data['ksb_title'][:50]}... [{grade}] ({eval_data.get('evidence_count', 0)} chunks{images_info})"):
            st.markdown(f"**Search:** {eval_data.get('search_strategy', 'N/A')} | "
                       f"**Evidence chunks:** {eval_data.get('evidence_count', 0)} | "
                       f"**Images analyzed:** {eval_data.get('images_analyzed', 0)}")
            
            tabs = st.tabs(["Pass Criteria", "Merit Criteria", "Referral Criteria"])
            with tabs[0]:
                st.info(eval_data.get('pass_criteria', 'N/A'))
            with tabs[1]:
                st.success(eval_data.get('merit_criteria', 'N/A'))
            with tabs[2]:
                st.error(eval_data.get('referral_criteria', 'N/A'))
            
            st.markdown("---")
            st.markdown("**Evaluation:**")
            st.markdown(eval_data.get('evaluation', 'No evaluation.'))


def render_search_settings():
    """Render search settings panel."""
    st.markdown("## 🔍 Search Settings")
    
    presets = ['BALANCED', 'SEMANTIC_HEAVY', 'KEYWORD_HEAVY', 'SEMANTIC_ONLY', 'HIGH_RECALL', 'CUSTOM']
    selected = st.selectbox("Preset", presets, index=presets.index(st.session_state.get('search_preset', 'BALANCED')))
    
    if selected != st.session_state.search_preset:
        st.session_state.search_preset = selected
        if selected != 'CUSTOM':
            apply_search_preset(selected)
            st.rerun()
    
    with st.expander("⚙️ Advanced", expanded=(selected == 'CUSTOM')):
        use_hybrid = st.toggle("Hybrid Search", value=st.session_state.use_hybrid)
        if use_hybrid != st.session_state.use_hybrid:
            st.session_state.use_hybrid = use_hybrid
            st.session_state.search_preset = 'CUSTOM'
        
        if use_hybrid:
            sem_w = st.slider("Semantic Weight", 0.0, 1.0, st.session_state.semantic_weight, 0.1)
            if sem_w != st.session_state.semantic_weight:
                st.session_state.semantic_weight = sem_w
                st.session_state.keyword_weight = 1.0 - sem_w
                st.session_state.search_preset = 'CUSTOM'
        
        threshold = st.slider("Similarity Threshold", 0.0, 0.5, st.session_state.similarity_threshold, 0.05)
        if threshold != st.session_state.similarity_threshold:
            st.session_state.similarity_threshold = threshold
            st.session_state.search_preset = 'CUSTOM'
        
        top_k = st.slider("Results per KSB", 3, 15, st.session_state.report_top_k, 1)
        if top_k != st.session_state.report_top_k:
            st.session_state.report_top_k = top_k
            st.session_state.search_preset = 'CUSTOM'
    
    st.caption(f"Config: {'Hybrid' if st.session_state.use_hybrid else 'Semantic'} | "
               f"S:{int(st.session_state.semantic_weight*100)}%/K:{int(st.session_state.keyword_weight*100)}% | "
               f"Thresh:{st.session_state.similarity_threshold} | K:{st.session_state.report_top_k}")


def render_vision_settings():
    """Render vision settings panel."""
    st.markdown("## 🖼️ Vision Settings")
    
    use_vision = st.toggle("Enable Vision Analysis", value=st.session_state.use_vision,
                           help="Analyze images, charts, and figures in the document")
    if use_vision != st.session_state.use_vision:
        st.session_state.use_vision = use_vision
    
    if use_vision:
        max_images = st.slider("Max images per KSB", 1, 10, st.session_state.max_images_per_ksb, 1,
                               help="Maximum number of images to send to vision model per criterion")
        if max_images != st.session_state.max_images_per_ksb:
            st.session_state.max_images_per_ksb = max_images
        
        st.caption("⚠️ Vision analysis increases processing time but allows the model to understand charts, diagrams, and figures.")


def get_current_search_settings() -> Dict[str, Any]:
    """Get current search settings."""
    return {
        'use_hybrid': st.session_state.use_hybrid,
        'semantic_weight': st.session_state.semantic_weight,
        'keyword_weight': st.session_state.keyword_weight,
        'similarity_threshold': st.session_state.similarity_threshold,
        'report_top_k': st.session_state.report_top_k,
        'max_images_per_ksb': st.session_state.max_images_per_ksb
    }


def main():
    """Main application."""
    init_session_state()
    
    st.markdown('<p class="main-header">📝 KSB Coursework Marker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered KSB assessment with vision support for charts & figures</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚡ Status")
        
        llm = load_ollama_client()
        embedder = load_embedder()
        image_processor = load_image_processor()
        
        if llm:
            st.markdown('<div class="status-ok">✓ Ollama Connected</div>', unsafe_allow_html=True)
            st.session_state.ollama_connected = True
        else:
            st.markdown('<div class="status-error">✗ Ollama Disconnected</div>', unsafe_allow_html=True)
        
        if embedder:
            st.markdown(f'<div class="status-ok">✓ Embedder Ready ({embedder.embedding_dim}d)</div>', unsafe_allow_html=True)
            st.session_state.embedder_loaded = True
        else:
            st.markdown('<div class="status-error">✗ Embedder Error</div>', unsafe_allow_html=True)
        
        if image_processor:
            st.markdown('<div class="status-ok">✓ Vision Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">✗ Vision Unavailable</div>', unsafe_allow_html=True)
        
        st.markdown("## 📚 Module")
        
        modules = get_available_modules()
        module_options = {code: info['name'] for code, info in modules.items()}
        selected_module = st.selectbox(
            "Module", list(module_options.keys()),
            format_func=lambda x: module_options[x],
            index=list(module_options.keys()).index(st.session_state.selected_module)
        )
        
        if selected_module != st.session_state.selected_module:
            st.session_state.selected_module = selected_module
            st.session_state.ksb_criteria = None
            st.session_state.feedback_results = None
            st.session_state.feedback_generated = False
            st.rerun()
        
        if st.session_state.ksb_criteria is None:
            st.session_state.ksb_criteria = get_module_criteria(selected_module)
        
        st.caption(f"{len(st.session_state.ksb_criteria)} KSBs to assess")
        
        render_search_settings()
        render_vision_settings()
        
        if st.button("🔄 Reset All"):
            for key in ['report_data', 'processed_images', 'feedback_results', 'feedback_generated', 'ksb_criteria']:
                st.session_state[key] = None if key != 'feedback_generated' else False
            st.rerun()
    
    st.markdown("## 📄 Student Report")
    
    uploaded_file = st.file_uploader("Upload report", type=['docx', 'pdf'])
    
    if uploaded_file:
        if st.session_state.report_data is None or st.session_state.report_data.get('filename') != uploaded_file.name:
            with st.spinner("Processing report (including images)..."):
                report_data = process_report(uploaded_file, image_processor)
                if report_data:
                    st.session_state.report_data = report_data
                    st.session_state.processed_images = report_data.get('processed_images', [])
                    st.session_state.feedback_generated = False
                    st.session_state.feedback_results = None
        
        if st.session_state.report_data:
            report = st.session_state.report_data
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Show if pages are accurate or estimated
                if report.get('pages_are_accurate', False):
                    st.metric("Pages", report['total_pages'])
                else:
                    st.metric("Pages (est.)", f"~{report['total_pages']}")
            with col2:
                st.metric("Chunks", len(report['chunks']))
            with col3:
                st.metric("Chars", len(report['raw_text']))
            with col4:
                num_images = len(report.get('processed_images', []))
                st.metric("🖼️ Images", num_images)
            
            # Show warning for DOCX page estimates
            if not report.get('pages_are_accurate', False):
                st.caption("⚠️ Page numbers are estimates for DOCX files. Sections are more reliable for citations.")
            
            if st.session_state.processed_images:
                with st.expander(f"📷 Preview extracted images ({len(st.session_state.processed_images)} found)"):
                    for img in st.session_state.processed_images[:5]:
                        st.caption(f"**{img.image_id}** - {img.width}x{img.height} {img.format}")
                        if img.caption:
                            st.caption(f"Caption: {img.caption}")
    
    can_generate = (
        st.session_state.report_data is not None and
        st.session_state.ksb_criteria is not None and
        st.session_state.ollama_connected and
        st.session_state.embedder_loaded
    )
    
    vision_status = ""
    if st.session_state.use_vision and st.session_state.processed_images:
        vision_status = f" (with {len(st.session_state.processed_images)} images)"
    
    if st.button(f"🚀 Generate KSB Assessment{vision_status}", type="primary", disabled=not can_generate, use_container_width=True):
        search_settings = get_current_search_settings()
        
        tmpdir = tempfile.mkdtemp()
        vector_store = ChromaStore(persist_directory=tmpdir)
        
        st.markdown("### 📥 Indexing Report")
        success = index_report(st.session_state.report_data, embedder, vector_store)
        
        if success:
            stats = vector_store.get_stats()
            
            if stats.get('report_count', 0) > 0:
                st.markdown("### 🤖 Evaluating Against KSBs")
                
                if st.session_state.use_vision and st.session_state.processed_images:
                    st.success(f"🖼️ Vision enabled: {len(st.session_state.processed_images)} images will be analyzed")
                
                st.info(f"Search: {st.session_state.search_preset} | "
                       f"{'Hybrid' if search_settings['use_hybrid'] else 'Semantic'} | "
                       f"Threshold: {search_settings['similarity_threshold']}")
                
                results = generate_feedback(
                    st.session_state.ksb_criteria,
                    embedder,
                    vector_store,
                    llm,
                    search_settings,
                    images=st.session_state.processed_images if st.session_state.use_vision else None,
                    use_vision=st.session_state.use_vision,
                    pages_are_accurate=st.session_state.report_data.get('pages_are_accurate', False)
                )
                
                if results:
                    st.session_state.feedback_results = results
                    st.session_state.feedback_generated = True
                    st.rerun()
            else:
                st.error("Indexing failed - vector store is empty!")
    
    if st.session_state.get('feedback_generated') and st.session_state.get('feedback_results'):
        st.divider()
        display_feedback(st.session_state.feedback_results)
        
        st.divider()
        st.markdown("### 💾 Export")
        
        export_text = "# KSB Assessment Report\n\n"
        export_text += f"**Module:** {st.session_state.selected_module}\n"
        export_text += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
        if st.session_state.feedback_results.get('vision_enabled'):
            export_text += f"**Vision Analysis:** Enabled ({st.session_state.feedback_results.get('total_images', 0)} images)\n"
        export_text += "\n"
        
        export_text += "## Summary\n\n| KSB | Grade | Evidence | Images |\n|-----|-------|----------|--------|\n"
        for e in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"| {e['ksb_code']} | {e['grade']} | {e.get('evidence_count', 0)} | {e.get('images_analyzed', 0)} |\n"
        
        export_text += f"\n\n## Overall\n\n{st.session_state.feedback_results.get('overall_summary', '')}\n\n"
        
        for e in st.session_state.feedback_results.get('ksb_evaluations', []):
            export_text += f"\n---\n\n## {e['ksb_code']} - {e['ksb_title']}\n\n**Grade: {e['grade']}**\n\n{e.get('evaluation', '')}\n"
        
        st.download_button(
            "📥 Download (Markdown)", export_text,
            f"assessment_{st.session_state.selected_module.lower()}_{time.strftime('%Y%m%d')}.md",
            "text/markdown"
        )


if __name__ == "__main__":
    main()
