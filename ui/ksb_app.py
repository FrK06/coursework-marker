"""
KSB Coursework Marker - Agentic Version

Three-agent architecture:
1. Analysis Agent - Multimodal document analysis
2. Scoring Agent - Rubric application and weighted scoring
3. Feedback Agent - Personalized feedback generation
"""
import streamlit as st
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processing import DocxProcessor, PDFProcessor, ImageProcessor, ProcessedImage
from src.chunking import SmartChunker
from src.embeddings import Embedder
from src.vector_store import ChromaStore
from src.llm import OllamaClient
from src.criteria import KSBCriterion, get_module_criteria, get_available_modules
from src.agents import create_agent_system, AgentOrchestrator


# Try to import brief module (optional)
try:
    from src.brief import get_default_brief
    HAS_BRIEF = True
except ImportError:
    HAS_BRIEF = False
    def get_default_brief(module): return None

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="KSB Marker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: 700; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #8b95a5; margin-bottom: 2rem; text-align: center; }
    .agent-card { background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); border-radius: 12px; padding: 1rem; border: 1px solid #3d4654; margin: 0.3rem 0; }
    .agent-card.active { border-color: #667eea; box-shadow: 0 0 15px rgba(102, 126, 234, 0.3); }
    .agent-card.complete { border-color: #10b981; }
    .agent-icon { font-size: 1.8rem; }
    .agent-name { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
    .agent-status { font-size: 0.8rem; color: #8b95a5; }
    .tool-tag { background: #2d3748; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.65rem; margin: 0.1rem; display: inline-block; color: #a0aec0; }
    .grade-badge { padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
    .grade-merit { background: #3b82f6; color: white; }
    .grade-pass { background: #10b981; color: white; }
    .grade-referral { background: #ef4444; color: white; }
    .stat-card { background: linear-gradient(135deg, #1e2530 0%, #252d3a 100%); border-radius: 10px; padding: 1rem; text-align: center; border: 1px solid #2d3748; }
    .stat-number { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        'report_data': None,
        'ksb_criteria': None,
        'assignment_brief': None,
        'agent_results': None,
        'assessment_complete': False,
        'current_phase': None,
        'selected_module': 'DSP',
        'verbose_mode': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_ollama_client():
    try:
        return OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=180)
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return None


@st.cache_resource
def load_embedder():
    try:
        return Embedder(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        return None


@st.cache_resource
def load_image_processor():
    try:
        return ImageProcessor(max_size=(1024, 1024))
    except Exception as e:
        return None


def process_report(uploaded_file, image_processor=None) -> Optional[Dict[str, Any]]:
    if uploaded_file is None:
        return None
    
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        processor = DocxProcessor() if suffix == '.docx' else PDFProcessor()
        doc = processor.process(tmp_path)
        
        chunker = SmartChunker()
        chunks = chunker.chunk_report(doc.chunks, document_id="report")
        
        images = []
        if image_processor:
            if suffix == '.docx' and hasattr(doc, 'figures') and doc.figures:
                images = image_processor.process_docx_images(doc.figures, getattr(doc, 'figure_captions', {}))
            elif suffix == '.pdf':
                images = image_processor.process_pdf_images(tmp_path)
        
        return {
            'chunks': [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks],
            'title': doc.title or uploaded_file.name,
            'filename': uploaded_file.name,
            'total_pages': getattr(doc, 'total_pages', 1) or getattr(doc, 'total_pages_estimate', 1),
            'pages_are_accurate': getattr(doc, 'pages_are_accurate', suffix == '.pdf'),
            'raw_text': doc.raw_text,
            'images': images,
            'tmp_path': tmp_path
        }
    except Exception as e:
        logger.exception("Error processing report")
        st.error(f"Error: {str(e)}")
        return None


def display_agent_status(phase: str):
    """Display agent status cards."""
    col1, col2, col3 = st.columns(3)
    
    analysis_status = 'complete' if phase in ['scoring', 'feedback', 'complete'] else 'active' if phase == 'analysis' else 'waiting'
    scoring_status = 'complete' if phase in ['feedback', 'complete'] else 'active' if phase == 'scoring' else 'waiting'
    feedback_status = 'complete' if phase == 'complete' else 'active' if phase == 'feedback' else 'waiting'
    
    def status_class(s):
        return 'active' if s == 'active' else 'complete' if s == 'complete' else ''
    
    def status_text(s):
        return '⚡ Processing...' if s == 'active' else '✓ Complete' if s == 'complete' else '⏳ Waiting'
    
    with col1:
        st.markdown(f"""
        <div class="agent-card {status_class(analysis_status)}">
            <span class="agent-icon">🔍</span>
            <span class="agent-name">Analysis</span>
            <div class="agent-status">{status_text(analysis_status)}</div>
            <div><span class="tool-tag">text</span><span class="tool-tag">chart</span><span class="tool-tag">image</span><span class="tool-tag">evidence</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="agent-card {status_class(scoring_status)}">
            <span class="agent-icon">📊</span>
            <span class="agent-name">Scoring</span>
            <div class="agent-status">{status_text(scoring_status)}</div>
            <div><span class="tool-tag">rubric</span><span class="tool-tag">weights</span><span class="tool-tag">grade</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="agent-card {status_class(feedback_status)}">
            <span class="agent-icon">💬</span>
            <span class="agent-name">Feedback</span>
            <div class="agent-status">{status_text(feedback_status)}</div>
            <div><span class="tool-tag">strengths</span><span class="tool-tag">gaps</span><span class="tool-tag">format</span></div>
        </div>
        """, unsafe_allow_html=True)


def run_agent_pipeline(report_data, ksb_criteria, assignment_brief, llm, embedder, progress_callback, verbose_log=None):
    """Run the three-agent pipeline."""
    
    # Import here to set verbose callback
    from src.agents.core import BaseAgent
    
    # Set up verbose callback if log container provided
    if verbose_log is not None:
        def verbose_callback(agent, message, data=None):
            verbose_log.append(f"[{agent.upper()}] {message}")
        BaseAgent.verbose_callback = verbose_callback
    else:
        BaseAgent.verbose_callback = None
    
    # Create vector store and index
    tmpdir = tempfile.mkdtemp()
    vector_store = ChromaStore(persist_directory=tmpdir)
    
    progress_callback("Indexing report...", 0.05, "analysis")
    report_texts = [c.get('content', '') for c in report_data['chunks']]
    report_embeddings = embedder.embed_documents(report_texts)
    vector_store.add_report(report_data['chunks'], report_embeddings)
    
    if verbose_log is not None:
        verbose_log.append(f"[INDEX] Indexed {len(report_data['chunks'])} chunks")
    
    # Create agent system
    orchestrator = create_agent_system(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        verbose=st.session_state.verbose_mode
    )
    
    # Convert KSB criteria
    ksb_list = []
    for ksb in ksb_criteria:
        if hasattr(ksb, 'code'):
            ksb_list.append({
                'code': ksb.code,
                'title': ksb.title,
                'pass_criteria': ksb.pass_criteria,
                'merit_criteria': ksb.merit_criteria,
                'referral_criteria': ksb.referral_criteria,
                'category': ksb.category
            })
        else:
            ksb_list.append(ksb)
    
    # Convert brief
    brief_dict = None
    if assignment_brief:
        brief_dict = assignment_brief.to_dict() if hasattr(assignment_brief, 'to_dict') else assignment_brief
    
    # Convert images
    images = []
    for img in report_data.get('images', []):
        if hasattr(img, 'image_id'):
            images.append({
                'image_id': img.image_id,
                'caption': getattr(img, 'caption', ''),
                'base64': getattr(img, 'base64_data', '')
            })
        elif isinstance(img, dict):
            images.append(img)
    
    # Run pipeline with progress callbacks
    def agent_progress(msg, pct):
        if "Analysis" in msg:
            progress_callback(msg, pct, "analysis")
        elif "Scoring" in msg:
            progress_callback(msg, pct, "scoring")
        elif "Feedback" in msg:
            progress_callback(msg, pct, "feedback")
        else:
            progress_callback(msg, pct, "complete")
    
    results = orchestrator.process(
        chunks=report_data['chunks'],
        ksb_criteria=ksb_list,
        assignment_brief=brief_dict,
        images=images,
        progress_callback=agent_progress
    )
    
    return results


def display_results(results: Dict[str, Any]):
    """Display assessment results."""
    
    summary = results.get("overall_summary", {})
    
    st.markdown("## 📊 Assessment Results")
    
    # Grade cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#fff">{summary.get("total_ksbs", 0)}</div><div>Total KSBs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#10b981">{summary.get("merit_count", 0)}</div><div>Merit</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#3b82f6">{summary.get("pass_count", 0)}</div><div>Pass</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#ef4444">{summary.get("referral_count", 0)}</div><div>Referral</div></div>', unsafe_allow_html=True)
    
    # Overall recommendation
    overall = summary.get("overall_recommendation", "UNKNOWN")
    grade_class = f"grade-{overall.lower()}"
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="grade-badge {grade_class}">{overall}</span>
        <span style="color: #8b95a5; margin-left: 0.5rem;">Overall Recommendation</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall feedback
    if results.get("overall_feedback"):
        with st.expander("📝 Overall Feedback", expanded=True):
            st.markdown(results["overall_feedback"])
    
    # Key strengths and improvements
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💪 Key Strengths")
        for s in summary.get("key_strengths", [])[:5]:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("### 📈 Priority Improvements")
        for i in summary.get("priority_improvements", [])[:5]:
            st.markdown(f"- {i}")
    
    # KSB Details
    st.markdown("## 📋 KSB Breakdown")
    
    for sr in results.get("scoring_results", []):
        ksb_code = sr.get("ksb_code", "")
        grade = sr.get("grade", "UNKNOWN")
        confidence = sr.get("confidence", "")
        ksb_title = sr.get("ksb_title", "")
        
        icon = "🟢" if grade == "MERIT" else "🟡" if grade == "PASS" else "🔴"
        
        #with st.expander(f"{icon} **{ksb_code}** - {sr.get('ksb_title', '')[:150]}... [{grade}]"):
        with st.expander(f"{icon} **{ksb_code}** - {ksb_title} [{grade}]"):
            st.markdown(f"**Confidence:** {confidence} | **Weighted Score:** {sr.get('weighted_score', 0):.3f}")
            st.markdown(f"**Rationale:** {sr.get('rationale', '')}")
            
            # Show feedback
            fb = next((f for f in results.get("feedback_results", []) if f.get("ksb_code") == ksb_code), {})
            if fb.get("formatted_feedback"):
                st.markdown(fb["formatted_feedback"])
            
            # Gaps
            gaps = sr.get("gaps_identified", [])
            if gaps:
                st.markdown("**Gaps:**")
                for g in gaps[:5]:
                    st.markdown(f"- {g}")


def main():
    init_session_state()
    
    st.markdown('<p class="main-header">🤖 KSB Coursework Marker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Three-Agent Assessment System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚡ Status")
        
        llm = load_ollama_client()
        embedder = load_embedder()
        image_processor = load_image_processor()
        
        if llm:
            st.success(f"✓ Ollama: {OLLAMA_MODEL}")
        else:
            st.error("✗ Ollama not connected")
        
        if embedder:
            st.success(f"✓ Embedder ({embedder.embedding_dim}d)")
        else:
            st.error("✗ Embedder error")
        
        if image_processor:
            st.success("✓ Vision ready")
        
        st.markdown("---")
        
        # Module selection
        st.markdown("## 📚 Module")
        modules = get_available_modules()
        module_options = {code: info['name'] for code, info in modules.items()}
        selected = st.selectbox("Module", list(module_options.keys()), format_func=lambda x: module_options[x],
                               index=list(module_options.keys()).index(st.session_state.selected_module))
        
        if selected != st.session_state.selected_module:
            st.session_state.selected_module = selected
            st.session_state.ksb_criteria = None
            st.session_state.assignment_brief = None
            st.rerun()
        
        if st.session_state.ksb_criteria is None:
            st.session_state.ksb_criteria = get_module_criteria(selected)
        
        if HAS_BRIEF and st.session_state.assignment_brief is None:
            st.session_state.assignment_brief = get_default_brief(selected)
        
        st.caption(f"{len(st.session_state.ksb_criteria)} KSBs")
        
        st.markdown("---")
        st.session_state.verbose_mode = st.checkbox("Verbose mode", value=st.session_state.verbose_mode)
        
        if st.button("🔄 Reset"):
            for key in ['report_data', 'agent_results', 'assessment_complete', 'current_phase']:
                st.session_state[key] = None if key != 'assessment_complete' else False
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📄 Upload Report")
        uploaded_file = st.file_uploader("Student report", type=['docx', 'pdf'])
        
        if uploaded_file:
            if st.session_state.report_data is None or st.session_state.report_data.get('filename') != uploaded_file.name:
                with st.spinner("Processing..."):
                    report_data = process_report(uploaded_file, image_processor)
                    if report_data:
                        st.session_state.report_data = report_data
                        st.session_state.assessment_complete = False
            
            if st.session_state.report_data:
                r = st.session_state.report_data
                c1, c2, c3 = st.columns(3)
                c1.metric("Pages", r['total_pages'])
                c2.metric("Chunks", len(r['chunks']))
                c3.metric("Images", len(r.get('images', [])))
    
    with col2:
        st.markdown("## 🤖 Agents")
        display_agent_status(st.session_state.current_phase)
    
    # Run button
    st.markdown("---")
    
    can_run = (st.session_state.report_data and st.session_state.ksb_criteria and llm and embedder)
    
    if st.button("🚀 Run Agentic Assessment", type="primary", disabled=not can_run, use_container_width=True):
        progress_bar = st.progress(0, "Starting...")
        status_placeholder = st.empty()
        agent_placeholder = st.empty()
        
        # Verbose mode log container
        verbose_log = [] if st.session_state.verbose_mode else None
        verbose_container = None
        if st.session_state.verbose_mode:
            verbose_container = st.expander("🔍 Verbose Log (Tool Calls)", expanded=True)
            verbose_text = verbose_container.empty()
        
        def update_progress(msg, pct, phase):
            st.session_state.current_phase = phase
            progress_bar.progress(pct, msg)
            with agent_placeholder:
                display_agent_status(phase)
            # Update verbose log display
            if verbose_log and verbose_container:
                verbose_text.code("\n".join(verbose_log[-30:]), language="text")
        
        try:
            results = run_agent_pipeline(
                st.session_state.report_data,
                st.session_state.ksb_criteria,
                st.session_state.assignment_brief,
                llm, embedder, update_progress,
                verbose_log=verbose_log
            )
            
            st.session_state.agent_results = results
            st.session_state.assessment_complete = True
            st.session_state.current_phase = "complete"
            
            # Final verbose log
            if verbose_log and verbose_container:
                verbose_text.code("\n".join(verbose_log), language="text")
            
            st.rerun()
            
        except Exception as e:
            logger.exception("Pipeline error")
            st.error(f"Error: {str(e)}")
    
    # Display results
    if st.session_state.assessment_complete and st.session_state.agent_results:
        st.markdown("---")
        display_results(st.session_state.agent_results)
        
        # Export
        st.markdown("### 💾 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            md = f"# Assessment Report\n\n{st.session_state.agent_results.get('overall_feedback', '')}\n\n"
            for fb in st.session_state.agent_results.get("feedback_results", []):
                md += fb.get("formatted_feedback", "") + "\n\n"
            st.download_button("📄 Feedback (MD)", md, f"feedback_{st.session_state.selected_module}_{time.strftime('%Y%m%d')}.md", "text/markdown")
        
        with col2:
            st.download_button("📊 Full Report (JSON)", json.dumps(st.session_state.agent_results, indent=2, default=str),
                             f"assessment_{st.session_state.selected_module}_{time.strftime('%Y%m%d')}.json", "application/json")


if __name__ == "__main__":
    main()
