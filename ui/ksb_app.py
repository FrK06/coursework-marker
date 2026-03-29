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
import csv
from io import StringIO
from datetime import datetime

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
    """Process uploaded report with input validation."""
    if uploaded_file is None:
        return None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        processor = DocxProcessor() if suffix == '.docx' else PDFProcessor()
        doc = processor.process(tmp_path)

        # Validate document content length
        raw_text = getattr(doc, 'raw_text', '')
        if len(raw_text.strip()) < 200:
            st.warning("⚠️ Document appears too short to assess (less than 200 characters). Please check the upload.")
            logger.warning(f"Document too short: {len(raw_text)} characters")
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except:
                pass
            return None

        chunker = SmartChunker()
        chunks = chunker.chunk_report(doc.chunks, document_id="report")

        # Validate chunk count
        if len(chunks) < 3:
            st.warning(f"⚠️ Document produced very few text sections ({len(chunks)} chunks). Results may be unreliable.")
            logger.warning(f"Low chunk count: {len(chunks)}")

        images = []
        if image_processor:
            if suffix == '.docx' and hasattr(doc, 'figures') and doc.figures:
                images = image_processor.process_docx_images(doc.figures, getattr(doc, 'figure_captions', {}))
            elif suffix == '.pdf':
                images = image_processor.process_pdf_images(tmp_path)

        # Get page count correctly for both PDF (total_pages) and DOCX (total_pages_estimate)
        total_pages = getattr(doc, 'total_pages', None) or getattr(doc, 'total_pages_estimate', 1)

        return {
            'chunks': [c.to_dict() if hasattr(c, 'to_dict') else c for c in chunks],
            'title': doc.title or uploaded_file.name,
            'filename': uploaded_file.name,
            'total_pages': total_pages,
            'pages_are_accurate': getattr(doc, 'pages_are_accurate', suffix == '.pdf'),
            'raw_text': raw_text,
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
    """Run the three-agent pipeline with proper resource cleanup."""
    import shutil

    # Import here to set verbose callback
    from src.agents.core import BaseAgent

    # Set up verbose callback if log container provided
    if verbose_log is not None:
        def verbose_callback(agent, message, data=None):
            verbose_log.append(f"[{agent.upper()}] {message}")
        BaseAgent.verbose_callback = verbose_callback
    else:
        BaseAgent.verbose_callback = None

    # Create vector store with temp directory (will be cleaned up in finally block)
    tmpdir = tempfile.mkdtemp()
    vector_store = None
    orchestrator = None
    results = None

    try:
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
            verbose=st.session_state.verbose_mode,
            module_code=st.session_state.selected_module
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

    finally:
        # Clean up temporary ChromaDB directory
        if tmpdir and Path(tmpdir).exists():
            try:
                # Release ChromaDB file locks (Windows file locking issue)
                try:
                    del vector_store
                    import gc
                    gc.collect()
                    import time
                    time.sleep(0.5)  # Give Windows time to release file handles
                except:
                    pass

                shutil.rmtree(tmpdir)
                if verbose_log is not None:
                    verbose_log.append(f"[CLEANUP] Removed temp directory: {tmpdir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {tmpdir}: {e}")

        # Clean up uploaded file temp path
        if report_data and 'tmp_path' in report_data:
            tmp_path = report_data['tmp_path']
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                    if verbose_log is not None:
                        verbose_log.append(f"[CLEANUP] Removed temp file: {tmp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {tmp_path}: {e}")


def export_results_to_csv(results: Dict[str, Any], module_code: str = "MLCC") -> str:
    """
    Convert assessment results to CSV format.

    Args:
        results: Assessment results from agent pipeline
        module_code: Module code for filename

    Returns:
        CSV string ready for download
    """
    output = StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "KSB Code",
        "KSB Title",
        "Grade",
        "Confidence",
        "Weighted Score",
        "Pass Criteria Met",
        "Merit Criteria Met",
        "Rationale",
        "Key Strengths",
        "Gaps Identified",
        "Improvement Suggestions"
    ])

    # Get scoring and feedback results
    scoring_results = results.get("scoring_results", [])
    feedback_results = results.get("feedback_results", [])

    # Write data rows
    for sr in scoring_results:
        ksb_code = sr.get("ksb_code", "")

        # Find corresponding feedback
        fb = next((f for f in feedback_results if f.get("ksb_code") == ksb_code), {})

        # Extract strengths (from feedback)
        strengths = fb.get("strengths", [])
        strengths_text = " | ".join(str(s) for s in strengths[:3]) if strengths else ""

        # Extract gaps
        gaps = sr.get("gaps_identified", [])
        gaps_text = " | ".join(str(g) for g in gaps[:5]) if gaps else ""

        # Extract improvements (from feedback)
        improvements = fb.get("improvements", [])
        if improvements:
            # Handle both dict and string formats
            imp_texts = []
            for imp in improvements[:3]:
                if isinstance(imp, dict):
                    imp_texts.append(imp.get("suggestion", str(imp)))
                else:
                    imp_texts.append(str(imp))
            improvements_text = " | ".join(imp_texts)
        else:
            improvements_text = ""

        writer.writerow([
            ksb_code,
            sr.get("ksb_title", ""),
            sr.get("grade", ""),
            sr.get("confidence", ""),
            sr.get("weighted_score", 0),
            sr.get("pass_met", ""),
            sr.get("merit_met", ""),
            sr.get("rationale", ""),
            strengths_text,
            gaps_text,
            improvements_text
        ])

    # Add summary row
    writer.writerow([])  # Empty row
    summary = results.get("overall_summary", {})
    writer.writerow(["OVERALL SUMMARY"])
    writer.writerow(["Total KSBs", summary.get("total_ksbs", 0)])
    writer.writerow(["Merit Count", summary.get("merit_count", 0)])
    writer.writerow(["Pass Count", summary.get("pass_count", 0)])
    writer.writerow(["Referral Count", summary.get("referral_count", 0)])
    writer.writerow(["Overall Recommendation", summary.get("overall_recommendation", "")])
    writer.writerow(["Confidence", summary.get("confidence", "")])

    # Add key strengths
    writer.writerow([])
    writer.writerow(["KEY STRENGTHS"])
    for strength in summary.get("key_strengths", [])[:5]:
        writer.writerow(["", strength])

    # Add priority improvements
    writer.writerow([])
    writer.writerow(["PRIORITY IMPROVEMENTS"])
    for improvement in summary.get("priority_improvements", [])[:5]:
        writer.writerow(["", improvement])

    return output.getvalue()


def display_results(results: Dict[str, Any]):
    """Display assessment results."""
    
    summary = results.get("overall_summary", {})
    
    st.markdown("## 📊 Assessment Results")

    # Content quality warnings
    content_quality = results.get("content_quality", {})
    off_topic_images = content_quality.get("off_topic_images", 0)
    adversarial_tables = content_quality.get("adversarial_tables_detected", 0)

    if adversarial_tables > 0:
        image_note = ""
        if off_topic_images > 0:
            image_note = (
                f" Additionally, **{off_topic_images}** unrelated image(s) were detected."
            )
        st.error(
            "**⚠️ Adversarial Content Detected**\n\n"
            "The KSB reflection table contains content unrelated to the module "
            f"(e.g. off-topic text instead of genuine reflections). "
            f"**{adversarial_tables}** adversarial table(s) found. "
            f"Affected KSBs have been automatically referred.{image_note}"
        )
    elif off_topic_images > 0:
        st.error(
            "**⚠️ Adversarial Content Detected**\n\n"
            f"**{off_topic_images}** image(s) in the report contain content "
            "unrelated to the module (e.g. off-topic diagrams or screenshots). "
            "These images have been excluded from evidence."
        )
    elif content_quality.get("quality_flag") == "CRITICAL":
        st.warning(
            "**⚠️ Content Quality Issue**\n\n"
            f"Report contains **{content_quality.get('off_topic_chunks', 0)}** off-topic section(s). "
            "Manual review recommended."
        )
    elif content_quality.get("quality_flag") == "WARNING":
        st.info(
            "Some sections contain content with low relevance to the module."
        )

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

    # Download Buttons
    st.markdown("---")
    st.markdown("### 💾 Export Results")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Generate CSV
        csv_data = export_results_to_csv(results, st.session_state.get("selected_module", "MLCC"))

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        module = st.session_state.get("selected_module", "MLCC")
        csv_filename = f"ksb_assessment_{module}_{timestamp}.csv"

        # CSV Download button
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            help="Spreadsheet format with grades, feedback, and improvements",
            use_container_width=True
        )

    with col2:
        # Generate JSON
        json_data = json.dumps(results, indent=2, default=str)
        json_filename = f"ksb_assessment_{module}_{timestamp}.json"

        # JSON Download button
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=json_filename,
            mime="application/json",
            help="Complete raw data in JSON format for further processing",
            use_container_width=True
        )

    with col3:
        # Generate Markdown feedback
        md = f"# Assessment Report\n\n{results.get('overall_feedback', '')}\n\n"
        for fb in results.get("feedback_results", []):
            md += fb.get("formatted_feedback", "") + "\n\n"

        st.download_button(
            label="📄 Download Markdown",
            data=md,
            file_name=f"feedback_{module}_{timestamp}.md",
            mime="text/markdown",
            help="Formatted feedback report in Markdown",
            use_container_width=True
        )

    st.markdown("---")

    # Overall feedback (includes key strengths, priority improvements, and next steps)
    if results.get("overall_feedback"):
        with st.expander("📝 Overall Assessment Summary", expanded=True):
            st.markdown(results["overall_feedback"])

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

            # === TRANSPARENCY PANEL: Assessment Details ===
            audit = sr.get('audit_trail', {})
            if audit:  # Only show if audit trail exists
                with st.expander(f"🔍 Assessment Details for {ksb_code}"):
                    tab1, tab2, tab3, tab4 = st.tabs(["📄 Evidence Retrieved", "🤖 LLM Reasoning", "✅ Validation", "📊 Grade Decision"])

                    # Tab 1: Evidence Retrieved
                    with tab1:
                        evidence_info = audit.get('evidence', {})
                        total_chunks = evidence_info.get('total_chunks_retrieved', 0)
                        filtered_chunks = evidence_info.get('chunks_after_filtering', 0)
                        search_strat = evidence_info.get('search_strategy', {})

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Retrieved", total_chunks)
                        col2.metric("After Filtering", filtered_chunks)
                        col3.metric("Boilerplate Filtered", search_strat.get('boilerplate_filtered', 0))

                        st.caption(f"**Search Mode:** {search_strat.get('mode', 'unknown')} | **Query Variations:** {search_strat.get('query_variations', 0)}")

                        st.markdown("---")
                        st.markdown("**Evidence Chunks:**")

                        chunks = evidence_info.get('chunks', [])
                        if chunks:
                            for idx, chunk in enumerate(chunks, 1):
                                with st.container():
                                    # Metadata badge
                                    section = chunk.get('section_id', 'unknown')
                                    relevance = chunk.get('relevance_score', 0.0)
                                    method = chunk.get('search_method', 'unknown')

                                    st.caption(f"**Chunk {idx}** | Section: `{section}` | Relevance: `{relevance:.2f}` | Method: `{method}`")

                                    # Chunk text with theme-adaptive styling
                                    chunk_text = chunk.get('text', '')
                                    if chunk_text:
                                        # Truncate if too long
                                        display_text = chunk_text if len(chunk_text) <= 500 else chunk_text + "..."
                                        # Use theme-compatible semi-transparent background with border accent
                                        st.markdown(
                                            f"<div style='background-color: rgba(255,255,255,0.05); color: inherit; "
                                            f"padding: 10px; border-radius: 5px; border-left: 3px solid #4CAF50; "
                                            f"margin-bottom: 0.5rem; font-family: monospace; font-size: 0.9em;'>"
                                            f"{display_text}</div>",
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.caption("_(No text)_")
                        else:
                            st.info("No evidence chunks available")

                    # Tab 2: LLM Reasoning
                    with tab2:
                        llm_info = audit.get('llm_evaluation', {})

                        col1, col2 = st.columns(2)
                        col1.metric("Evidence Summary Length", f"{llm_info.get('evidence_summary_length', 0)} chars")
                        col2.metric("Model", llm_info.get('model', 'unknown'))

                        st.markdown("---")
                        st.markdown("**Full LLM Response:**")

                        raw_response = llm_info.get('raw_response', '')
                        if raw_response:
                            st.code(raw_response, language="text")
                        else:
                            st.info("No LLM response captured")

                    # Tab 3: Validation
                    with tab3:
                        val_info = audit.get('validation', {})
                        action = val_info.get('action', 'unknown')
                        conf = val_info.get('confidence', 0.0)
                        warnings = val_info.get('warnings', [])
                        retried = val_info.get('retried', False)

                        # Colored badge for validation action
                        action_color = "#10b981" if action == "accept" else "#f59e0b" if action == "flag_for_review" else "#ef4444"
                        st.markdown(f"<span style='background-color: {action_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600;'>{action.upper()}</span>", unsafe_allow_html=True)

                        st.metric("Validation Confidence", f"{conf:.2f}")

                        if retried:
                            st.warning("⚠️ Response was retried due to validation failure")

                        st.markdown("---")

                        if warnings:
                            st.markdown("**Validation Warnings:**")
                            for idx, warning in enumerate(warnings, 1):
                                st.warning(f"{idx}. {warning}")
                        else:
                            st.success("✓ No validation warnings")

                    # Tab 4: Grade Decision
                    with tab4:
                        decision = audit.get('grade_decision', {})

                        final_grade = decision.get('grade', 'UNKNOWN')
                        grade_color = "#10b981" if final_grade == "MERIT" else "#3b82f6" if final_grade == "PASS" else "#ef4444"
                        st.markdown(f"<div style='text-align: center; margin: 1rem 0;'><span style='background-color: {grade_color}; color: white; padding: 0.5rem 1.5rem; border-radius: 20px; font-weight: 700; font-size: 1.2rem;'>{final_grade}</span></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Criteria Met:**")
                            pass_met = decision.get('pass_criteria_met', False)
                            merit_met = decision.get('merit_criteria_met', False)
                            st.checkbox("Pass Criteria", value=pass_met, disabled=True, key=f"{ksb_code}_pass_check")
                            st.checkbox("Merit Criteria", value=merit_met, disabled=True, key=f"{ksb_code}_merit_check")

                        with col2:
                            st.markdown("**Assessment Metrics:**")
                            st.metric("Evidence Strength", decision.get('evidence_strength', 'unknown'))
                            st.metric("Confidence", decision.get('confidence', 'unknown'))

                        st.caption(f"**Extraction Method:** {decision.get('extraction_method', 'unknown')}")


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

        # Reset index button (needed after embedding model upgrade)
        if st.button("🗑️ Reset Index", help="Delete vector index (use after upgrading embedding model)"):
            import shutil
            from config import INDEX_DIR
            index_dirs = [
                INDEX_DIR,
                INDEX_DIR.parent / "indexes_e5-base-v2",
                INDEX_DIR.parent / "indexes"
            ]
            deleted_count = 0
            for idx_dir in index_dirs:
                if idx_dir.exists():
                    try:
                        shutil.rmtree(idx_dir)
                        deleted_count += 1
                        logger.info(f"Deleted index directory: {idx_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {idx_dir}: {e}")
            if deleted_count > 0:
                st.success(f"✓ Deleted {deleted_count} index director{'y' if deleted_count == 1 else 'ies'}")
            else:
                st.info("No index directories found to delete")
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

    # Validate prerequisites
    can_run = (st.session_state.report_data and st.session_state.ksb_criteria and llm and embedder)

    # Additional validation: check chunk count
    if can_run and st.session_state.report_data:
        chunks_count = len(st.session_state.report_data.get('chunks', []))
        if chunks_count < 3:
            can_run = False
            st.error(f"⚠️ Document has too few sections ({chunks_count} chunks). Cannot assess reliably.")

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


if __name__ == "__main__":
    main()
