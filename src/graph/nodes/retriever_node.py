"""
Retriever Node — Hybrid BM25 + semantic retrieval for all KSBs.

For every KSB in the selected module, performs hybrid retrieval and
populates evidence_map. Also detects adversarial/off-topic content.
"""
import re
import logging
from typing import Dict, Any, List

from ...retrieval.retriever import Retriever, QueryExpander, BM25
from ...embeddings.embedder import Embedder
from ...vector_store.chroma_store import ChromaStore
from ...criteria.ksb_parser import MODULE_RELEVANCE_TERMS, ADVERSARIAL_REFLECTION_KSBS
from ...prompts.base_templates import is_boilerplate
from ..state import GraphState, KSBEvidence

from config import RetrievalConfig

logger = logging.getLogger(__name__)


def retriever_node(state: GraphState) -> dict:
    """
    LangGraph node: Hybrid evidence retrieval for all KSBs.

    Reads:
        state["ksb_criteria"], state["report_chunks"], state["module_code"]

    Writes:
        evidence_map, content_quality, phase, errors
    """
    module_code = state["module_code"]
    ksb_criteria = state["ksb_criteria"]
    errors = list(state.get("errors", []))

    logger.info(f"Retriever node: starting retrieval for {len(ksb_criteria)} KSBs ({module_code})")

    # Build retriever from existing components
    try:
        embedder = Embedder()
        vector_store = _get_vector_store(state)
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            report_top_k=RetrievalConfig.REPORT_TOP_K,
            max_context_tokens=RetrievalConfig.MAX_CONTEXT_TOKENS,
            similarity_threshold=RetrievalConfig.SIMILARITY_THRESHOLD,
            use_hybrid=RetrievalConfig.USE_HYBRID_SEARCH,
            semantic_weight=RetrievalConfig.SEMANTIC_WEIGHT,
            keyword_weight=RetrievalConfig.KEYWORD_WEIGHT,
        )
    except Exception as e:
        logger.error(f"Failed to initialise retriever: {e}")
        errors.append(f"Retriever init failed: {e}")
        return {
            "evidence_map": {},
            "content_quality": {"status": "ERROR", "error": str(e)},
            "phase": "retrieval",
            "errors": errors,
        }

    # ── OCR extraction for images ────────────────────────────────
    image_analyses = _extract_image_ocr(state, embedder, vector_store)
    if image_analyses:
        logger.info(f"OCR extracted text from {len(image_analyses)} images")

    evidence_map: Dict[str, KSBEvidence] = {}
    adversarial_ksbs: List[str] = []
    off_topic_count = 0

    # Retrieve evidence for each KSB
    for criterion in ksb_criteria:
        ksb_code = criterion["code"]
        ksb_title = criterion.get("title", "")
        ksb_description = criterion.get("full_description", ksb_title)

        try:
            result = retriever.retrieve_for_criterion(
                criterion_text=ksb_description,
                criterion_id=ksb_code,
            )

            # Filter out boilerplate chunks
            filtered_chunks = [
                chunk for chunk in result.retrieved_chunks
                if not is_boilerplate(chunk)
            ]

            # Extract similarity scores
            similarity_scores = [
                chunk.get("similarity", 0.0) for chunk in filtered_chunks
            ]

            evidence_map[ksb_code] = KSBEvidence(
                ksb_code=ksb_code,
                chunks=filtered_chunks,
                query_variations=result.query_variations,
                search_strategy=result.search_strategy,
                total_retrieved=len(filtered_chunks),
                similarity_scores=similarity_scores,
            )

            # Check for off-topic (all chunks below 0.05 similarity)
            if filtered_chunks and max(similarity_scores) < 0.05:
                off_topic_count += 1
                logger.warning(f"{ksb_code}: all evidence below 0.05 similarity — off-topic")

        except Exception as e:
            logger.error(f"Retrieval failed for {ksb_code}: {e}")
            errors.append(f"Retrieval error for {ksb_code}: {e}")
            evidence_map[ksb_code] = KSBEvidence(
                ksb_code=ksb_code,
                chunks=[],
                query_variations=[],
                search_strategy="error",
                total_retrieved=0,
                similarity_scores=[],
            )

    # Adversarial table detection
    adversarial_ksbs = _detect_adversarial_tables(
        vector_store, module_code, ksb_criteria
    )

    # Build content quality report
    content_quality = _build_content_quality(
        module_code, adversarial_ksbs, off_topic_count
    )

    logger.info(
        f"Retriever node complete: {len(evidence_map)} KSBs, "
        f"{sum(e['total_retrieved'] for e in evidence_map.values())} total chunks, "
        f"{len(adversarial_ksbs)} adversarial, {off_topic_count} off-topic"
    )

    return {
        "evidence_map": evidence_map,
        "content_quality": content_quality,
        "image_analyses": image_analyses,
        "phase": "retrieval",
        "errors": errors,
    }


def _get_vector_store(state: GraphState) -> ChromaStore:
    """
    Build a ChromaStore from the report chunks already in state.

    The chunks should already be indexed in ChromaDB by the ingestion pipeline
    before the graph is invoked. We reconnect to the same ephemeral store.
    """
    # The vector store is reconstructed from the same temp directory
    # that was used during document ingestion. The graph_builder passes
    # the vector_store instance via state or we reconstruct it.
    from config import CHROMA_PERSIST_DIR
    store = ChromaStore(
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return store


def _detect_adversarial_tables(
    vector_store: ChromaStore,
    module_code: str,
    ksb_criteria: list,
) -> List[str]:
    """
    Detect adversarial KSB reflection tables.

    A table is adversarial if it contains KSB codes but < 20% of rows
    contain module-relevant terms.
    """
    adversarial_ksbs = []
    module_terms = MODULE_RELEVANCE_TERMS.get(module_code, [])
    reflection_ksbs = set(ADVERSARIAL_REFLECTION_KSBS.get(module_code, []))

    if not module_terms or not reflection_ksbs:
        return adversarial_ksbs

    try:
        all_chunks = vector_store.get_all_report_chunks()
    except Exception:
        return adversarial_ksbs

    for chunk in all_chunks:
        content = chunk.get('content', '')
        content_lower = content.lower()

        # Look for table-like content with KSB codes
        ksb_codes_found = set(re.findall(r'\b[KSB]\d{1,2}\b', content))
        pipe_count = content.count('|')

        if len(ksb_codes_found) >= 2 and pipe_count >= 3:
            # Count rows with module-relevant terms
            rows = content.split('\n')
            rows_with_terms = 0
            total_rows = 0

            for row in rows:
                if '|' in row and row.strip() and not row.strip().startswith('|--'):
                    total_rows += 1
                    row_lower = row.lower()
                    if any(term.lower() in row_lower for term in module_terms):
                        rows_with_terms += 1

            if total_rows > 0 and (rows_with_terms / total_rows) < 0.20:
                # Adversarial table detected
                affected = ksb_codes_found & reflection_ksbs
                adversarial_ksbs.extend(affected)
                logger.warning(
                    f"Adversarial table detected: {rows_with_terms}/{total_rows} "
                    f"rows have module terms, affecting KSBs: {affected}"
                )

    return list(set(adversarial_ksbs))


def _build_content_quality(
    module_code: str,
    adversarial_ksbs: List[str],
    off_topic_count: int,
) -> Dict[str, Any]:
    """Build the content quality report."""
    # Determine overall status
    if adversarial_ksbs and off_topic_count > 0:
        status = "CRITICAL"
    elif adversarial_ksbs:
        status = "WARNING_ADVERSARIAL"
    elif off_topic_count > 2:
        status = "CRITICAL"
    elif off_topic_count > 0:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "status": status,
        "adversarial_ksbs": adversarial_ksbs,
        "off_topic_count": off_topic_count,
        "module_code": module_code,
    }


def _extract_image_ocr(
    state: GraphState,
    embedder: Embedder,
    vector_store: ChromaStore,
) -> List[Dict[str, Any]]:
    """
    Run OCR on report images and index extracted text into the vector store.

    Returns list of {image_id, caption, ocr_text} for each successfully processed image.
    """
    report_images = state.get("report_images", [])
    if not report_images:
        return []

    from config import OCR_ENABLED, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
    if not OCR_ENABLED:
        logger.debug("OCR disabled in config — skipping image analysis")
        return []

    from ...document_processing.image_processor import ImageProcessor, ProcessedImage
    from ...llm.ollama_client import OllamaClient

    try:
        ocr_client = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model="glm-ocr",
            timeout=OLLAMA_TIMEOUT,
        )
        img_processor = ImageProcessor(ollama_client=ocr_client)
    except Exception as e:
        logger.warning(f"Could not initialise OCR pipeline: {e}")
        return []

    image_analyses = []
    ocr_chunks = []

    for img in report_images[:10]:  # Cap at 10 images like old pipeline
        base64_data = img.get("base64") or img.get("base64_data", "")
        if not base64_data:
            continue

        proc_img = ProcessedImage(
            image_id=img.get("image_id", "unknown"),
            base64_data=base64_data,
            format="png",
            width=0,
            height=0,
            caption=img.get("caption", ""),
        )

        try:
            ocr_text = img_processor.extract_text_with_ocr(proc_img)
        except Exception as e:
            logger.warning(f"OCR failed for {proc_img.image_id}: {e}")
            continue

        if ocr_text and len(ocr_text) > 10:
            image_analyses.append({
                "image_id": proc_img.image_id,
                "caption": proc_img.caption,
                "ocr_text": ocr_text,
            })
            ocr_chunks.append({
                "content": f"[Image: {proc_img.image_id}] {ocr_text}",
                "metadata": {
                    "source_image": proc_img.image_id,
                    "caption": proc_img.caption,
                    "chunk_type": "ocr",
                },
            })
            logger.info(f"  OCR: {proc_img.image_id} -> {len(ocr_text)} chars")

    # Embed and index OCR chunks so they're retrievable per-KSB
    if ocr_chunks:
        try:
            ocr_texts = [c["content"] for c in ocr_chunks]
            ocr_embeddings = embedder.embed_documents(ocr_texts)
            vector_store.add_report(ocr_chunks, ocr_embeddings)
            logger.info(f"Indexed {len(ocr_chunks)} OCR chunks into vector store")
        except Exception as e:
            logger.warning(f"Failed to index OCR chunks: {e}")

    return image_analyses
