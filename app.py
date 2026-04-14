# app.py
# app.py

import streamlit as st
import os
import tempfile
import re
import html
import gc
import pandas as pd
import time

from ingestion.loader import DocumentLoader
from embeddings.embedder import MultilingualEmbedder
from embeddings.vector_store import VectorStore
from llm.prompt_builder import PromptBuilder
from llm.client_factory import build_llm_client
from llm.hype_generator import HyPEGenerator
from orchestration.langgraph_query import LangGraphQueryOrchestrator
from orchestration.grounding_verifier import GroundingVerifier
from orchestration.llm_judge import LLMJudge
from orchestration.query_intent_router import QueryIntentRouter
from orchestration.query_decomposer import QueryDecomposer
from orchestration.graphrag_router import GraphRAGRouter
from orchestration.graphrag_index import GraphRAGIndex
from orchestration.citation_verifier import CitationVerifier
from orchestration.judge_consensus import JudgeConsensus
from orchestration.hierarchical_retriever import HierarchicalRetriever
from orchestration.tier3_agentic_rag import Tier3AgenticRAG
from orchestration.langgraph_index import LangGraphIndexOrchestrator
from orchestration.langgraph_audio_index import LangGraphAudioIndexOrchestrator
from processing.language_detector import LanguageDetector
from utils.conversation_memory import ConversationMemory
from utils.eval_logger import EvalLogger
from utils.eval_dashboard import EvalDashboard
from utils.pii_guard import PIIGuard
from utils.feedback_store import FeedbackStore
from utils.data_lifecycle import DataLifecycleManager
from utils.task_queue import TaskQueue
from utils.reliability_guard import CircuitBreaker, QualityRollbackGuard
from utils.otel_tracer import OTelTracer
from utils.production_observability import (
    ProductionTelemetry,
    SLOMonitor,
    estimate_tokens,
    estimate_cost_usd,
)
from retrieval.late_interaction_reranker import LateInteractionReranker
from config.settings import (
    ENABLE_HYPE,
    ENABLE_AUDIO_VIDEO_INGESTION,
    ENABLE_CONVERSATION_MEMORY,
    SELF_RAG_HARD_REFUSAL_ENABLED,
    PIPELINE_DEBUG,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAST_CHUNK_SIZE,
    FAST_CHUNK_OVERLAP,
    EMBEDDING_MAX_WORDS_PER_CHUNK,
    FAST_EMBEDDING_MAX_WORDS_PER_CHUNK,
    HYPE_PROMPTS_PER_CHUNK,
    FAST_HYPE_PROMPTS_PER_CHUNK,
    MCQ_RESPONSE_MAX_TOKENS,
    MCQ_CONTINUATION_MAX_ROUNDS,
    RESPONSE_MAX_TOKENS,
    STRUCTURED_RESPONSE_MAX_TOKENS,
    LLM_PROVIDER_DEFAULT,
    GROQ_MODEL_NAME,
    OPENROUTER_MODEL_NAME,
    ENABLE_TIER3_AGENTIC_RAG,
    ENABLE_LLM_JUDGE,
    LLM_JUDGE_MIN_OVERALL_SCORE,
    ENABLE_LANGUAGE_COMPLIANCE_REWRITE,
    LANGUAGE_COMPLIANCE_MIN_CHARS,
    ENABLE_GROUNDING_VERIFIER,
    GROUNDING_MIN_SUPPORT_RATIO,
    ENABLE_SHOW_FLAGGED_ANSWERS,
    ENABLE_QUERY_DECOMPOSITION,
    QUERY_DECOMPOSER_MAX_SUB_QUERIES,
    ENABLE_CITATION_GROUNDING,
    ENABLE_CITATION_AUGMENTATION,
    CITATION_MIN_SUPPORT_RATIO,
    ENABLE_HIERARCHICAL_RETRIEVAL,
    HIERARCHICAL_TOP_DOCUMENTS,
    HIERARCHICAL_TOP_CHUNKS_PER_DOC,
    HIERARCHICAL_DIVERSITY_PENALTY,
    HIERARCHICAL_USE_DIVERSITY,
    ENABLE_EVAL_DASHBOARD,
    ENABLE_PII_REDACTION,
    PII_REDACTION_LOGS_ONLY,
    ENABLE_DEGRADED_MODE,
    ENABLE_CIRCUIT_BREAKER,
    ENABLE_PRODUCTION_TELEMETRY,
    ENABLE_HUMAN_FEEDBACK,
    ENABLE_DATA_LIFECYCLE,
    ENABLE_QUALITY_ROLLBACK_GUARD,
    CITATION_IN_PROMPT,
    ENABLE_ASYNC_INGESTION_WORKERS,
    ENABLE_GRAPHRAG_ROUTER,
    GRAPH_GLOBAL_TOP_K_BOOST,
    GRAPH_COMMUNITY_TOP_K,
    ENABLE_LATE_INTERACTION_RERANK,
    LATE_INTERACTION_TOP_K,
    ENABLE_OTEL_TRACING,
    ENABLE_JUDGE_CONSENSUS,
    JUDGE_CONSENSUS_COUNT,
    JUDGE_MAX_DISAGREEMENT,
    JUDGE_CONSENSUS_SPECS,
)
from pipeline_logger import get_logger, set_debug_enabled


# -------------------------
# Streamlit Page Config
# -------------------------

st.set_page_config(page_title="Indic Multilingual RAG", layout="wide")
st.title("Indic Multilingual Multimodal RAG")

logger = get_logger("app")


def _humanize_graph_step(step_name: str) -> str:
    return (step_name or "").replace("_", " ").strip().title()


def _is_video_file(file_name: str) -> bool:
    ext = os.path.splitext((file_name or "").lower())[1]
    return ext in {".mp4", ".mkv", ".mov", ".webm", ".avi"}


def _reset_runtime_for_mode_switch():
    if "vector_store" in st.session_state and hasattr(st.session_state.vector_store, "clear"):
        st.session_state.vector_store.clear()

    st.session_state.documents_indexed = False
    st.session_state.query_orchestrator = None
    st.session_state.index_orchestrator = None
    st.session_state.audio_index_orchestrator = None
    st.session_state.embedder = None
    st.session_state.av_debug_records = []
    gc.collect()


def _reset_corpus_state(reason: str = "manual_reset"):
    """Clear the persisted corpus and reset index/query runtime objects."""
    try:
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()

        if hasattr(st.session_state.vector_store, "clear"):
            st.session_state.vector_store.clear()
        else:
            st.session_state.vector_store = VectorStore()
    except Exception as ex:
        logger.warning("Corpus reset encountered an error; recreating vector store | reason=%s | error=%s", reason, ex)
        st.session_state.vector_store = VectorStore()
        try:
            st.session_state.vector_store.clear()
        except Exception as inner_ex:
            logger.warning("Fallback corpus clear failed | reason=%s | error=%s", reason, inner_ex)

    st.session_state.documents_indexed = False
    st.session_state.query_orchestrator = None
    st.session_state.index_orchestrator = None
    st.session_state.audio_index_orchestrator = None
    st.session_state.av_debug_records = []
    gc.collect()


def _extract_mcq_target_count(query: str) -> int:
    if not query:
        return 0

    mcq_match = re.search(r"(\d{1,3})\s*(mcq|mcqs|multiple\s*choice)", query, flags=re.IGNORECASE)
    if not mcq_match:
        return 0

    try:
        return int(mcq_match.group(1))
    except (TypeError, ValueError):
        return 0


def _count_mcqs_in_text(answer: str) -> int:
    if not answer:
        return 0

    numbered = re.findall(r"(?m)^\s*\d{1,3}[\)\.:\-]", answer)
    if numbered:
        return len(numbered)

    q_lines = re.findall(r"(?mi)^\s*(q\.?\s*\d+|question\s*\d+)\b", answer)
    return len(q_lines)


def _strip_question_echo(answer: str, query: str) -> str:
    if not answer:
        return answer

    cleaned = answer.strip()
    q = (query or "").strip()
    if not q:
        return cleaned

    def _norm(text: str) -> str:
        return re.sub(r"\W+", "", text or "").lower()

    normalized_query = _norm(q)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    if lines:
        first_line = re.sub(r"(?i)^question\s*[:\-]?\s*", "", lines[0]).strip()
        if _norm(first_line) == normalized_query:
            lines = lines[1:]

    return "\n".join(lines).strip() if lines else cleaned


def _safe_for_logs(payload):
    if ENABLE_PII_REDACTION and PII_REDACTION_LOGS_ONLY:
        return PIIGuard.sanitize_payload(payload)
    return payload


def _build_degraded_answer(query: str, retrieved_docs: list) -> str:
    if not retrieved_docs:
        return (
            "I could not complete full generation right now. "
            "Please retry in a moment."
        )

    lines = [
        "The generator is temporarily unavailable. Here is a grounded extractive summary from top sources:",
        "",
    ]
    for idx, item in enumerate(retrieved_docs[:3], start=1):
        meta = item.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        text = (item.get("text", "") or "").strip().replace("\n", " ")
        lines.append(f"{idx}. [{source} | Page {page}] {text[:280]}")

    lines.append("")
    lines.append("Retry shortly for a full synthesized answer.")
    return "\n".join(lines)


def _enforce_answer_language(
    llm_client,
    query: str,
    answer: str,
    expected_language_code: str,
    expected_language_name: str,
    response_language_instruction: str,
    min_chars: int,
):
    """Rewrite once if final answer language does not match requested output language."""
    if not answer or len((answer or "").strip()) < int(min_chars):
        return answer, {"rewritten": False, "detected": "unknown", "target": expected_language_code}

    detector = LanguageDetector()
    detected = detector.detect_language(answer)
    target = (expected_language_code or "en").strip().lower()

    if detected == target:
        return answer, {"rewritten": False, "detected": detected, "target": target}

    rewrite_prompt = [
        {
            "role": "system",
            "content": (
                "Rewrite the provided answer into exactly one target language. "
                "Do not include bilingual output. Keep meaning unchanged."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Current answer:\n{answer}\n\n"
                f"Target language: {expected_language_name} ({target}).\n"
                f"{response_language_instruction if response_language_instruction else ''}\n\n"
                "Return only the rewritten final answer in the target language."
            ),
        },
    ]

    try:
        rewritten = llm_client.generate(rewrite_prompt, max_tokens=RESPONSE_MAX_TOKENS, temperature=0.0)
        final_answer = (rewritten or answer).strip()
        redetected = detector.detect_language(final_answer)
        return final_answer, {"rewritten": True, "detected": redetected, "target": target}
    except Exception as ex:
        logger.warning("Language compliance rewrite failed | error=%s", ex)
        return answer, {"rewritten": False, "detected": detected, "target": target}


def _apply_runtime_profile(loader: DocumentLoader, embedder: MultilingualEmbedder, hype_generator: HyPEGenerator, fast_mode: bool):
    active_chunk_size = FAST_CHUNK_SIZE if fast_mode else CHUNK_SIZE
    active_overlap = FAST_CHUNK_OVERLAP if fast_mode else CHUNK_OVERLAP
    active_max_words = FAST_EMBEDDING_MAX_WORDS_PER_CHUNK if fast_mode else EMBEDDING_MAX_WORDS_PER_CHUNK
    active_hype_ppc = FAST_HYPE_PROMPTS_PER_CHUNK if fast_mode else HYPE_PROMPTS_PER_CHUNK

    for chunker in [
        loader.chunker,
        loader.pdf_parser.chunker,
        loader.image_loader.chunker,
    ]:
        chunker.chunk_size = active_chunk_size
        chunker.chunk_overlap = active_overlap

    embedder.max_words_per_chunk = active_max_words

    if hype_generator is not None:
        hype_generator.set_prompts_per_chunk(active_hype_ppc)

    logger.info(
        "Runtime profile applied | fast_mode=%s | chunk_size=%d | overlap=%d | max_words=%d | hype_ppc=%d",
        fast_mode,
        active_chunk_size,
        active_overlap,
        active_max_words,
        active_hype_ppc,
    )


def _apply_final_self_rag_gates(query: str, answer: str, retrieved_docs: list, retrieval_output: dict):
    """
    Apply final Self-RAG gates (faithfulness & usefulness) to the generated answer.
    Returns dict with updated confidence scores.
    """
    from orchestration.self_rag_gates import SelfRAGGates
    
    try:
        gates = SelfRAGGates(llm_client=None)  # Uses default Groq client
        
        # Gate 3: Check faithfulness to retrieved documents
        is_faithful, faithfulness_conf, _ = gates.gate_answer_faithfulness(
            query, answer, retrieved_docs
        )
        
        # Gate 4: Check usefulness of answer
        is_useful, usefulness_conf, _ = gates.gate_answer_usefulness(query, answer)
        
        # Get existing scores from retrieval and compute overall confidence
        doc_rel_score = retrieval_output.get("doc_relevance_score", 1.0)
        overall = gates.compute_overall_confidence(
            retrieval_confidence=1.0,
            doc_relevance_confidence=doc_rel_score,
            faithfulness_confidence=faithfulness_conf,
            usefulness_confidence=usefulness_conf,
        )
        badge, level = gates.get_confidence_badge(overall)
        
        logger.info(
            "Self-RAG final gates | faithful=%.2f | useful=%.2f | overall=%.2f | badge=%s",
            faithfulness_conf,
            usefulness_conf,
            overall,
            badge,
        )
        
        return {
            "faithfulness_score": faithfulness_conf,
            "usefulness_score": usefulness_conf,
            "overall_confidence": overall,
            "confidence_badge": badge,
            "confidence_level": level,
        }
    except Exception as ex:
        logger.warning("Self-RAG final gates failed; using defaults | error=%s", ex)
        return {
            "faithfulness_score": 0.5,
            "usefulness_score": 0.5,
            "overall_confidence": 0.5,
            "confidence_badge": "🟡",
            "confidence_level": "MEDIUM",
        }


def _extract_first_markdown_table(text: str):
    if not text:
        return None

    lines = [line.rstrip() for line in text.splitlines()]
    blocks = []
    current = []

    for line in lines:
        if "|" in line:
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []

    if len(current) >= 2:
        blocks.append(current)

    for block in blocks:
        if len(block) >= 2 and re.search(r"\|?\s*[-:]{3,}", block[1]):
            return block

    return None


def _parse_markdown_table(block_lines):
    if not block_lines or len(block_lines) < 2:
        return None, None

    def split_row(line: str):
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        return [part for part in parts]

    headers = split_row(block_lines[0])
    body_lines = block_lines[2:] if len(block_lines) > 2 else []
    rows = [split_row(line) for line in body_lines if line.strip()]

    if not headers:
        return None, None

    normalized_rows = []
    col_count = len(headers)
    for row in rows:
        if len(row) < col_count:
            row = row + [""] * (col_count - len(row))
        elif len(row) > col_count:
            row = row[:col_count]
        normalized_rows.append(row)

    return headers, normalized_rows


def _render_styled_table(headers, rows, title: str):
    if not headers:
        return

    escaped_headers = [html.escape(h) for h in headers]
    escaped_rows = [[html.escape(cell) for cell in row] for row in rows]

    table_html = [
        "<div style='margin-top:0.5rem;margin-bottom:1rem;'>",
        f"<div style='font-size:1.0rem;font-weight:600;margin-bottom:0.5rem;'>{html.escape(title)}</div>",
        "<div style='overflow-x:auto;border:1px solid #2e2e2e;border-radius:12px;'>",
        "<table style='width:100%;border-collapse:collapse;font-size:0.95rem;'>",
        "<thead><tr>",
    ]

    for h in escaped_headers:
        table_html.append(
            f"<th style='padding:10px 12px;background:#111827;border-bottom:1px solid #374151;text-align:left;'>{h}</th>"
        )

    table_html.append("</tr></thead><tbody>")

    for row_idx, row in enumerate(escaped_rows):
        bg = "#0b1220" if row_idx % 2 == 0 else "#111827"
        table_html.append(f"<tr style='background:{bg};'>")
        for cell in row:
            table_html.append(
                f"<td style='padding:10px 12px;border-bottom:1px solid #1f2937;vertical-align:top;'>{cell}</td>"
            )
        table_html.append("</tr>")

    table_html.extend(["</tbody></table></div></div>"])
    st.markdown("".join(table_html), unsafe_allow_html=True)


def _render_matrix_grid(headers, rows, title: str):
    if not headers:
        return

    cols = st.columns(len(headers))
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")

    for row in rows:
        row_cols = st.columns(len(headers))
        for col, cell in zip(row_cols, row):
            col.markdown(
                f"<div style='padding:10px;border:1px solid #2e2e2e;border-radius:8px;background:#0b1220;text-align:center;'>{html.escape(cell)}</div>",
                unsafe_allow_html=True,
            )

    st.caption(title)

# -------------------------
# Initialize Session State
# -------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "embedder" not in st.session_state or st.session_state.embedder is None:
    st.session_state.embedder = MultilingualEmbedder()

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = False

if "query_orchestrator" not in st.session_state:
    st.session_state.query_orchestrator = None

if "index_orchestrator" not in st.session_state:
    st.session_state.index_orchestrator = None

if "audio_index_orchestrator" not in st.session_state:
    st.session_state.audio_index_orchestrator = None

if "av_debug_records" not in st.session_state:
    st.session_state.av_debug_records = []

if "ingestion_mode" not in st.session_state:
    st.session_state.ingestion_mode = "Documents & Code"

if "last_ingestion_mode" not in st.session_state:
    st.session_state.last_ingestion_mode = st.session_state.ingestion_mode

if "debug_enabled" not in st.session_state:
    st.session_state.debug_enabled = PIPELINE_DEBUG

if "fast_mode" not in st.session_state:
    st.session_state.fast_mode = True

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = LLM_PROVIDER_DEFAULT

if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = GROQ_MODEL_NAME if st.session_state.llm_provider == "groq" else OPENROUTER_MODEL_NAME

if "groq_api_key_input" not in st.session_state:
    st.session_state.groq_api_key_input = ""

if "openrouter_api_key_input" not in st.session_state:
    st.session_state.openrouter_api_key_input = ""

if "prod_telemetry" not in st.session_state:
    st.session_state.prod_telemetry = ProductionTelemetry()

if "slo_monitor" not in st.session_state:
    st.session_state.slo_monitor = SLOMonitor()

if "rollback_guard" not in st.session_state:
    st.session_state.rollback_guard = QualityRollbackGuard()

if "task_queue" not in st.session_state:
    st.session_state.task_queue = TaskQueue()

if "otel_tracer" not in st.session_state:
    st.session_state.otel_tracer = OTelTracer(enabled=ENABLE_OTEL_TRACING)

with st.sidebar:
    st.markdown("### LLM Runtime")
    selected_provider_label = st.selectbox(
        "Provider",
        options=["Groq", "OpenRouter"],
        index=0 if st.session_state.llm_provider == "groq" else 1,
    )
    selected_provider = "groq" if selected_provider_label == "Groq" else "openrouter"

    default_model = GROQ_MODEL_NAME if selected_provider == "groq" else OPENROUTER_MODEL_NAME
    model_name = st.text_input(
        "Model",
        value=st.session_state.llm_model_name if st.session_state.llm_provider == selected_provider else default_model,
        help="Provider-specific model id",
    ).strip()

    st.markdown("### API Keys")
    st.text_input(
        "Groq API Key (optional session override)",
        key="groq_api_key_input",
        type="password",
        help="If blank, uses GROQ_API_KEY from .env/environment.",
    )
    st.text_input(
        "OpenRouter API Key (optional session override)",
        key="openrouter_api_key_input",
        type="password",
        help="If blank, uses OPENROUTER_API_KEY from .env/environment.",
    )

    if selected_provider != st.session_state.llm_provider or model_name != st.session_state.llm_model_name:
        st.session_state.llm_provider = selected_provider
        st.session_state.llm_model_name = model_name or default_model
        st.session_state.query_orchestrator = None
        st.session_state.index_orchestrator = None
        st.info("LLM provider/model updated. Query and index orchestrators will reinitialize.")

def _build_active_llm_client():
    provider = st.session_state.llm_provider
    model_name = st.session_state.llm_model_name
    runtime_key = ""
    if provider == "groq":
        runtime_key = (st.session_state.get("groq_api_key_input") or "").strip()
    elif provider == "openrouter":
        runtime_key = (st.session_state.get("openrouter_api_key_input") or "").strip()

    return build_llm_client(provider=provider, model_name=model_name, api_key=runtime_key or None)

st.session_state.debug_enabled = st.checkbox(
    "Enable detailed CLI debug logs",
    value=st.session_state.debug_enabled,
)
st.session_state.fast_mode = st.checkbox(
    "Fast indexing mode (recommended for large PDFs)",
    value=st.session_state.fast_mode,
)
set_debug_enabled(st.session_state.debug_enabled)
logger.info("Debug logging enabled: %s", st.session_state.debug_enabled)
logger.info("Fast indexing mode: %s", st.session_state.fast_mode)

if ENABLE_AUDIO_VIDEO_INGESTION:
    st.session_state.ingestion_mode = st.radio(
        "Ingestion Mode",
        ["Documents & Code", "Audio / Video"],
        index=0 if st.session_state.ingestion_mode == "Documents & Code" else 1,
        horizontal=True,
    )

    if st.session_state.ingestion_mode != st.session_state.last_ingestion_mode:
        _reset_runtime_for_mode_switch()
        st.session_state.last_ingestion_mode = st.session_state.ingestion_mode
        st.warning("Mode changed. Previous corpus and in-memory models were unloaded for clean isolation.")
        st.rerun()


if st.button("Reset Corpus"):
    logger.info("Reset Corpus button clicked")
    _reset_corpus_state(reason="manual_reset_button")
    st.success("Corpus reset complete. Upload and index documents again.")
    logger.info("Corpus reset completed")
    st.rerun()


# -------------------------
# File Upload Section
# -------------------------

is_audio_mode = st.session_state.ingestion_mode == "Audio / Video"

if is_audio_mode:
    uploaded_files = st.file_uploader(
        "Upload audio/video files",
        type=["mp3", "wav", "m4a", "flac", "ogg", "aac", "mp4", "mkv", "mov", "webm", "avi"],
        accept_multiple_files=True,
        key="av_uploader",
    )
    reference_files = st.file_uploader(
        "Optional reference transcripts (.txt) for WER/CER benchmark",
        type=["txt"],
        accept_multiple_files=True,
        key="av_reference_uploader",
    )
else:
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, PPT/PPTX, CSV/Excel, HTML/JSON, code files, images)",
        type=[
            "pdf", "txt", "md", "markdown",
            "docx", "ppt", "pptx",
            "csv", "xlsx", "xls", "json", "html", "htm",
            "py", "java", "c", "h", "cpp", "cc", "cxx", "hpp", "hh", "hxx",
            "js", "jsx", "ts", "tsx", "css", "scss", "sass",
            "go", "rs", "php", "rb", "swift", "kt", "kts", "sql", "xml", "yaml", "yml", "ini", "toml",
            "jpg", "jpeg", "png",
        ],
        accept_multiple_files=True,
        key="doc_uploader",
    )
    reference_files = []

if uploaded_files:
    button_label = "Index Audio/Video" if is_audio_mode else "Index Documents"

    if st.button(button_label):
        logger.info("%s clicked | mode=%s | files=%d", button_label, st.session_state.ingestion_mode, len(uploaded_files))

        # Replace the corpus on each indexing run so only the current upload is searchable.
        _reset_corpus_state(reason="reindex_replace_mode")

        embedder = st.session_state.embedder
        vector_store = st.session_state.vector_store

        temp_file_paths = []
        file_items = []
        uploaded_payloads = {}

        for uploaded_file in uploaded_files:
            logger.info("Processing upload: %s", uploaded_file.name)
            suffix = os.path.splitext(uploaded_file.name)[1]
            file_bytes = uploaded_file.getvalue()
            uploaded_payloads[uploaded_file.name] = file_bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            temp_file_paths.append(tmp_path)
            file_items.append({"path": tmp_path, "source": uploaded_file.name})

        progress_col_1, progress_col_2 = st.columns([2, 5])
        with progress_col_1:
            st.markdown("**LangGraph Indexing**")
        progress_text = progress_col_2.empty()
        progress_bar = st.progress(0)

        def _index_progress(step_name: str, completed: int, total: int):
            ratio = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
            progress_bar.progress(ratio)
            progress_text.info(
                f"Step {completed}/{total}: {_humanize_graph_step(step_name)} ({int(ratio * 100)}%)"
            )

        try:
            if is_audio_mode:
                from ingestion.audio_video_loader import AudioVideoLoader

                av_loader = AudioVideoLoader()

                if st.session_state.audio_index_orchestrator is None:
                    st.session_state.audio_index_orchestrator = LangGraphAudioIndexOrchestrator(
                        av_loader=av_loader,
                        embedder=embedder,
                        vector_store=vector_store,
                    )
                else:
                    st.session_state.audio_index_orchestrator.av_loader = av_loader
                    st.session_state.audio_index_orchestrator.embedder = embedder
                    st.session_state.audio_index_orchestrator.vector_store = vector_store

                ref_by_stem = {}
                for reference_file in (reference_files or []):
                    stem = os.path.splitext(reference_file.name)[0].lower().strip()
                    ref_text = reference_file.getvalue().decode("utf-8", errors="ignore")
                    ref_by_stem[stem] = ref_text

                reference_map = {}
                for item in file_items:
                    src = item.get("source", "")
                    stem = os.path.splitext(src)[0].lower().strip()
                    reference_map[src] = ref_by_stem.get(stem, "")

                if ENABLE_ASYNC_INGESTION_WORKERS:
                    # Streamlit UI updates are not thread-safe from worker threads.
                    index_output = st.session_state.task_queue.run(
                        st.session_state.audio_index_orchestrator.index_audio_video,
                        file_items=file_items,
                        reference_map=reference_map,
                        progress_callback=None,
                    )
                else:
                    index_output = st.session_state.audio_index_orchestrator.index_audio_video(
                        file_items=file_items,
                        reference_map=reference_map,
                        progress_callback=_index_progress,
                    )
                st.session_state.av_debug_records = index_output.get("debug_records", [])
            else:
                loader = DocumentLoader()
                hype_generator = None
                if ENABLE_HYPE:
                    try:
                        hype_generator = HyPEGenerator(llm_client=_build_active_llm_client())
                    except Exception as ex:
                        logger.warning("HyPE disabled for this run due to LLM client error | error=%s", ex)
                        st.warning(f"HyPE temporarily disabled: {ex}")
                _apply_runtime_profile(loader, embedder, hype_generator, st.session_state.fast_mode)

                if st.session_state.index_orchestrator is None:
                    st.session_state.index_orchestrator = LangGraphIndexOrchestrator(
                        loader=loader,
                        embedder=embedder,
                        vector_store=vector_store,
                        hype_generator=hype_generator,
                    )
                else:
                    st.session_state.index_orchestrator.loader = loader
                    st.session_state.index_orchestrator.embedder = embedder
                    st.session_state.index_orchestrator.vector_store = vector_store
                    st.session_state.index_orchestrator.hype_generator = hype_generator
                    st.session_state.index_orchestrator.enable_hype = ENABLE_HYPE and hype_generator is not None

                if ENABLE_ASYNC_INGESTION_WORKERS:
                    # Streamlit UI updates are not thread-safe from worker threads.
                    index_output = st.session_state.task_queue.run(
                        st.session_state.index_orchestrator.index_documents,
                        file_items=file_items,
                        progress_callback=None,
                    )
                else:
                    index_output = st.session_state.index_orchestrator.index_documents(
                        file_items=file_items,
                        progress_callback=_index_progress,
                    )
                st.session_state.av_debug_records = []
        finally:
            for path in temp_file_paths:
                if os.path.exists(path):
                    os.remove(path)

        all_chunks = index_output.get("all_chunks", [])
        hype_count = index_output.get("hype_count", 0)
        source_counts = index_output.get("source_counts", {})

        if all_chunks:
            st.write(f"Indexed chunks: {len(all_chunks)}")
            if ENABLE_HYPE and not is_audio_mode:
                st.write(f"Indexed HyPE prompts: {hype_count}")
            st.write("Chunks per file:")
            for source_name, count in sorted(source_counts.items(), key=lambda x: x[0].lower()):
                st.write(f"- {source_name}: {count}")

            if is_audio_mode and st.session_state.av_debug_records:
                st.subheader("ASR Debug + Benchmark")
                for debug in st.session_state.av_debug_records:
                    source = debug.get("source", "unknown")
                    with st.expander(f"ASR Debug: {source}", expanded=False):
                        payload = uploaded_payloads.get(source)
                        if payload:
                            if _is_video_file(source):
                                st.video(payload)
                            else:
                                st.audio(payload)

                        st.write(f"Detected language: {debug.get('detected_language', 'unknown')}")
                        benchmark = debug.get("benchmark", {})
                        if benchmark:
                            st.write("Benchmark:")
                            for key, value in benchmark.items():
                                st.write(f"- {key}: {value}")

                        transcript = debug.get("transcript", "")
                        if transcript:
                            st.text_area(
                                f"Transcript preview - {source}",
                                value=transcript[:6000],
                                height=180,
                                key=f"transcript_preview_{source}",
                            )

                        segments = debug.get("segments", [])
                        if segments:
                            st.write("Segments (first 25):")
                            for seg in segments[:25]:
                                st.write(
                                    f"[{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] {seg.get('text', '')}"
                                )

            st.session_state.documents_indexed = True
            if ENABLE_DATA_LIFECYCLE:
                try:
                    DataLifecycleManager().register_ingestion(file_items)
                except Exception as ex:
                    logger.warning("Data lifecycle registration failed | error=%s", ex)

            if ENABLE_GRAPHRAG_ROUTER:
                try:
                    graph_chunks = [
                        {
                            "text": c.get("text", ""),
                            "metadata": c.get("metadata", {}),
                        }
                        for c in all_chunks
                        if isinstance(c, dict)
                    ]
                    GraphRAGIndex().build(graph_chunks)
                except Exception as ex:
                    logger.warning("Graph index build failed | error=%s", ex)
            st.success(f"{button_label} completed successfully.")
            logger.info("Indexing completed successfully | mode=%s", st.session_state.ingestion_mode)
        else:
            st.warning("No extractable content found.")
            logger.warning("Indexing produced zero chunks | mode=%s", st.session_state.ingestion_mode)


# -------------------------
# Main Tabs: Query, Eval Dashboard
# -------------------------

if st.session_state.documents_indexed:
    st.divider()

    if ENABLE_DATA_LIFECYCLE:
        try:
            stale = DataLifecycleManager().stale_sources()
            if stale:
                st.warning(f"Stale indexed sources detected ({len(stale)}). Consider re-indexing soon.")
        except Exception as ex:
            logger.warning("Stale source check failed | error=%s", ex)
    
    tab_query, tab_eval_dashboard = st.tabs(["Query & Retrieval", "Eval Dashboard"])
    
    # ========== TAB 1: Query & Retrieval ==========
    with tab_query:
        st.subheader("Ask a Question")

    with st.expander("Structured Generation (Matrix / Table)", expanded=False):
        generation_mode = st.selectbox(
            "Generation mode",
            ["Normal", "Matrix", "Table"],
            index=0,
        )

        gen_cols = st.columns(2)
        with gen_cols[0]:
            matrix_rows = st.number_input("Rows", min_value=2, max_value=12, value=3, step=1)
        with gen_cols[1]:
            matrix_cols = st.number_input("Columns", min_value=2, max_value=12, value=3, step=1)

        st.caption("For Matrix/Table mode, output will be enforced as Markdown table and rendered visually.")

    with st.expander("Advanced Retrieval Controls", expanded=False):
        source_filter = st.text_input("Source name contains (optional)").strip().lower()
        recency_boost = st.slider("Recency boost", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

    query = st.text_input("Enter your question")

    if query and st.button("Get Answer"):
        query_start = time.perf_counter()
        logger.info("Get Answer clicked | query=%s", _safe_for_logs(query))

        prompt_builder = PromptBuilder()
        provider_key = st.session_state.llm_provider
        model_key = st.session_state.llm_model_name
        breaker = CircuitBreaker(f"{provider_key}:{model_key}")
        llm_allowed = (not ENABLE_CIRCUIT_BREAKER) or breaker.allow_request()
        try:
            llm_client = _build_active_llm_client() if llm_allowed else None
        except Exception as ex:
            llm_client = None
            breaker.record_failure()
            logger.exception("LLM client initialization failed | error=%s", ex)

        if not llm_allowed:
            st.warning("Primary model circuit breaker is open; running in degraded mode.")

        if llm_client is None and st.session_state.query_orchestrator is None:
            st.error("LLM provider is unavailable and no warm query orchestrator exists yet. Retry after cooldown.")
            st.stop()

        if st.session_state.query_orchestrator is None:
            with st.spinner("Initializing LangGraph retrieval orchestrator..."):
                st.session_state.query_orchestrator = LangGraphQueryOrchestrator(llm_client=llm_client)
            logger.info("LangGraph query orchestrator initialized")

        # Query Decomposition (optional multi-part question handling)
        final_query = query
        decomposition_meta = {"is_multi_part": False, "sub_queries": []}
        if ENABLE_QUERY_DECOMPOSITION:
            try:
                decomposer = QueryDecomposer(llm_client=llm_client)
                decomposition_meta = decomposer.decompose(
                    query,
                    max_sub_queries=QUERY_DECOMPOSER_MAX_SUB_QUERIES,
                )
                if decomposition_meta.get("is_multi_part") and decomposition_meta.get("sub_queries"):
                    final_query = " ; ".join(decomposition_meta.get("sub_queries", []))
                    if st.session_state.debug_enabled:
                        st.caption(
                            f"Query decomposed: {decomposition_meta.get('decomposition_notes')}"
                        )
                        with st.expander("Sub-queries", expanded=False):
                            for i, sq in enumerate(decomposition_meta["sub_queries"], 1):
                                st.write(f"{i}. {sq}")
            except Exception as ex:
                logger.warning("Query decomposition failed; using original query | error=%s", ex)

        intent_router = QueryIntentRouter()
        intent_policy = intent_router.route(final_query)
        routed_top_k = int(intent_policy.get("top_k", 5))

        graph_route = {"mode": "baseline_local", "reason": "disabled", "graph_enabled": False}
        if ENABLE_GRAPHRAG_ROUTER:
            graph_route = GraphRAGRouter().route(final_query, intent_policy.get("intent", "qa"))
            if graph_route.get("graph_enabled"):
                routed_top_k = routed_top_k + int(GRAPH_GLOBAL_TOP_K_BOOST)

        if st.session_state.debug_enabled:
            st.caption(
                f"Intent: {intent_policy.get('intent', 'qa')} | Route={graph_route.get('mode')} | Routed top_k={routed_top_k}"
            )

        with st.spinner("Retrieving relevant context..."):
            tracer = st.session_state.otel_tracer
            progress_col_1, progress_col_2 = st.columns([2, 5])
            with progress_col_1:
                st.markdown("**LangGraph Retrieval**")
            retrieval_progress_text = progress_col_2.empty()
            retrieval_progress_bar = st.progress(0)

            def _query_progress(step_name: str, completed: int, total: int):
                ratio = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
                retrieval_progress_bar.progress(ratio)
                retrieval_progress_text.info(
                    f"Step {completed}/{total}: {_humanize_graph_step(step_name)} ({int(ratio * 100)}%)"
                )

            with tracer.span(
                "rag.retrieve",
                {
                    "gen_ai.operation.name": "retrieve",
                    "gen_ai.provider.name": st.session_state.llm_provider,
                    "rag.intent": intent_policy.get("intent", "qa"),
                    "rag.route.mode": graph_route.get("mode", "baseline_local"),
                    "rag.top_k": routed_top_k,
                },
            ):
                retrieval_output = st.session_state.query_orchestrator.retrieve(
                    final_query,
                    top_k=routed_top_k,
                    progress_callback=_query_progress,
                    conversation_history=st.session_state.conversation_memory.get_history(num_turns=3) if ENABLE_CONVERSATION_MEMORY else None,
                )

            query_language = retrieval_output["query_language"]
            response_language = retrieval_output.get("response_language", query_language)
            response_language_name = retrieval_output.get("response_language_name", "English")
            response_script = retrieval_output.get("response_script", "Latin")
            response_language_reason = retrieval_output.get("response_language_reason", "detected")
            response_language_instruction = retrieval_output.get("response_language_instruction", "")
            reranked_results = retrieval_output["results"]

            if graph_route.get("graph_enabled"):
                try:
                    with tracer.span(
                        "rag.retrieve.graph",
                        {
                            "gen_ai.operation.name": "retrieve",
                            "rag.route.mode": "graph_global",
                        },
                    ):
                        graph_results = GraphRAGIndex().query(final_query, top_k=GRAPH_COMMUNITY_TOP_K)
                    if graph_results:
                        reranked_results = (graph_results + reranked_results)[: max(len(graph_results), routed_top_k)]
                        retrieval_output["results"] = reranked_results
                except Exception as ex:
                    logger.warning("Graph retrieval enrichment failed | error=%s", ex)

            if ENABLE_HIERARCHICAL_RETRIEVAL:
                try:
                    query_embedding = st.session_state.embedder.embed_query(query)
                    hierarchical = HierarchicalRetriever(st.session_state.vector_store)
                    if HIERARCHICAL_USE_DIVERSITY:
                        h_result = hierarchical.retrieve_with_diversity(
                            query_embedding=query_embedding,
                            top_documents=HIERARCHICAL_TOP_DOCUMENTS,
                            top_chunks_per_doc=HIERARCHICAL_TOP_CHUNKS_PER_DOC,
                            diversity_penalty=HIERARCHICAL_DIVERSITY_PENALTY,
                        )
                    else:
                        h_result = hierarchical.retrieve_hierarchical(
                            query_embedding=query_embedding,
                            top_documents=HIERARCHICAL_TOP_DOCUMENTS,
                            top_chunks_per_doc=HIERARCHICAL_TOP_CHUNKS_PER_DOC,
                        )
                    reranked_results = h_result.get("results", reranked_results)
                    retrieval_output["results"] = reranked_results
                    if st.session_state.debug_enabled:
                        st.caption(
                            f"Hierarchical retrieval active | docs={h_result.get('stage1_doc_count', 0)} | chunks={len(reranked_results)}"
                        )
                except Exception as ex:
                    logger.warning("Hierarchical retrieval failed; using orchestrator results | error=%s", ex)

            if source_filter:
                reranked_results = [
                    item for item in reranked_results
                    if source_filter in str(item.get("metadata", {}).get("source", "")).lower()
                ]

            if recency_boost > 0 and reranked_results and ENABLE_DATA_LIFECYCLE:
                try:
                    ts_map = DataLifecycleManager().get_source_timestamps()

                    def _recency_score(meta):
                        source = meta.get("source", "")
                        ts = ts_map.get(source, "")
                        if not ts:
                            return 0.0
                        try:
                            dt = pd.to_datetime(ts, utc=True)
                            age_days = max(0.0, (pd.Timestamp.utcnow() - dt).total_seconds() / 86400.0)
                            return 1.0 / (1.0 + age_days)
                        except Exception:
                            return 0.0

                    for item in reranked_results:
                        base_score = float(item.get("rerank_score", item.get("score", 0.0)))
                        r_score = _recency_score(item.get("metadata", {}))
                        item["rerank_score"] = base_score + (float(recency_boost) * r_score)

                    reranked_results.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0))), reverse=True)
                except Exception as ex:
                    logger.warning("Recency boost scoring failed | error=%s", ex)

            if ENABLE_LATE_INTERACTION_RERANK and reranked_results:
                try:
                    with tracer.span(
                        "rag.rerank.late_interaction",
                        {
                            "gen_ai.operation.name": "rerank",
                            "rag.stage": "late_interaction",
                        },
                    ):
                        reranked_results = LateInteractionReranker().rerank(
                            query=final_query,
                            docs=reranked_results,
                            top_k=min(max(1, int(LATE_INTERACTION_TOP_K)), len(reranked_results)),
                        )
                    retrieval_output["results"] = reranked_results
                except Exception as ex:
                    logger.warning("Late-interaction rerank failed; continuing | error=%s", ex)

            logger.info(
                "LangGraph retrieval complete | results=%d | query_language=%s",
                len(reranked_results),
                query_language,
            )
            logger.info(
                "Resolved response language | code=%s | name=%s | script=%s | reason=%s",
                response_language,
                response_language_name,
                response_script,
                response_language_reason,
            )

            if st.session_state.debug_enabled:
                st.caption(
                    (
                        f"Detected query language: {query_language} | "
                        f"Response language: {response_language_name} ({response_language}) "
                        f"[{response_script}] via {response_language_reason}"
                    )
                )
            if st.session_state.debug_enabled and reranked_results:
                for i, item in enumerate(reranked_results[:5], start=1):
                    meta = item.get("metadata", {})
                    logger.debug(
                        "Reranked[%d] | source=%s | page=%s | score=%.4f | semantic=%.4f | hype=%.4f",
                        i,
                        meta.get("source"),
                        meta.get("page"),
                        float(item.get("rerank_score", item.get("score", 0.0))),
                        float(item.get("semantic_score", 0.0)),
                        float(item.get("hype_score", 0.0)),
                    )

        with st.spinner("Generating answer..."):
            requested_mcq_count = _extract_mcq_target_count(query)
            messages = prompt_builder.build_prompt(
                query=query,
                query_language=response_language,
                retrieved_chunks=reranked_results,
                response_language_instruction=response_language_instruction,
                response_language_name=response_language_name,
            )

            if CITATION_IN_PROMPT:
                messages.append(
                    {
                        "role": "user",
                        "content": "Include inline citations in the format [Source N] for factual claims wherever possible.",
                    }
                )

            if generation_mode == "Matrix":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Return the final answer as a markdown table with exactly {int(matrix_rows)} rows and {int(matrix_cols)} columns. "
                            "Use concise cell text. Output only the table."
                        ),
                    }
                )
                logger.info("Structured mode | Matrix | rows=%d | cols=%d", matrix_rows, matrix_cols)
            elif generation_mode == "Table":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return the final answer as a clean markdown table with meaningful headers. "
                            "Output only the table."
                        ),
                    }
                )
                logger.info("Structured mode | Table")

            if requested_mcq_count > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Return exactly {requested_mcq_count} MCQs in the same language as the question. "
                            "Each MCQ must include options and the correct answer. "
                            "Do not stop early."
                        ),
                    }
                )
                logger.info("MCQ mode detected | target_count=%d", requested_mcq_count)

            if requested_mcq_count > 0:
                max_tokens = MCQ_RESPONSE_MAX_TOKENS
            elif generation_mode in ["Matrix", "Table"]:
                max_tokens = STRUCTURED_RESPONSE_MAX_TOKENS
            else:
                max_tokens = RESPONSE_MAX_TOKENS

            generated_in_degraded_mode = False
            if llm_client is None and ENABLE_DEGRADED_MODE:
                answer = _build_degraded_answer(query=query, retrieved_docs=reranked_results)
                generated_in_degraded_mode = True
            else:
                try:
                    with st.session_state.otel_tracer.span(
                        "rag.generate",
                        {
                            "gen_ai.operation.name": "chat.completions",
                            "gen_ai.provider.name": st.session_state.llm_provider,
                            "gen_ai.request.model": st.session_state.llm_model_name,
                            "gen_ai.usage.input_tokens": estimate_tokens("\n".join([m.get("content", "") for m in messages])),
                        },
                    ):
                        answer = llm_client.generate(
                            messages,
                            max_tokens=max_tokens,
                            temperature=0.2,
                        )
                    breaker.record_success()
                except Exception as ex:
                    breaker.record_failure()
                    if ENABLE_DEGRADED_MODE:
                        logger.warning("Generation failed, using degraded mode | error=%s", ex)
                        answer = _build_degraded_answer(query=query, retrieved_docs=reranked_results)
                        generated_in_degraded_mode = True
                    else:
                        raise

            if requested_mcq_count > 0 and llm_client is not None and not generated_in_degraded_mode:
                generated_mcqs = _count_mcqs_in_text(answer)
                logger.info("MCQ count after first pass | generated=%d | target=%d", generated_mcqs, requested_mcq_count)

                rounds = 0
                while generated_mcqs < requested_mcq_count and rounds < MCQ_CONTINUATION_MAX_ROUNDS:
                    rounds += 1
                    remaining = requested_mcq_count - generated_mcqs
                    logger.info(
                        "MCQ continuation round=%d | remaining=%d",
                        rounds,
                        remaining,
                    )

                    continuation_messages = [
                        {"role": "system", "content": "You continue incomplete outputs exactly as instructed."},
                        {"role": "assistant", "content": answer},
                        {
                            "role": "user",
                            "content": (
                                f"Generate exactly {remaining} additional MCQs only, continuing numbering from {generated_mcqs + 1}. "
                                "Do not repeat existing questions."
                            ),
                        },
                    ]

                    continuation = llm_client.generate(
                        continuation_messages,
                        max_tokens=MCQ_RESPONSE_MAX_TOKENS,
                        temperature=0.2,
                    )

                    answer = (answer.rstrip() + "\n\n" + (continuation or "").strip()).strip()
                    generated_mcqs = _count_mcqs_in_text(answer)
                    logger.info(
                        "MCQ count after continuation round=%d | generated=%d | target=%d",
                        rounds,
                        generated_mcqs,
                        requested_mcq_count,
                    )

            logger.info("Answer generated | chars=%d", len(answer or ""))

        answer = _strip_question_echo(answer, query)

        language_compliance_meta = {"rewritten": False, "detected": "unknown", "target": response_language}
        if ENABLE_LANGUAGE_COMPLIANCE_REWRITE and llm_client is not None:
            answer, language_compliance_meta = _enforce_answer_language(
                llm_client=llm_client,
                query=query,
                answer=answer,
                expected_language_code=response_language,
                expected_language_name=response_language_name,
                response_language_instruction=response_language_instruction,
                min_chars=LANGUAGE_COMPLIANCE_MIN_CHARS,
            )
            if st.session_state.debug_enabled and language_compliance_meta.get("rewritten"):
                st.caption(
                    (
                        "Language compliance rewrite applied | "
                        f"detected={language_compliance_meta.get('detected')} -> target={language_compliance_meta.get('target')}"
                    )
                )

        # Tier 3: Agentic answer refinement loop (faithfulness/usefulness guided)
        tier3_meta = {"enabled": False, "refined": False, "rounds_used": 0}
        if ENABLE_TIER3_AGENTIC_RAG and llm_client is not None:
            try:
                tier3_agent = Tier3AgenticRAG(
                    llm_client=llm_client,
                    gates=st.session_state.query_orchestrator.self_rag_gates,
                )
                answer, tier3_meta = tier3_agent.refine_answer(
                    query=query,
                    answer=answer,
                    retrieved_docs=reranked_results,
                    response_language_instruction=response_language_instruction,
                    retrieval_confidence=float(retrieval_output.get("doc_relevance_score", 1.0)),
                )
                if st.session_state.debug_enabled and tier3_meta.get("refined"):
                    st.caption(
                        f"Tier 3 refinement applied | rounds={tier3_meta.get('rounds_used', 0)}"
                    )
            except Exception as ex:
                logger.warning("Tier 3 refinement failed; using original answer | error=%s", ex)
        
        # Apply final Self-RAG gates to the generated answer
        confidence_scores = _apply_final_self_rag_gates(query, answer, reranked_results, retrieval_output)

        grounding_result = {
            "support_ratio": 0.5,
            "supported_sentences": 0,
            "total_sentences": 0,
            "unsupported_examples": [],
        }
        if ENABLE_GROUNDING_VERIFIER:
            try:
                grounding_result = GroundingVerifier().evaluate(answer=answer, retrieved_docs=reranked_results)
                logger.info(
                    "Grounding verifier | support_ratio=%.2f | supported=%d/%d",
                    float(grounding_result.get("support_ratio", 0.5)),
                    int(grounding_result.get("supported_sentences", 0)),
                    int(grounding_result.get("total_sentences", 0)),
                )
            except Exception as ex:
                logger.warning("Grounding verifier failed | error=%s", ex)

        judge_result = {
            "overall_score": 0.5,
            "verdict": "caution",
            "notes": "judge_disabled",
            "retrieval": {"relevance": 0.5, "coverage": 0.5, "noise": 0.5},
            "generation": {"faithfulness": 0.5, "completeness": 0.5, "language_adherence": 0.5},
        }
        if ENABLE_LLM_JUDGE and llm_client is not None:
            try:
                with st.session_state.otel_tracer.span(
                    "rag.judge",
                    {
                        "gen_ai.operation.name": "judge",
                        "rag.judge.consensus_enabled": bool(ENABLE_JUDGE_CONSENSUS),
                    },
                ):
                    if ENABLE_JUDGE_CONSENSUS:
                        judge_result = JudgeConsensus(
                            llm_client=llm_client,
                            judges=JUDGE_CONSENSUS_COUNT,
                            judge_specs=JUDGE_CONSENSUS_SPECS,
                        ).evaluate(
                            query=query,
                            retrieved_docs=reranked_results,
                            answer=answer,
                            expected_language_code=response_language,
                            expected_language_name=response_language_name,
                        )
                    else:
                        judge = LLMJudge(llm_client=llm_client)
                        judge_result = judge.evaluate(
                            query=query,
                            retrieved_docs=reranked_results,
                            answer=answer,
                            expected_language_code=response_language,
                            expected_language_name=response_language_name,
                        )
                logger.info(
                    "LLM Judge | overall=%.2f | verdict=%s",
                    float(judge_result.get("overall_score", 0.5)),
                    judge_result.get("verdict", "caution"),
                )
            except Exception as ex:
                logger.warning("LLM Judge failed; proceeding with Self-RAG only | error=%s", ex)

        # Citation-Grounded Generation verification
        citation_result = {
            "citations_found": [],
            "uncited_claims": [],
            "claim_support_ratio": 0.7,
            "issues": [],
            "is_valid": True,
        }
        if ENABLE_CITATION_GROUNDING:
            try:
                citation_verifier = CitationVerifier(llm_client=llm_client if ENABLE_CITATION_AUGMENTATION and llm_client is not None else None)
                citation_result = citation_verifier.verify_citations(
                    answer=answer,
                    retrieved_docs=reranked_results,
                )
                logger.info(
                    "Citation verification | support_ratio=%.2f | citations=%d",
                    float(citation_result.get("claim_support_ratio", 0.7)),
                    len(citation_result.get("citations_found", [])),
                )
                
                # Augment answer with missing citations if enabled
                if ENABLE_CITATION_AUGMENTATION and not citation_result.get("is_valid", True):
                    try:
                        augmented_answer, aug_meta = citation_verifier.augment_answer_with_citations(
                            answer=answer,
                            retrieved_docs=reranked_results,
                        )
                        if aug_meta.get("augmented"):
                            answer = augmented_answer
                            logger.info("Citation augmentation applied")
                    except Exception as ex:
                        logger.warning("Citation augmentation failed | error=%s", ex)
            except Exception as ex:
                logger.warning("Citation verification failed | error=%s", ex)

        st.subheader("Answer")
        
        # Display confidence badge and contextual retrieval info
        badge = confidence_scores.get("confidence_badge", "🟡")
        level = confidence_scores.get("confidence_level", "MEDIUM")
        confidence = confidence_scores.get("overall_confidence", 0.5)
        judge_overall = float(judge_result.get("overall_score", 0.5))

        if ENABLE_LLM_JUDGE:
            confidence = min(confidence, judge_overall)
            badge, level = st.session_state.query_orchestrator.self_rag_gates.get_confidence_badge(confidence)

        if ENABLE_GROUNDING_VERIFIER:
            support_ratio = float(grounding_result.get("support_ratio", 0.5))
            confidence = min(confidence, support_ratio)
            badge, level = st.session_state.query_orchestrator.self_rag_gates.get_confidence_badge(confidence)
        
        info_parts = [
            f"{badge} **Confidence: {level.title()}** ({confidence:.0%}) | "
            f"Faithfulness: {confidence_scores.get('faithfulness_score', 0.5):.0%} | "
            f"Usefulness: {confidence_scores.get('usefulness_score', 0.5):.0%}"
        ]

        if ENABLE_LLM_JUDGE:
            info_parts.append(
                f"Judge: {judge_result.get('verdict', 'caution').title()} ({judge_overall:.0%})"
            )

        if ENABLE_GROUNDING_VERIFIER:
            info_parts.append(
                f"Grounding: {float(grounding_result.get('support_ratio', 0.5)):.0%}"
            )

        if ENABLE_CITATION_GROUNDING:
            info_parts.append(
                f"Citation Support: {float(citation_result.get('claim_support_ratio', 0.7)):.0%}"
            )
        
        # Show if query was contextualized from history
        if retrieval_output.get("query_contextualized"):
            original = retrieval_output.get("original_query", query)
            info_parts.append(f"**Tip:** Query was contextualized from history")
        
        st.caption(" | ".join(info_parts))
        
        # Hard refusal check
        should_refuse, refusal_reason = st.session_state.query_orchestrator.self_rag_gates.should_refuse_answer(confidence)
        if ENABLE_LLM_JUDGE and judge_overall < float(LLM_JUDGE_MIN_OVERALL_SCORE):
            should_refuse = True
            refusal_reason = f"judge_low_score_{judge_overall:.2f}"
        if ENABLE_LLM_JUDGE and ENABLE_JUDGE_CONSENSUS:
            disagreement = float(judge_result.get("consensus_meta", {}).get("disagreement", 0.0))
            if disagreement > float(JUDGE_MAX_DISAGREEMENT):
                should_refuse = True
                refusal_reason = f"judge_high_disagreement_{disagreement:.2f}"
        if ENABLE_GROUNDING_VERIFIER and float(grounding_result.get("support_ratio", 0.5)) < float(GROUNDING_MIN_SUPPORT_RATIO):
            should_refuse = True
            refusal_reason = f"grounding_low_support_{float(grounding_result.get('support_ratio', 0.5)):.2f}"
        if ENABLE_CITATION_GROUNDING and float(citation_result.get("claim_support_ratio", 0.7)) < float(CITATION_MIN_SUPPORT_RATIO):
            should_refuse = True
            refusal_reason = f"citation_low_support_{float(citation_result.get('claim_support_ratio', 0.7)):.2f}"
        if should_refuse:
            st.warning(
                f"⚠️ **Unable to provide a reliable answer** \n\n"
                f"Confidence score ({confidence:.0%}) is below the reliability threshold. "
                f"The retrieved documents may not contain sufficient information to answer your question. "
                f"Please try:\n"
                f"- Rephrasing your question\n"
                f"- Uploading additional relevant documents\n"
                f"- Checking if the necessary context is in your indexed documents"
            )
            logger.warning("Hard refusal triggered | confidence=%.2f | reason=%s", confidence, refusal_reason)

            if ENABLE_SHOW_FLAGGED_ANSWERS:
                st.info(
                    "Showing flagged answer for debugging only. "
                    "Treat this as low-trust output."
                )
                with st.expander("Flagged Answer (Low Confidence)", expanded=True):
                    st.write(answer)

                if st.session_state.debug_enabled and ENABLE_LLM_JUDGE:
                    with st.expander("LLM Judge Details"):
                        st.json(judge_result)

                if st.session_state.debug_enabled and ENABLE_GROUNDING_VERIFIER:
                    with st.expander("Grounding Verifier Details"):
                        st.json(grounding_result)

                if st.session_state.debug_enabled and ENABLE_CITATION_GROUNDING:
                    with st.expander("Citation Verifier Details"):
                        st.json(citation_result)
        else:
            st.write(answer)

            if st.session_state.debug_enabled and ENABLE_LLM_JUDGE:
                with st.expander("LLM Judge Details"):
                    st.json(judge_result)

            if st.session_state.debug_enabled and ENABLE_GROUNDING_VERIFIER:
                with st.expander("Grounding Verifier Details"):
                    st.json(grounding_result)

            if st.session_state.debug_enabled and ENABLE_CITATION_GROUNDING:
                with st.expander("Citation Verifier Details"):
                    st.json(citation_result)
            
            # Store in conversation memory
            if ENABLE_CONVERSATION_MEMORY:
                st.session_state.conversation_memory.add_turn(
                    question=query,
                    answer=answer[:500],
                    metadata={
                        "confidence": confidence,
                        "badge": badge,
                        "contextualized": retrieval_output.get("query_contextualized", False),
                    }
                )

        try:
            eval_payload = {
                    "query": query,
                    "intent": intent_policy.get("intent", "qa"),
                    "route": graph_route,
                    "routed_top_k": routed_top_k,
                    "retrieved_count": len(reranked_results),
                    "response_language": response_language,
                    "confidence": float(confidence),
                    "self_rag": {
                        "faithfulness": float(confidence_scores.get("faithfulness_score", 0.5)),
                        "usefulness": float(confidence_scores.get("usefulness_score", 0.5)),
                    },
                    "judge": judge_result,
                    "judge_consensus": judge_result.get("consensus_meta", {}) if ENABLE_JUDGE_CONSENSUS else {},
                    "grounding": grounding_result,
                    "citations": citation_result if ENABLE_CITATION_GROUNDING else {},
                    "decomposition": {
                        "is_multi_part": decomposition_meta.get("is_multi_part", False),
                        "sub_query_count": len(decomposition_meta.get("sub_queries", [])),
                    } if ENABLE_QUERY_DECOMPOSITION else {},
                    "language_compliance": language_compliance_meta,
                    "degraded_mode": bool(generated_in_degraded_mode),
                    "refused": bool(should_refuse),
                    "refusal_reason": refusal_reason if should_refuse else "",
                }
            EvalLogger().write(_safe_for_logs(eval_payload))
        except Exception as ex:
            logger.warning("Eval logger write failed | error=%s", ex)

        query_ms = (time.perf_counter() - query_start) * 1000.0
        prompt_tokens = estimate_tokens("\n".join([m.get("content", "") for m in messages]))
        completion_tokens = estimate_tokens(answer)
        est_cost = estimate_cost_usd(
            provider=provider_key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        if ENABLE_PRODUCTION_TELEMETRY:
            st.session_state.prod_telemetry.write(
                _safe_for_logs(
                    {
                        "provider": provider_key,
                        "model": model_key,
                        "latency_ms": query_ms,
                        "prompt_tokens_est": prompt_tokens,
                        "completion_tokens_est": completion_tokens,
                        "cost_est_usd": est_cost,
                        "confidence": float(confidence),
                        "judge_score": float(judge_result.get("overall_score", 0.5)),
                        "judge_disagreement": float(judge_result.get("consensus_meta", {}).get("disagreement", 0.0)),
                        "route_mode": graph_route.get("mode", "baseline_local"),
                        "refused": bool(should_refuse),
                    }
                )
            )

        slo = st.session_state.slo_monitor.record(
            latency_ms=query_ms,
            success=not should_refuse,
            language_adherence=(language_compliance_meta.get("detected") == response_language) if answer else True,
        )
        if slo.get("breached"):
            st.warning("SLO breach detected. Reliability alert has been recorded.")

        if ENABLE_QUALITY_ROLLBACK_GUARD:
            rollback_state = st.session_state.rollback_guard.record(
                confidence=float(confidence),
                judge_score=float(judge_result.get("overall_score", 0.5)),
            )
            if rollback_state.get("triggered"):
                st.error("Quality rollback guard triggered: recent quality drift detected.")

        table_block = _extract_first_markdown_table(answer)
        if table_block and not should_refuse:
            headers, rows = _parse_markdown_table(table_block)
            if headers is not None and rows is not None:
                if generation_mode == "Matrix":
                    st.subheader("Matrix View")
                    _render_matrix_grid(headers, rows, "Rendered matrix grid")
                    st.subheader("Table View")
                    _render_styled_table(headers, rows, "Rendered markdown table")
                else:
                    st.subheader("Table View")
                    _render_styled_table(headers, rows, "Rendered markdown table")

        st.subheader("Sources")
        for item in reranked_results:
            meta = item["metadata"]
            st.write(
                f"- {meta.get('source')} | Page {meta.get('page')} | Score {item.get('rerank_score', item.get('score')):.4f}"
            )

        if ENABLE_HUMAN_FEEDBACK:
            st.divider()
            st.subheader("Feedback")
            feedback_choice = st.radio("Was this answer helpful?", ["Yes", "No"], horizontal=True, key=f"feedback_choice_{hash(query)}")
            feedback_note = st.text_input("Optional feedback note", key=f"feedback_note_{hash(query)}")
            if st.button("Submit Feedback", key=f"submit_feedback_{hash(query)}"):
                try:
                    FeedbackStore().write(
                        _safe_for_logs(
                            {
                                "query": query,
                                "intent": intent_policy.get("intent", "qa"),
                                "helpful": feedback_choice == "Yes",
                                "note": feedback_note,
                                "confidence": float(confidence),
                                "refused": bool(should_refuse),
                            }
                        )
                    )
                    st.success("Feedback recorded.")
                except Exception as ex:
                    logger.warning("Feedback logging failed | error=%s", ex)

    # ========== TAB 2: Eval Dashboard ==========
    with tab_eval_dashboard:
        st.subheader("Evaluation Telemetry & Analytics")
        
        if not ENABLE_EVAL_DASHBOARD:
            st.info("Eval dashboard is disabled. Enable ENABLE_EVAL_DASHBOARD in config.")
        else:
            try:
                dashboard = EvalDashboard()
                records = dashboard.read_logs()
                
                if not records:
                    st.info("No evaluation records yet. Run queries to populate telemetry.")
                else:
                    # Summary metrics
                    summary = dashboard.get_summary_stats(records)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Queries", summary["total_queries"])
                    with col2:
                        st.metric("Avg Confidence", f"{summary['avg_confidence']:.0%}")
                    with col3:
                        st.metric("Refusal Rate", f"{summary['refusal_rate']:.0%}")
                    with col4:
                        st.metric("Avg Judge Score", f"{summary.get('avg_judge_score', 0.5):.0%}")
                    
                    st.divider()
                    
                    # Detailed metrics by tabs
                    metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs(["Overview", "Failed Queries", "Detailed Logs", "Graph Health"])
                    
                    with metrics_tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Intent Distribution")
                            if summary["intents"]:
                                import pandas as pd
                                intent_df = pd.DataFrame(
                                    list(summary["intents"].items()),
                                    columns=["Intent", "Count"]
                                )
                                st.bar_chart(intent_df.set_index("Intent"))
                            else:
                                st.info("No intent data available")
                        
                        with col2:
                            st.markdown("### Language Distribution")
                            if summary["languages"]:
                                lang_df = pd.DataFrame(
                                    list(summary["languages"].items()),
                                    columns=["Language", "Count"]
                                )
                                st.bar_chart(lang_df.set_index("Language"))
                            else:
                                st.info("No language data available")
                        
                        st.markdown("### Confidence by Intent")
                        if summary["avg_confidence_by_intent"]:
                            import pandas as pd
                            conf_df = pd.DataFrame(
                                list(summary["avg_confidence_by_intent"].items()),
                                columns=["Intent", "Avg Confidence"]
                            )
                            st.bar_chart(conf_df.set_index("Intent"))
                        
                        st.markdown("### Judge Verdict Distribution")
                        if summary["judge_verdict_dist"]:
                            import pandas as pd
                            verdict_df = pd.DataFrame(
                                list(summary["judge_verdict_dist"].items()),
                                columns=["Verdict", "Count"]
                            )
                            st.bar_chart(verdict_df.set_index("Verdict"))
                        
                        st.markdown("### Average Quality Scores")
                        quality_cols = st.columns(3)
                        with quality_cols[0]:
                            st.metric("Faithfulness", f"{summary.get('avg_faithfulness', 0.5):.0%}")
                        with quality_cols[1]:
                            st.metric("Usefulness", f"{summary.get('avg_usefulness', 0.5):.0%}")
                        with quality_cols[2]:
                            st.metric("Grounding", f"{summary.get('avg_grounding_score', 0.5):.0%}")
                    
                    with metrics_tab2:
                        st.markdown("### Failed or Low-Confidence Queries")
                        failed = dashboard.get_failed_queries(records, limit=10)
                        
                        if failed:
                            for i, record in enumerate(failed, 1):
                                with st.expander(
                                    f"[{i}] {record.get('query', 'N/A')[:60]}... "
                                    f"(conf: {record.get('confidence', 0.5):.0%}, "
                                    f"refused: {record.get('refused', False)})"
                                ):
                                    st.write(f"**Query:** {record.get('query', 'N/A')}")
                                    st.write(f"**Intent:** {record.get('intent', 'unknown')}")
                                    st.write(f"**Confidence:** {record.get('confidence', 0.5):.0%}")
                                    st.write(f"**Refused:** {record.get('refused', False)}")
                                    st.write(f"**Refusal Reason:** {record.get('refusal_reason', 'N/A')}")
                                    
                                    if record.get("judge"):
                                        st.write("**Judge Verdict:** " + record["judge"].get("verdict", "N/A"))
                                    if record.get("grounding"):
                                        st.write(f"**Grounding Support:** {record['grounding'].get('support_ratio', 0.5):.0%}")
                        else:
                            st.success("All queries have good confidence!")
                    
                    with metrics_tab3:
                        st.markdown("### All Query Records (Table View)")
                        df = dashboard.to_dataframe(records)
                        st.dataframe(df, use_container_width=True)
                        
                        # Export option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="rag_eval_logs.csv",
                            mime="text/csv"
                        )

                    with metrics_tab4:
                        st.markdown("### Graph Index Quality")
                        graph_payload = dashboard.read_graph_index()
                        graph_stats = dashboard.get_graph_health_stats(graph_payload)

                        if not graph_stats.get("has_graph"):
                            st.info("No graph index found yet. Index documents to build GraphRAG communities.")
                        else:
                            g1, g2, g3, g4 = st.columns(4)
                            with g1:
                                st.metric("Entities", graph_stats.get("entity_count", 0))
                            with g2:
                                st.metric("Relations", graph_stats.get("relation_count", 0))
                            with g3:
                                st.metric("Communities", graph_stats.get("community_count", 0))
                            with g4:
                                st.metric("Avg Community Size", f"{graph_stats.get('avg_community_size', 0.0):.1f}")

                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("### Top Entities")
                                top_entities = graph_stats.get("top_entities", [])
                                if top_entities:
                                    ent_df = pd.DataFrame(top_entities)
                                    st.bar_chart(ent_df.set_index("entity"))
                                else:
                                    st.caption("No entities extracted yet.")

                            with c2:
                                st.markdown("### Top Source Communities")
                                top_sources = graph_stats.get("top_sources", [])
                                if top_sources:
                                    src_df = pd.DataFrame(top_sources)
                                    st.bar_chart(src_df.set_index("source"))
                                else:
                                    st.caption("No community/source stats yet.")

                            with st.expander("Graph Index Raw Sample", expanded=False):
                                st.json(
                                    {
                                        "entities": graph_payload.get("entities", [])[:10],
                                        "relations": graph_payload.get("relations", [])[:10],
                                        "communities": graph_payload.get("communities", [])[:5],
                                    }
                                )
            except Exception as ex:
                st.error(f"Eval dashboard error: {ex}")
                logger.exception("Eval dashboard exception | error=%s", ex)

else:
    st.info("Upload and index documents to begin.")