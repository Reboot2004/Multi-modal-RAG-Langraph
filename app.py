# app.py
# app.py

import streamlit as st
import os
import tempfile
import re
import html
import gc

from ingestion.loader import DocumentLoader
from embeddings.embedder import MultilingualEmbedder
from embeddings.vector_store import VectorStore
from llm.prompt_builder import PromptBuilder
from llm.client_factory import build_llm_client
from llm.hype_generator import HyPEGenerator
from orchestration.langgraph_query import LangGraphQueryOrchestrator
from orchestration.tier3_agentic_rag import Tier3AgenticRAG
from orchestration.langgraph_index import LangGraphIndexOrchestrator
from orchestration.langgraph_audio_index import LangGraphAudioIndexOrchestrator
from utils.conversation_memory import ConversationMemory
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
    if not hasattr(st.session_state.vector_store, "clear"):
        logger.warning("Session vector_store did not expose clear(); reinitializing instance")
        st.session_state.vector_store = VectorStore()

    st.session_state.vector_store.clear()
    st.session_state.documents_indexed = False
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
            st.success(f"{button_label} completed successfully.")
            logger.info("Indexing completed successfully | mode=%s", st.session_state.ingestion_mode)
        else:
            st.warning("No extractable content found.")
            logger.warning("Indexing produced zero chunks | mode=%s", st.session_state.ingestion_mode)


# -------------------------
# Question Section
# -------------------------

if st.session_state.documents_indexed:

    st.divider()
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

    query = st.text_input("Enter your question")

    if query and st.button("Get Answer"):
        logger.info("Get Answer clicked | query=%s", query)

        prompt_builder = PromptBuilder()
        try:
            llm_client = _build_active_llm_client()
        except Exception as ex:
            st.error(f"Unable to initialize selected LLM provider: {ex}")
            logger.exception("LLM client initialization failed | error=%s", ex)
            st.stop()

        if st.session_state.query_orchestrator is None:
            with st.spinner("Initializing LangGraph retrieval orchestrator..."):
                st.session_state.query_orchestrator = LangGraphQueryOrchestrator(llm_client=llm_client)
            logger.info("LangGraph query orchestrator initialized")

        with st.spinner("Retrieving relevant context..."):
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

            retrieval_output = st.session_state.query_orchestrator.retrieve(
                query,
                top_k=5,
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

            answer = llm_client.generate(
                messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )

            if requested_mcq_count > 0:
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

        # Tier 3: Agentic answer refinement loop (faithfulness/usefulness guided)
        tier3_meta = {"enabled": False, "refined": False, "rounds_used": 0}
        if ENABLE_TIER3_AGENTIC_RAG:
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

        st.subheader("Answer")
        
        # Display confidence badge and contextual retrieval info
        badge = confidence_scores.get("confidence_badge", "🟡")
        level = confidence_scores.get("confidence_level", "MEDIUM")
        confidence = confidence_scores.get("overall_confidence", 0.5)
        
        info_parts = [
            f"{badge} **Confidence: {level.title()}** ({confidence:.0%}) | "
            f"Faithfulness: {confidence_scores.get('faithfulness_score', 0.5):.0%} | "
            f"Usefulness: {confidence_scores.get('usefulness_score', 0.5):.0%}"
        ]
        
        # Show if query was contextualized from history
        if retrieval_output.get("query_contextualized"):
            original = retrieval_output.get("original_query", query)
            info_parts.append(f"**Tip:** Query was contextualized from history")
        
        st.caption(" | ".join(info_parts))
        
        # Hard refusal check
        should_refuse, refusal_reason = st.session_state.query_orchestrator.self_rag_gates.should_refuse_answer(confidence)
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
        else:
            st.write(answer)
            
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

else:
    st.info("Upload and index documents to begin.")