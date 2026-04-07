from typing import Callable, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from config.settings import ENABLE_HYPE
from config.settings import HYPE_MAX_CHUNKS_PER_DOCUMENT
from pipeline_logger import get_logger


logger = get_logger("langgraph_index")


class IndexGraphState(TypedDict, total=False):
    file_items: List[Dict]
    all_chunks: List[Dict]
    semantic_embeddings_shape: str
    hype_entries: List[Dict]
    hype_count: int
    source_counts: Dict[str, int]


class LangGraphIndexOrchestrator:
    STEP_ORDER = [
        "load_chunks",
        "embed_semantic",
        "generate_hype_entries",
        "embed_hype",
        "finalize_stats",
    ]

    def __init__(self, loader, embedder, vector_store, hype_generator=None):
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.hype_generator = hype_generator
        self.enable_hype = ENABLE_HYPE and hype_generator is not None
        self.progress_callback = None
        self.graph = self._build_graph()
        logger.info(
            "LangGraphIndexOrchestrator initialized | hype_enabled=%s",
            self.enable_hype,
        )

    def _build_graph(self):
        graph = StateGraph(IndexGraphState)

        graph.add_node("load_chunks", self._load_chunks)
        graph.add_node("embed_semantic", self._embed_semantic)
        graph.add_node("generate_hype_entries", self._generate_hype_entries)
        graph.add_node("embed_hype", self._embed_hype)
        graph.add_node("finalize_stats", self._finalize_stats)

        graph.set_entry_point("load_chunks")
        graph.add_edge("load_chunks", "embed_semantic")
        graph.add_edge("embed_semantic", "generate_hype_entries")
        graph.add_edge("generate_hype_entries", "embed_hype")
        graph.add_edge("embed_hype", "finalize_stats")
        graph.add_edge("finalize_stats", END)

        return graph.compile()

    def _emit_progress(self, step_name: str):
        if self.progress_callback is None:
            return

        try:
            position = self.STEP_ORDER.index(step_name) + 1
            total = len(self.STEP_ORDER)
            self.progress_callback(step_name, position, total)
        except Exception:
            logger.debug("Progress callback failed for step=%s", step_name)

    def index_documents(self, file_items: List[Dict], progress_callback: Callable[[str, int, int], None] = None) -> Dict:
        self.progress_callback = progress_callback

        try:
            final_state = self.graph.invoke({"file_items": file_items})
            return {
                "all_chunks": final_state.get("all_chunks", []),
                "indexed_chunks": len(final_state.get("all_chunks", [])),
                "hype_count": int(final_state.get("hype_count", 0)),
                "source_counts": final_state.get("source_counts", {}),
                "semantic_embeddings_shape": final_state.get("semantic_embeddings_shape", "unknown"),
            }
        finally:
            self.progress_callback = None

    def _load_chunks(self, state: IndexGraphState) -> IndexGraphState:
        self._emit_progress("load_chunks")

        file_items = state.get("file_items", [])
        all_chunks = []

        for item in file_items:
            file_path = item.get("path")
            source_name = item.get("source")
            chunks = self.loader.load(file_path, source_name=source_name)
            logger.info("Loaded %s | chunks=%d", source_name or file_path, len(chunks))
            all_chunks.extend(chunks)

        logger.info("LangGraph index load_chunks completed | total_chunks=%d", len(all_chunks))
        return {"all_chunks": all_chunks}

    def _embed_semantic(self, state: IndexGraphState) -> IndexGraphState:
        self._emit_progress("embed_semantic")

        all_chunks = state.get("all_chunks", [])
        if not all_chunks:
            logger.warning("No chunks found for semantic embedding")
            return {"semantic_embeddings_shape": "(0, 0)"}

        texts = [chunk.get("text", "") for chunk in all_chunks]
        semantic_embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_embeddings(semantic_embeddings, all_chunks)

        shape_text = str(getattr(semantic_embeddings, "shape", "unknown"))
        logger.info("Semantic embeddings stored | shape=%s", shape_text)
        return {"semantic_embeddings_shape": shape_text}

    def _generate_hype_entries(self, state: IndexGraphState) -> IndexGraphState:
        self._emit_progress("generate_hype_entries")

        if not self.enable_hype:
            return {"hype_entries": []}

        all_chunks = state.get("all_chunks", [])
        hype_entries = []

        selected_chunks = self._select_hype_chunks(all_chunks)
        logger.info(
            "HyPE chunk selection | total_chunks=%d | selected=%d | cap=%d",
            len(all_chunks),
            len(selected_chunks),
            int(HYPE_MAX_CHUNKS_PER_DOCUMENT),
        )

        for idx, chunk in enumerate(selected_chunks, start=1):
            chunk_id = chunk.get("chunk_id")
            if chunk_id is None:
                continue

            chunk_language = chunk.get("language", "en")
            prompts = self.hype_generator.generate_prompts_for_chunk(
                chunk_text=chunk.get("text", ""),
                language=chunk_language,
            )

            for prompt in prompts:
                hype_entries.append(
                    {
                        "text": prompt,
                        "chunk_id": chunk_id,
                        "source": chunk.get("source"),
                        "page": chunk.get("page"),
                        "language": chunk_language,
                    }
                )

            if idx % 5 == 0:
                logger.debug("HyPE generation progress | processed=%d/%d", idx, len(selected_chunks))

        logger.info("HyPE prompt generation completed | prompts=%d", len(hype_entries))
        return {"hype_entries": hype_entries}

    def _select_hype_chunks(self, all_chunks: List[Dict]) -> List[Dict]:
        if not all_chunks:
            return []

        cap = max(1, int(HYPE_MAX_CHUNKS_PER_DOCUMENT))
        if len(all_chunks) <= cap:
            return all_chunks

        # Evenly sample chunks across the document to preserve coverage.
        last_index = len(all_chunks) - 1
        picked = []
        used = set()
        for i in range(cap):
            index = round(i * last_index / max(1, cap - 1))
            if index in used:
                continue
            used.add(index)
            picked.append(all_chunks[index])

        return picked

    def _embed_hype(self, state: IndexGraphState) -> IndexGraphState:
        self._emit_progress("embed_hype")

        hype_entries = state.get("hype_entries", [])
        if not hype_entries:
            return {"hype_count": 0}

        hype_texts = [item.get("text", "") for item in hype_entries]
        hype_embeddings = self.embedder.embed_texts(hype_texts)
        self.vector_store.add_hype_embeddings(hype_embeddings, hype_entries)

        logger.info("HyPE embeddings stored | count=%d", len(hype_entries))
        return {"hype_count": len(hype_entries)}

    def _finalize_stats(self, state: IndexGraphState) -> IndexGraphState:
        self._emit_progress("finalize_stats")

        all_chunks = state.get("all_chunks", [])
        source_counts = {}

        for chunk in all_chunks:
            source = chunk.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return {"source_counts": source_counts}
