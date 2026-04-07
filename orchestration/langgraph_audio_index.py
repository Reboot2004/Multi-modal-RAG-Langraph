from typing import Callable, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from pipeline_logger import get_logger


logger = get_logger("langgraph_audio_index")


class AudioIndexGraphState(TypedDict, total=False):
    file_items: List[Dict]
    reference_map: Dict[str, str]
    all_chunks: List[Dict]
    debug_records: List[Dict]
    source_counts: Dict[str, int]
    semantic_embeddings_shape: str


class LangGraphAudioIndexOrchestrator:
    STEP_ORDER = [
        "transcribe_files",
        "embed_semantic",
        "finalize_stats",
    ]

    def __init__(self, av_loader, embedder, vector_store):
        self.av_loader = av_loader
        self.embedder = embedder
        self.vector_store = vector_store
        self.progress_callback = None
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AudioIndexGraphState)

        graph.add_node("transcribe_files", self._transcribe_files)
        graph.add_node("embed_semantic", self._embed_semantic)
        graph.add_node("finalize_stats", self._finalize_stats)

        graph.set_entry_point("transcribe_files")
        graph.add_edge("transcribe_files", "embed_semantic")
        graph.add_edge("embed_semantic", "finalize_stats")
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

    def index_audio_video(
        self,
        file_items: List[Dict],
        reference_map: Dict[str, str],
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> Dict:
        self.progress_callback = progress_callback

        try:
            final_state = self.graph.invoke(
                {
                    "file_items": file_items,
                    "reference_map": reference_map,
                }
            )

            all_chunks = final_state.get("all_chunks", [])
            return {
                "all_chunks": all_chunks,
                "indexed_chunks": len(all_chunks),
                "source_counts": final_state.get("source_counts", {}),
                "debug_records": final_state.get("debug_records", []),
                "semantic_embeddings_shape": final_state.get("semantic_embeddings_shape", "unknown"),
            }
        finally:
            self.progress_callback = None

    def _transcribe_files(self, state: AudioIndexGraphState) -> AudioIndexGraphState:
        self._emit_progress("transcribe_files")

        file_items = state.get("file_items", [])
        reference_map = state.get("reference_map", {})

        all_chunks = []
        debug_records = []

        for item in file_items:
            file_path = item.get("path")
            source_name = item.get("source")
            reference_text = reference_map.get(source_name, "")

            parsed = self.av_loader.parse(
                file_path=file_path,
                source_name=source_name,
                reference_text=reference_text,
            )

            chunks = parsed.get("chunks", [])
            debug = parsed.get("debug", {})

            all_chunks.extend(chunks)
            debug_records.append(debug)
            logger.info("Transcribed %s | chunks=%d", source_name, len(chunks))

        return {
            "all_chunks": all_chunks,
            "debug_records": debug_records,
        }

    def _embed_semantic(self, state: AudioIndexGraphState) -> AudioIndexGraphState:
        self._emit_progress("embed_semantic")

        all_chunks = state.get("all_chunks", [])
        if not all_chunks:
            return {"semantic_embeddings_shape": "(0, 0)"}

        texts = [item.get("text", "") for item in all_chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_embeddings(embeddings, all_chunks)

        return {
            "semantic_embeddings_shape": str(getattr(embeddings, "shape", "unknown")),
        }

    def _finalize_stats(self, state: AudioIndexGraphState) -> AudioIndexGraphState:
        self._emit_progress("finalize_stats")

        source_counts = {}
        for chunk in state.get("all_chunks", []):
            source = chunk.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return {"source_counts": source_counts}
