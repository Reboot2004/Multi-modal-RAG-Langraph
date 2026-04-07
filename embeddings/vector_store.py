# vector_store.py
# embeddings/vector_store.py

import os
import json
import faiss
import numpy as np
from embeddings.lexical_store import LexicalStore
from config.settings import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_DIR,
    TOP_K,
)
from pipeline_logger import get_logger


logger = get_logger("vector_store")


class VectorStore:
    def __init__(self):
        self.index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
        self.metadata_path = os.path.join(FAISS_INDEX_DIR, "metadata.json")
        self.hype_index_path = os.path.join(FAISS_INDEX_DIR, "hype_index.faiss")
        self.hype_metadata_path = os.path.join(FAISS_INDEX_DIR, "hype_metadata.json")

        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

        self.index = None
        self.metadata = []
        self.hype_index = None
        self.hype_metadata = []

        self._load_or_initialize()
        self.lexical_store = LexicalStore()
        logger.info("VectorStore initialized | semantic=%d | hype=%d", self.index.ntotal, self.hype_index.ntotal)

    def _load_or_initialize(self):
        metadata_updated = False

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                metadata_updated = self._ensure_chunk_ids()
        else:
            # Using Inner Product since embeddings are normalized
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []

        if os.path.exists(self.hype_index_path):
            self.hype_index = faiss.read_index(self.hype_index_path)

            if os.path.exists(self.hype_metadata_path):
                with open(self.hype_metadata_path, "r", encoding="utf-8") as f:
                    self.hype_metadata = json.load(f)
        else:
            self.hype_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.hype_metadata = []

        if metadata_updated:
            self._save()
        logger.debug(
            "VectorStore loaded | semantic_docs=%d | hype_docs=%d | metadata=%d | hype_metadata=%d",
            self.index.ntotal,
            self.hype_index.ntotal,
            len(self.metadata),
            len(self.hype_metadata),
        )

    def add_embeddings(self, embeddings: np.ndarray, metadatas: list):
        """
        embeddings: numpy array (n, dim)
        metadatas: list of metadata dicts (length n)
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        start_chunk_id = self._next_chunk_id()
        normalized_meta = []

        for idx, meta in enumerate(metadatas):
            item = dict(meta)
            if "chunk_id" not in item:
                item["chunk_id"] = start_chunk_id + idx
                if isinstance(meta, dict):
                    meta["chunk_id"] = item["chunk_id"]
            normalized_meta.append(item)

        self.index.add(embeddings)
        self.metadata.extend(normalized_meta)
        try:
            self.lexical_store.upsert_chunks(normalized_meta)
        except Exception as ex:
            logger.warning("Lexical upsert failed; continuing with semantic index only | error=%s", ex)
        logger.info("Added semantic embeddings | added=%d | total=%d", len(normalized_meta), self.index.ntotal)

        self._save()

    def add_hype_embeddings(self, embeddings: np.ndarray, hype_metadatas: list):
        """
        embeddings: numpy array (n, dim)
        hype_metadatas: list of dicts containing hypothetical prompt entries
        """
        if len(embeddings) != len(hype_metadatas):
            raise ValueError("HyPE embeddings and metadata length mismatch")

        self.hype_index.add(embeddings)
        self.hype_metadata.extend(hype_metadatas)
        logger.info("Added HyPE embeddings | added=%d | total=%d", len(hype_metadatas), self.hype_index.ntotal)

        self._save()

    def clear(self):
        """
        Reset both semantic and HyPE indexes and remove persisted files.
        """
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata = []

        self.hype_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.hype_metadata = []

        for path in [
            self.index_path,
            self.metadata_path,
            self.hype_index_path,
            self.hype_metadata_path,
        ]:
            if os.path.exists(path):
                os.remove(path)

        self._save()
        logger.warning("VectorStore cleared (semantic + HyPE)")
        self.lexical_store.clear()

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K):
        """
        Returns top_k results:
        [
            {
                "score": float,
                "text": str,
                "metadata": {...}
            }
        ]
        """
        query_embedding = np.array([query_embedding])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(
                    {
                        "score": float(score),
                        "text": meta["text"],
                        "metadata": meta,
                    }
                )

        logger.debug("Semantic search completed | requested=%d | returned=%d", top_k, len(results))
        return results

    def search_hype(self, query_embedding: np.ndarray, top_k: int = TOP_K):
        """
        Search hypothetical prompt index, then map results to original chunks.
        """
        if self.hype_index is None or self.hype_index.ntotal == 0:
            return []

        query_embedding = np.array([query_embedding])
        scores, indices = self.hype_index.search(query_embedding, top_k)

        chunk_lookup = self._chunk_lookup_by_id()
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.hype_metadata):
                continue

            hype_meta = self.hype_metadata[idx]
            chunk_id = hype_meta.get("chunk_id")
            chunk_meta = chunk_lookup.get(chunk_id)

            if not chunk_meta:
                continue

            results.append(
                {
                    "score": float(score),
                    "text": chunk_meta.get("text", ""),
                    "metadata": chunk_meta,
                    "hype_prompt": hype_meta.get("text", ""),
                }
            )

        logger.debug("HyPE search completed | requested=%d | returned=%d", top_k, len(results))
        return results

    def search_lexical(self, query: str, top_k: int = TOP_K):
        try:
            return self.lexical_store.search(query=query, top_k=top_k)
        except Exception as ex:
            logger.warning("Lexical search failed; returning no lexical results | error=%s", ex)
            return []

    def _next_chunk_id(self):
        max_chunk_id = -1

        for item in self.metadata:
            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, int) and chunk_id > max_chunk_id:
                max_chunk_id = chunk_id

        return max_chunk_id + 1

    def _chunk_lookup_by_id(self):
        lookup = {}

        for idx, item in enumerate(self.metadata):
            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, int):
                chunk_id = idx
                item["chunk_id"] = chunk_id
            lookup[chunk_id] = item

        return lookup

    def _ensure_chunk_ids(self):
        updated = False

        for idx, item in enumerate(self.metadata):
            if not isinstance(item.get("chunk_id"), int):
                item["chunk_id"] = idx
                updated = True

        return updated

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        faiss.write_index(self.hype_index, self.hype_index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        with open(self.hype_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.hype_metadata, f, ensure_ascii=False, indent=2)

        logger.debug(
            "VectorStore persisted | semantic_docs=%d | hype_docs=%d | metadata=%d | hype_metadata=%d",
            self.index.ntotal,
            self.hype_index.ntotal,
            len(self.metadata),
            len(self.hype_metadata),
        )