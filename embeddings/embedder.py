# embedder.py
# embeddings/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import time
from config.settings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MAX_WORDS_PER_CHUNK,
)
from pipeline_logger import get_logger


logger = get_logger("embedder")


class MultilingualEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.max_words_per_chunk = EMBEDDING_MAX_WORDS_PER_CHUNK
        logger.info("Embedder initialized | model=%s", EMBEDDING_MODEL_NAME)
        logger.info(
            "Embedder runtime config | batch_size=%d | max_seq_length=%d | max_words_per_chunk=%d",
            self.batch_size,
            self.model.max_seq_length,
            self.max_words_per_chunk,
        )

    def _prepare_text(self, text: str) -> str:
        words = (text or "").split()
        if len(words) > self.max_words_per_chunk:
            return " ".join(words[: self.max_words_per_chunk])
        return text or ""

    def embed_texts(self, texts):
        """
        Embed a list of texts.
        Returns normalized numpy array of shape (n, dim)
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prepared_texts = [self._prepare_text(text) for text in texts]
        total = len(prepared_texts)
        total_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info("Embedding START | texts=%d | batches=%d", total, total_batches)

        all_embeddings = []
        for batch_idx, start in enumerate(range(0, total, self.batch_size), start=1):
            end = min(start + self.batch_size, total)
            batch_texts = prepared_texts[start:end]

            t0 = time.perf_counter()
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            elapsed = time.perf_counter() - t0
            all_embeddings.append(batch_embeddings)
            logger.info(
                "Embedding batch done | batch=%d/%d | items=%d | seconds=%.2f",
                batch_idx,
                total_batches,
                len(batch_texts),
                elapsed,
            )

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.debug("Embedded texts | count=%d | shape=%s", len(texts), getattr(embeddings, "shape", "unknown"))
        logger.info("Embedding END | total=%d | shape=%s", len(texts), getattr(embeddings, "shape", "unknown"))
        return embeddings

    def embed_query(self, query):
        """
        Embed a single query string.
        Returns normalized numpy vector of shape (dim,)
        """
        query_prepared = self._prepare_text(query)
        embedding = self.model.encode(
            query_prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )
        logger.debug("Embedded query | chars=%d | shape=%s", len(query or ""), getattr(embedding, "shape", "unknown"))
        return embedding