# chunker.py
# processing/chunker.py

from typing import List, Dict
import re
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKER_BACKEND
from pipeline_logger import get_logger


logger = get_logger("chunker")


class TextChunker:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.chunker_backend = (CHUNKER_BACKEND or "legacy").strip().lower()
        self._chonkie_chunker = self._build_chonkie_chunker() if self.chunker_backend == "chonkie" else None
        self._chonkie_config = (int(self.chunk_size), int(self.chunk_overlap))
        logger.info("TextChunker initialized | chunk_size=%d | overlap=%d", self.chunk_size, self.chunk_overlap)
        logger.info("TextChunker backend=%s", self.chunker_backend)

    def _build_chonkie_chunker(self):
        try:
            from chonkie import RecursiveChunker

            # Chonkie constructor args differ across versions; try compatible signatures.
            builders = [
                (RecursiveChunker, {"chunk_size": int(self.chunk_size), "chunk_overlap": int(self.chunk_overlap)}, "RecursiveChunker(chunk_size, chunk_overlap)"),
                (RecursiveChunker, {"chunk_size": int(self.chunk_size), "overlap": int(self.chunk_overlap)}, "RecursiveChunker(chunk_size, overlap)"),
                (RecursiveChunker, {"chunk_size": int(self.chunk_size)}, "RecursiveChunker(chunk_size)"),
            ]

            try:
                from chonkie import TokenChunker

                builders.extend(
                    [
                        (TokenChunker, {"chunk_size": int(self.chunk_size), "chunk_overlap": int(self.chunk_overlap)}, "TokenChunker(chunk_size, chunk_overlap)"),
                        (TokenChunker, {"chunk_size": int(self.chunk_size), "overlap": int(self.chunk_overlap)}, "TokenChunker(chunk_size, overlap)"),
                        (TokenChunker, {"chunk_size": int(self.chunk_size)}, "TokenChunker(chunk_size)"),
                    ]
                )
            except Exception:
                pass

            last_error = None
            for cls, kwargs, label in builders:
                try:
                    chunker = cls(**kwargs)
                    logger.info("Chonkie chunker initialized via %s", label)
                    return chunker
                except TypeError as ex:
                    last_error = ex
                    continue

            raise RuntimeError(last_error or "No compatible Chonkie constructor signature found")
        except Exception as ex:
            logger.warning("Failed to initialize Chonkie chunker; using legacy chunker | error=%s", ex)
            return None

    def _sync_chonkie_chunker(self):
        if self.chunker_backend != "chonkie":
            return

        current = (int(self.chunk_size), int(self.chunk_overlap))
        if self._chonkie_chunker is None or current != self._chonkie_config:
            self._chonkie_chunker = self._build_chonkie_chunker()
            self._chonkie_config = current

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using blank-line boundaries.
        Falls back to single-line grouping when OCR/text has weak paragraph marks.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

        if len(paragraphs) <= 1:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            paragraphs = []
            buffer = []
            for line in lines:
                buffer.append(line)
                if len(buffer) >= 4:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
            if buffer:
                paragraphs.append(" ".join(buffer))

        if not paragraphs:
            paragraphs = [text.strip()]

        return paragraphs

    def _approx_token_length(self, text: str) -> int:
        """
        Rough token estimation using word count.
        Works reasonably well for multilingual content.
        """
        return len(text.split())

    def chunk_text(
        self,
        text: str,
        source: str,
        page: int,
        language: str,
    ) -> List[Dict]:
        """
        Returns list of chunk dictionaries:
        [
            {
                "text": "...",
                "source": "...",
                "page": 1,
                "language": "hi"
            }
        ]
        """

        if self._chonkie_chunker is not None:
            chonkie_chunks = self._chunk_with_chonkie(text=text, source=source, page=page, language=language)
            if chonkie_chunks:
                logger.debug(
                    "Chunking completed via Chonkie | source=%s | page=%s | language=%s | chunks=%d",
                    source,
                    page,
                    language,
                    len(chonkie_chunks),
                )
                return chonkie_chunks

        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para_length = self._approx_token_length(para)

            # If paragraph itself is too large, split it directly
            if para_length > self.chunk_size:
                words = para.split()
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, len(words), step):
                    split_chunk = " ".join(words[i:i + self.chunk_size])
                    if not split_chunk.strip():
                        continue
                    chunks.append(
                        {
                            "text": split_chunk,
                            "source": source,
                            "page": page,
                            "language": language,
                        }
                    )
                continue

            # If adding paragraph exceeds chunk size, finalize current chunk
            if current_length + para_length > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "source": source,
                            "page": page,
                            "language": language,
                        }
                    )

                    # Apply overlap
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = " ".join(overlap_words)
                    current_length = len(overlap_words)

            # Add paragraph to current chunk
            current_chunk += " " + para
            current_length += para_length

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "source": source,
                    "page": page,
                    "language": language,
                }
            )

        logger.debug(
            "Chunking completed | source=%s | page=%s | language=%s | chunks=%d",
            source,
            page,
            language,
            len(chunks),
        )

        return chunks

    def _chunk_with_chonkie(self, text: str, source: str, page: int, language: str) -> List[Dict]:
        if not text or not text.strip():
            return []

        self._sync_chonkie_chunker()
        if self._chonkie_chunker is None:
            return []

        try:
            raw_chunks = self._chonkie_chunker(text)
        except Exception:
            try:
                raw_chunks = self._chonkie_chunker.chunk(text)
            except Exception as ex:
                logger.warning("Chonkie chunking failed; using legacy chunker | error=%s", ex)
                return []

        normalized = []
        for item in raw_chunks or []:
            if hasattr(item, "text"):
                chunk_text = str(getattr(item, "text") or "").strip()
            else:
                chunk_text = str(item or "").strip()

            if not chunk_text:
                continue

            normalized.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "language": language,
                }
            )

        return normalized