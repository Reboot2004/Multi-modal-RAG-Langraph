# chunker.py
# processing/chunker.py

from typing import List, Dict, Tuple
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

    def _is_heading_line(self, line: str) -> bool:
        stripped = (line or "").strip()
        if not stripped:
            return False

        if stripped.startswith("#"):
            return True

        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&/-]*", stripped)
        if not (2 <= len(words) <= 12):
            return False

        alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
        if not alpha_words:
            return False

        caps_ratio = sum(1 for word in alpha_words if word[:1].isupper()) / max(1, len(alpha_words))
        is_title_length = len(stripped) <= 120 and not stripped.endswith((".", "!", "?"))
        return caps_ratio >= 0.75 and is_title_length

    def _is_bullet_line(self, line: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line or ""))

    def _is_speaker_line(self, line: str) -> bool:
        stripped = (line or "").strip()
        if len(stripped) < 4:
            return False

        return bool(
            re.match(
                r"^[A-Z][A-Za-z0-9 .,'\-/&]{1,40}(?:[:\-]\s+|\s*$)",
                stripped,
            )
        )

    def _split_structure_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text by structural markers first so chunk boundaries preserve
        section starts, bullet lists, and speaker turns.
        """
        if not text or not text.strip():
            return []

        lines = [line.rstrip() for line in text.splitlines()]
        blocks: List[Tuple[str, str]] = []
        current_lines: List[str] = []
        current_type = "paragraph"

        def flush_current():
            nonlocal current_lines, current_type
            if not current_lines:
                return
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append((block_text, current_type))
            current_lines = []
            current_type = "paragraph"

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                flush_current()
                continue

            if self._is_heading_line(line):
                flush_current()
                current_lines = [line]
                current_type = "heading"
                continue

            if self._is_speaker_line(line):
                flush_current()
                current_lines = [line]
                current_type = "speaker"
                continue

            if self._is_bullet_line(line):
                if current_type not in {"list", "bullet"} and current_lines:
                    flush_current()
                if not current_lines:
                    current_type = "list"
                current_lines.append(line)
                current_type = "list"
                continue

            if not current_lines:
                current_type = "paragraph"

            current_lines.append(line)

        flush_current()
        return blocks

    def _build_chunk_record(self, text: str, source: str, page: int, language: str, chunk_type: str) -> Dict:
        record = {
            "text": text,
            "source": source,
            "page": page,
            "language": language,
        }
        if chunk_type:
            record["chunk_type"] = chunk_type
        return record

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

        paragraphs = self._split_structure_blocks(text)

        chunks = []
        current_chunk = ""
        current_length = 0
        current_type = ""

        for para, para_type in paragraphs:
            para_length = self._approx_token_length(para)

            # If paragraph itself is too large, split it directly
            if para_length > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(
                        self._build_chunk_record(
                            text=current_chunk.strip(),
                            source=source,
                            page=page,
                            language=language,
                            chunk_type=current_type or "mixed",
                        )
                    )
                    current_chunk = ""
                    current_length = 0
                    current_type = ""

                words = para.split()
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, len(words), step):
                    split_chunk = " ".join(words[i:i + self.chunk_size])
                    if not split_chunk.strip():
                        continue
                    chunks.append(
                        self._build_chunk_record(
                            text=split_chunk,
                            source=source,
                            page=page,
                            language=language,
                            chunk_type=para_type,
                        )
                    )
                continue

            # If adding paragraph exceeds chunk size, finalize current chunk
            if current_length + para_length > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        self._build_chunk_record(
                            text=current_chunk.strip(),
                            source=source,
                            page=page,
                            language=language,
                            chunk_type=current_type or para_type,
                        )
                    )

                    # Apply overlap
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = " ".join(overlap_words)
                    current_length = len(overlap_words)
                    current_type = "mixed" if overlap_words else ""

            # Add paragraph to current chunk
            current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
            current_length += para_length
            if not current_type:
                current_type = para_type
            elif current_type != para_type:
                current_type = "mixed"

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                self._build_chunk_record(
                    text=current_chunk.strip(),
                    source=source,
                    page=page,
                    language=language,
                    chunk_type=current_type or "mixed",
                )
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