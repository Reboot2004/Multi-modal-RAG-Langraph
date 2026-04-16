import re
from typing import Dict, List, Optional

import pandas as pd

from pipeline_logger import get_logger
from processing.chunker import TextChunker


logger = get_logger("element_chunker")


class ElementChunker:
    """
    Element-aware chunking facade that routes content through modality-specific
    strategies (markdown, table, OCR image, generic text).
    """

    def __init__(self):
        self.text_chunker = TextChunker()

    def chunk_generic(
        self,
        text: str,
        source: str,
        page: int,
        language: str,
        element_type: str = "text",
        section_hint: str = "",
    ) -> List[Dict]:
        chunks = self.text_chunker.chunk_text(
            text=text,
            source=source,
            page=page,
            language=language,
        )
        for idx, chunk in enumerate(chunks):
            meta = self._build_element_metadata(
                source=source,
                page=page,
                element_type=element_type,
                section_hint=section_hint or chunk.get("section_hint", ""),
                child_index=idx,
                parent_id=chunk.get("parent_id"),
            )
            chunk.update(meta)
        return chunks

    def chunk_markdown(
        self,
        text: str,
        source: str,
        page: int,
        language: str,
    ) -> List[Dict]:
        elements = self._parse_markdown_elements(text)
        all_chunks: List[Dict] = []

        for element in elements:
            element_type = element.get("element_type", "markdown")
            content = element.get("content", "")
            section_hint = element.get("section_hint", "")
            if not content.strip():
                continue

            chunked = self.chunk_generic(
                text=content,
                source=source,
                page=page,
                language=language,
                element_type=element_type,
                section_hint=section_hint,
            )
            all_chunks.extend(chunked)

        logger.info(
            "Element chunking (markdown) completed | source=%s | elements=%d | chunks=%d",
            source,
            len(elements),
            len(all_chunks),
        )
        return all_chunks

    def chunk_table_dataframe(
        self,
        df: pd.DataFrame,
        source: str,
        page: int,
        language: str,
        table_id: str,
        section_hint: str = "",
        rows_per_chunk: int = 20,
    ) -> List[Dict]:
        if df is None or df.empty:
            return []

        normalized = df.fillna("")
        headers = [str(col).strip() for col in normalized.columns]
        header_line = " | ".join(headers)

        parent_id = self._compose_parent_id(
            source=source,
            page=page,
            element_type="table",
            section_hint=section_hint or table_id,
        )

        chunks: List[Dict] = []

        schema_chunk = {
            "text": f"Table schema ({table_id}): {header_line}",
            "source": source,
            "page": page,
            "language": language,
            "chunk_type": "table_schema",
            "section_hint": section_hint or table_id,
            "element_type": "table",
            "table_id": table_id,
            "parent_id": parent_id,
            "child_index": 0,
            "child_level": "child",
        }
        chunks.append(schema_chunk)

        row_texts = []
        for _, row in normalized.iterrows():
            parts = [f"{col}: {row[col]}" for col in normalized.columns]
            row_texts.append(" | ".join(parts))

        for offset in range(0, len(row_texts), rows_per_chunk):
            subset = row_texts[offset:offset + rows_per_chunk]
            body = "\n".join(subset)
            text = (
                f"Table {table_id} rows {offset + 1}-{offset + len(subset)}\n"
                f"Headers: {header_line}\n"
                f"Rows:\n{body}"
            )
            child_index = 1 + (offset // rows_per_chunk)
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "page": page,
                    "language": language,
                    "chunk_type": "table_rows",
                    "section_hint": section_hint or table_id,
                    "element_type": "table",
                    "table_id": table_id,
                    "row_start": offset + 1,
                    "row_end": offset + len(subset),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "child_level": "child",
                }
            )

        return chunks

    def chunk_image_ocr(
        self,
        text: str,
        source: str,
        page: int,
        language: str,
        caption_hint: Optional[str] = None,
    ) -> List[Dict]:
        section_hint = caption_hint or "image_ocr"
        base_chunks = self.chunk_generic(
            text=text,
            source=source,
            page=page,
            language=language,
            element_type="image_ocr",
            section_hint=section_hint,
        )

        if not base_chunks:
            return []

        summary_text = self._build_image_summary(text=text, caption_hint=caption_hint)
        parent_id = self._compose_parent_id(
            source=source,
            page=page,
            element_type="image_ocr",
            section_hint=section_hint,
        )

        summary_chunk = {
            "text": summary_text,
            "source": source,
            "page": page,
            "language": language,
            "chunk_type": "image_summary",
            "section_hint": section_hint,
            "element_type": "image_ocr",
            "parent_id": parent_id,
            "child_index": 0,
            "child_level": "parent",
        }

        for idx, chunk in enumerate(base_chunks, start=1):
            chunk["parent_id"] = parent_id
            chunk["child_index"] = idx
            chunk["child_level"] = "child"

        return [summary_chunk] + base_chunks

    def _parse_markdown_elements(self, text: str) -> List[Dict]:
        lines = (text or "").splitlines()
        elements: List[Dict] = []

        section_stack: List[str] = []
        current_lines: List[str] = []
        current_type = "markdown_paragraph"
        in_code_block = False

        def flush_current():
            nonlocal current_lines, current_type
            if not current_lines:
                return
            content = "\n".join(current_lines).strip()
            if content:
                section_hint = " > ".join(section_stack[-3:]) if section_stack else ""
                elements.append(
                    {
                        "element_type": current_type,
                        "content": content,
                        "section_hint": section_hint,
                    }
                )
            current_lines = []
            current_type = "markdown_paragraph"

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    current_lines.append(line)
                    flush_current()
                    in_code_block = False
                else:
                    flush_current()
                    in_code_block = True
                    current_type = "code_block"
                    current_lines.append(line)
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
            if heading:
                flush_current()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section_stack[:] = section_stack[: max(0, level - 1)]
                section_stack.append(title)
                elements.append(
                    {
                        "element_type": "heading",
                        "content": title,
                        "section_hint": " > ".join(section_stack[-3:]),
                    }
                )
                continue

            if stripped.startswith("|") and stripped.endswith("|"):
                if current_type != "markdown_table":
                    flush_current()
                    current_type = "markdown_table"
                current_lines.append(line)
                continue

            if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
                if current_type != "markdown_list":
                    flush_current()
                    current_type = "markdown_list"
                current_lines.append(line)
                continue

            if not stripped:
                flush_current()
                continue

            if current_type not in {"markdown_paragraph", "markdown_quote"}:
                flush_current()
                current_type = "markdown_paragraph"
            current_lines.append(line)

        flush_current()
        return elements

    def _build_image_summary(self, text: str, caption_hint: Optional[str]) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        preview = normalized[:500]
        if caption_hint:
            return f"Image content summary ({caption_hint}): {preview}"
        return f"Image content summary: {preview}"

    def _compose_parent_id(self, source: str, page: int, element_type: str, section_hint: str) -> str:
        safe_section = re.sub(r"[^A-Za-z0-9_\-]+", "_", (section_hint or "")).strip("_")
        if safe_section:
            return f"{source}::p{page}::{element_type}::{safe_section}"
        return f"{source}::p{page}::{element_type}"

    def _build_element_metadata(
        self,
        source: str,
        page: int,
        element_type: str,
        section_hint: str,
        child_index: int,
        parent_id: Optional[str] = None,
    ) -> Dict:
        parent = parent_id or self._compose_parent_id(
            source=source,
            page=page,
            element_type=element_type,
            section_hint=section_hint,
        )
        payload = {
            "element_type": element_type,
            "parent_id": parent,
            "child_index": child_index,
            "child_level": "child",
        }
        if section_hint:
            payload["section_hint"] = section_hint
        return payload
