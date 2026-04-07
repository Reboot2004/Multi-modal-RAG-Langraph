# pdf_parser.py
# ingestion/pdf_parser.py

import fitz  # PyMuPDF
import os
from typing import List

from ingestion.ocr_engine import OCREngine
from processing.cleaner import TextCleaner
from processing.language_detector import LanguageDetector
from processing.chunker import TextChunker
from config.settings import ENABLE_OCR, PDF_OCR_FALLBACK_MIN_CHARS
from pipeline_logger import get_logger


logger = get_logger("pdf_parser")


class PDFParser:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.cleaner = TextCleaner()
        self.lang_detector = LanguageDetector()
        self.chunker = TextChunker()

    def parse(self, file_path: str, source_name: str = None) -> List[dict]:
        """
        Parse PDF and return list of chunk dictionaries.
        Handles:
        - Text PDFs
        - Scanned PDFs
        - Mixed PDFs
        """

        document = fitz.open(file_path)
        all_chunks = []

        file_name = source_name or os.path.basename(file_path)
        logger.info("PDF parse started | source=%s | pages=%d", file_name, len(document))

        for page_number in range(len(document)):
            page = document.load_page(page_number)

            # Try direct text extraction
            text = page.get_text("text")

            # If very little text, assume scanned page
            extracted_chars = len((text or "").strip())
            used_ocr = False

            if extracted_chars < PDF_OCR_FALLBACK_MIN_CHARS and ENABLE_OCR:
                pix = page.get_pixmap()
                image_bytes = pix.tobytes("png")
                text = self.ocr_engine.extract_text_from_image_bytes(image_bytes)
                used_ocr = True
            elif extracted_chars < PDF_OCR_FALLBACK_MIN_CHARS and not ENABLE_OCR:
                logger.debug(
                    "PDF page under OCR threshold but OCR disabled | source=%s | page=%d | chars=%d",
                    file_name,
                    page_number + 1,
                    extracted_chars,
                )

            # Clean text
            text = self.cleaner.clean(text)

            if not text.strip():
                logger.debug("Empty page after cleaning | source=%s | page=%d", file_name, page_number + 1)
                continue

            # Detect language
            language = self.lang_detector.detect_language(text)

            # Chunk page
            page_chunks = self.chunker.chunk_text(
                text=text,
                source=file_name,
                page=page_number + 1,
                language=language,
            )

            all_chunks.extend(page_chunks)
            logger.debug(
                "PDF page processed | source=%s | page=%d | chars=%d | language=%s | used_ocr=%s | chunks=%d",
                file_name,
                page_number + 1,
                len(text),
                language,
                used_ocr,
                len(page_chunks),
            )

        document.close()
        logger.info("PDF parse completed | source=%s | total_chunks=%d", file_name, len(all_chunks))

        return all_chunks