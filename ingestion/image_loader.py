# image_loader.py
# ingestion/image_loader.py

import os
from typing import List

from ingestion.element_chunker import ElementChunker
from ingestion.ocr_engine import OCREngine
from processing.cleaner import TextCleaner
from processing.language_detector import LanguageDetector
from processing.chunker import TextChunker
from config.settings import ENABLE_OCR
from pipeline_logger import get_logger


logger = get_logger("image_loader")


class ImageLoader:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.cleaner = TextCleaner()
        self.lang_detector = LanguageDetector()
        self.chunker = TextChunker()
        self.element_chunker = ElementChunker()

    def parse(self, file_path: str, source_name: str = None) -> List[dict]:
        """
        Parse image file (JPG/PNG).
        Returns list of chunk dictionaries.
        """

        file_name = source_name or os.path.basename(file_path)
        logger.info("Image parse started | source=%s", file_name)

        # Read image as bytes
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # OCR
        if ENABLE_OCR:
            text = self.ocr_engine.extract_text_from_image_bytes(image_bytes)
        else:
            logger.warning("OCR disabled; image text extraction skipped | source=%s", file_name)
            text = ""

        # Clean text
        text = self.cleaner.clean(text)

        if not text.strip():
            logger.info("Image parse produced no text | source=%s", file_name)
            return []

        # Detect language
        language = self.lang_detector.detect_language(text)

        # Chunk text (treat entire image as page 1)
        chunks = self.element_chunker.chunk_image_ocr(
            text=text,
            source=file_name,
            page=1,
            language=language,
            caption_hint=os.path.splitext(file_name)[0],
        )

        logger.info("Image parse completed | source=%s | language=%s | chunks=%d", file_name, language, len(chunks))

        return chunks