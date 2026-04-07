# ingestion/loader.py

import os
from typing import List

from ingestion.pdf_parser import PDFParser
from ingestion.image_loader import ImageLoader
from ingestion.structured_loader import StructuredFileParser
from processing.cleaner import TextCleaner
from processing.language_detector import LanguageDetector
from processing.chunker import TextChunker
from pipeline_logger import get_logger


logger = get_logger("loader")


class DocumentLoader:
    TEXT_LIKE_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".py",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".hxx",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".css",
        ".scss",
        ".sass",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".sql",
        ".xml",
        ".yaml",
        ".yml",
        ".ini",
        ".toml",
    }

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.image_loader = ImageLoader()
        self.structured_parser = StructuredFileParser()
        self.cleaner = TextCleaner()
        self.lang_detector = LanguageDetector()
        self.chunker = TextChunker()

    def load(self, file_path: str, source_name: str = None) -> List[dict]:
        """
        Route file to correct parser based on extension.
        Returns list of chunk dictionaries.
        """

        extension = os.path.splitext(file_path)[1].lower()
        logger.info("Load called | source=%s | extension=%s", source_name or file_path, extension)

        if extension == ".pdf":
            return self.pdf_parser.parse(file_path, source_name=source_name)

        elif extension in [".jpg", ".jpeg", ".png"]:
            return self.image_loader.parse(file_path, source_name=source_name)

        elif extension in self.TEXT_LIKE_EXTENSIONS:
            return self.structured_parser.parse_text_like(file_path, source_name=source_name)

        elif extension == ".docx":
            return self.structured_parser.parse_docx(file_path, source_name=source_name)

        elif extension in [".csv"]:
            return self.structured_parser.parse_csv(file_path, source_name=source_name)

        elif extension in [".xlsx", ".xls"]:
            return self.structured_parser.parse_excel(file_path, source_name=source_name)

        elif extension in [".html", ".htm"]:
            return self.structured_parser.parse_html(file_path, source_name=source_name)

        elif extension in [".json"]:
            return self.structured_parser.parse_json(file_path, source_name=source_name)

        elif extension in [".pptx"]:
            return self.structured_parser.parse_pptx(file_path, source_name=source_name)

        elif extension in [".ppt"]:
            return self.structured_parser.parse_ppt(file_path, source_name=source_name)

        else:
            raise ValueError(f"Unsupported file type: {extension}")