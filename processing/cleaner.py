# cleaner.py
# processing/cleaner.py

import re
import unicodedata


class TextCleaner:
    def __init__(self):
        pass

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters (important for Indic scripts).
        """
        return unicodedata.normalize("NFKC", text)

    def remove_control_characters(self, text: str) -> str:
        """
        Remove non-printable control characters.
        """
        return re.sub(r"[\x00-\x1F\x7F]", "", text)

    def fix_line_breaks(self, text: str) -> str:
        """
        Merge broken lines while preserving paragraph structure.
        """
        # Replace multiple newlines with double newline (paragraph)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Replace single newlines inside paragraphs with space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        return text

    def remove_extra_spaces(self, text: str) -> str:
        """
        Remove excessive whitespace.
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        return text.strip()

    def clean(self, text: str) -> str:
        """
        Full cleaning pipeline.
        """
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.fix_line_breaks(text)
        text = self.remove_extra_spaces(text)
        return text