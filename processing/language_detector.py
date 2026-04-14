# language_detector.py
# processing/language_detector.py

import re
from typing import Dict

from langdetect import detect, DetectorFactory, LangDetectException
from config.settings import SUPPORTED_LANGUAGES

# Make detection deterministic
DetectorFactory.seed = 42


class LanguageDetector:
    def __init__(self):
        self.supported_languages = SUPPORTED_LANGUAGES
        self._language_aliases = {
            "english": "en",
            "hindi": "hi",
            "hinglish": "hi",
            "telugu": "te",
            "tinglish": "te",
            "teluglish": "te",
            "tamil": "ta",
            "tanglish": "ta",
            "kannada": "kn",
            "bengali": "bn",
            "bangla": "bn",
            "malayalam": "ml",
            "marathi": "mr",
            "gujarati": "gu",
            "punjabi": "pa",
            "urdu": "ur",
        }

        self._language_meta = {
            "en": {"name": "English", "script": "Latin"},
            "hi": {"name": "Hindi", "script": "Devanagari"},
            "te": {"name": "Telugu", "script": "Telugu"},
            "ta": {"name": "Tamil", "script": "Tamil"},
            "kn": {"name": "Kannada", "script": "Kannada"},
            "bn": {"name": "Bengali", "script": "Bengali"},
            "ml": {"name": "Malayalam", "script": "Malayalam"},
            "mr": {"name": "Marathi", "script": "Devanagari"},
            "gu": {"name": "Gujarati", "script": "Gujarati"},
            "pa": {"name": "Punjabi", "script": "Gurmukhi"},
            "ur": {"name": "Urdu", "script": "Perso-Arabic"},
        }

        self._roman_hindi_tokens = {
            "hai", "hain", "kya", "kaise", "nahi", "nahin", "kyu", "kyun", "mujhe",
            "tum", "aap", "mera", "meri", "karo", "karna", "mein", "tha", "thi", "haan", "acha",
        }
        self._roman_telugu_tokens = {
            "nenu", "nuvvu", "meeru", "emi", "em", "ela", "enti", "kada", "andi",
            "ledu", "undi", "unnanu", "unna", "cheppu", "chala", "manchi", "avunu",
            "inka", "ippudu", "evaru", "enduku", "matladu", "vinu",
        }

    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.
        Returns ISO 639-1 language code.
        Defaults to 'en' if detection fails or unsupported.
        """

        if not text or len(text.strip()) < 5:
            return "en"

        try:
            lang = detect(text)

            if lang in self.supported_languages:
                return lang
            else:
                return "en"

        except LangDetectException:
            return "en"

    def resolve_response_language(self, text: str, detected_language: str = None) -> Dict[str, str]:
        """
        Resolve output language with native-script preference.

        Priority:
        1. Explicit language mention in prompt (e.g., "answer in Hindi", "in Hinglish")
        2. Transliteration heuristics (e.g., Hinglish/Tinglish written in Latin script)
        3. Baseline language detection fallback
        """

        normalized_text = (text or "").strip()
        lowered = normalized_text.lower()

        explicit = self._extract_explicit_language(lowered)
        if explicit:
            return self._build_resolution(explicit, "explicit_mention")

        transliterated = self._detect_transliterated_language(lowered)
        if transliterated:
            return self._build_resolution(transliterated, "transliterated_input")

        lang = detected_language or self.detect_language(normalized_text)
        if lang not in self.supported_languages:
            lang = "en"

        return self._build_resolution(lang, "detected")

    def _extract_explicit_language(self, lowered_text: str) -> str:
        if not lowered_text:
            return ""

        # Support explicit ISO language code requests, e.g. "answer in te"
        for code in self.supported_languages:
            code_escaped = re.escape(code)
            code_patterns = [
                rf"\b(?:answer|respond|reply|write|explain|output|return)\s+(?:in|using)\s+{code_escaped}\b",
                rf"\bin\s+{code_escaped}\s+(?:language|script)\b",
                rf"\blanguage\s+code\s*[:=]?\s*{code_escaped}\b",
            ]
            for pattern in code_patterns:
                if re.search(pattern, lowered_text):
                    return code

        for alias, code in self._language_aliases.items():
            escaped = re.escape(alias)
            patterns = [
                rf"\b(?:answer|respond|reply|write|explain|output|return)\s+(?:in|using)\s+{escaped}\b",
                rf"\bin\s+{escaped}\s+(?:language|script)\b",
                rf"\b{escaped}\s+(?:language|script)\b",
                rf"\b(?:make|give|tell)\s+it\s+{escaped}\b",
                rf"\b{escaped}\b",
            ]
            for pattern in patterns:
                if re.search(pattern, lowered_text):
                    return code

        return ""

    def _detect_transliterated_language(self, lowered_text: str) -> str:
        if not lowered_text:
            return ""

        latin_tokens = re.findall(r"[a-z]+", lowered_text)
        if len(latin_tokens) < 3:
            return ""

        hindi_hits = sum(1 for token in latin_tokens if token in self._roman_hindi_tokens)
        telugu_hits = sum(1 for token in latin_tokens if token in self._roman_telugu_tokens)

        # Avoid false positives from ambiguous English words (e.g., "main" in "main logic").
        if hindi_hits >= 2 and hindi_hits >= telugu_hits + 1:
            return "hi"
        if telugu_hits >= 2 and telugu_hits >= hindi_hits + 1:
            return "te"

        return ""

    def _build_resolution(self, language_code: str, reason: str) -> Dict[str, str]:
        meta = self._language_meta.get(language_code, {"name": "English", "script": "Latin"})
        return {
            "language_code": language_code,
            "language_name": meta["name"],
            "script": meta["script"],
            "reason": reason,
            "instruction": (
                f"Respond in {meta['name']} using native {meta['script']} script. "
                "Do not use Romanized transliteration unless explicitly requested."
            ),
        }