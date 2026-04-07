from typing import List
import os
import json
import hashlib

from config.settings import (
    HYPE_PROMPTS_PER_CHUNK,
    HYPE_SOURCE_CHAR_LIMIT,
    ENABLE_HYPE_CACHE,
    HYPE_CACHE_PATH,
)
from llm.groq_client import GroqClient
from pipeline_logger import get_logger


logger = get_logger("hype_generator")


class HyPEGenerator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or GroqClient()
        self.prompts_per_chunk = HYPE_PROMPTS_PER_CHUNK
        self.source_char_limit = HYPE_SOURCE_CHAR_LIMIT
        self.enable_cache = ENABLE_HYPE_CACHE
        self.cache_path = HYPE_CACHE_PATH
        self.prompt_cache = self._load_cache()
        logger.info(
            "HyPEGenerator initialized | prompts_per_chunk=%d | source_char_limit=%d",
            self.prompts_per_chunk,
            self.source_char_limit,
        )
        logger.info(
            "HyPE cache | enabled=%s | entries=%d",
            self.enable_cache,
            len(self.prompt_cache),
        )

    def set_prompts_per_chunk(self, value: int):
        self.prompts_per_chunk = max(1, int(value))
        logger.info("HyPE prompts_per_chunk updated to %d", self.prompts_per_chunk)

    def generate_prompts_for_chunk(self, chunk_text: str, language: str) -> List[str]:
        """
        Generate hypothetical user questions for a document chunk.
        Returns a short list of clean question strings.
        """

        if not chunk_text or not chunk_text.strip():
            logger.debug("Skipped HyPE generation for empty chunk")
            return []

        clipped_text = chunk_text[: self.source_char_limit]
        cache_key = self._build_cache_key(clipped_text, language, self.prompts_per_chunk)

        if self.enable_cache:
            cached = self.prompt_cache.get(cache_key)
            if cached and isinstance(cached, list):
                logger.debug("HyPE cache hit | key=%s | prompts=%d", cache_key[:12], len(cached))
                return cached[: self.prompts_per_chunk]

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate short search-style user questions from document text. "
                    "Return one question per line. No numbering. No bullets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate exactly {self.prompts_per_chunk} realistic user questions "
                    f"that can be answered using the text below.\n"
                    f"Language code to use: {language}.\n"
                    "Keep questions concise and retrieval-friendly.\n\n"
                    f"Text:\n{clipped_text}"
                ),
            },
        ]

        raw_output = self.llm_client.generate(messages)
        logger.debug("Raw HyPE model output: %s", (raw_output or "")[:800])
        parsed_prompts = self._parse_prompts(raw_output)
        logger.debug("Parsed HyPE prompts: %s", parsed_prompts)

        if len(parsed_prompts) >= self.prompts_per_chunk:
            final_prompts = parsed_prompts[: self.prompts_per_chunk]
            self._store_cache(cache_key, final_prompts)
            return final_prompts

        fallback = []
        if parsed_prompts:
            fallback.extend(parsed_prompts)

        while len(fallback) < self.prompts_per_chunk:
            fallback.append("What does this section explain?")

        logger.debug("Using fallback HyPE prompts: %s", fallback)
        final_prompts = fallback[: self.prompts_per_chunk]
        self._store_cache(cache_key, final_prompts)
        return final_prompts

    def _parse_prompts(self, raw_output: str) -> List[str]:
        lines = [line.strip() for line in (raw_output or "").splitlines() if line.strip()]

        clean = []
        for line in lines:
            trimmed = line.lstrip("-•0123456789. ").strip()
            if trimmed and len(trimmed) > 4:
                clean.append(trimmed)

        deduped = []
        seen = set()
        for item in clean:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped

    def _build_cache_key(self, clipped_text: str, language: str, prompts_per_chunk: int) -> str:
        digest = hashlib.sha256(clipped_text.encode("utf-8")).hexdigest()
        return f"{language}:{prompts_per_chunk}:{digest}"

    def _load_cache(self):
        if not self.enable_cache:
            return {}

        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception as ex:
            logger.warning("Failed to load HyPE cache: %s", ex)

        return {}

    def _store_cache(self, key: str, prompts: List[str]):
        if not self.enable_cache:
            return

        self.prompt_cache[key] = prompts

        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.prompt_cache, f, ensure_ascii=False, indent=2)
        except Exception as ex:
            logger.warning("Failed to persist HyPE cache: %s", ex)