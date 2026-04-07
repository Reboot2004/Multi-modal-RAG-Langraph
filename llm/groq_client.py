# groq_client.py
# llm/groq_client.py

import os

from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME
from pipeline_logger import get_logger


logger = get_logger("groq_client")


class GroqClient:
    def __init__(self, api_key: str = None):
        resolved_api_key = (api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY") or "").strip()
        if not resolved_api_key:
            raise ValueError("GROQ_API_KEY not set in environment variables or runtime input")

        self.client = Groq(api_key=resolved_api_key)
        self.model_name = GROQ_MODEL_NAME
        logger.info("GroqClient initialized | model=%s", self.model_name)

    def generate(self, messages, max_tokens=None, temperature=0.2):
        """
        messages: list of {"role": "...", "content": "..."}
        Returns: model response text
        """

        safe_messages = [dict(item) for item in messages]
        safe_max_tokens = int(max_tokens) if max_tokens is not None else None

        for attempt in range(1, 4):
            request_args = {
                "model": self.model_name,
                "messages": safe_messages,
                "temperature": temperature,
            }
            if safe_max_tokens is not None:
                request_args["max_tokens"] = int(safe_max_tokens)

            try:
                completion = self.client.chat.completions.create(**request_args)
                output = completion.choices[0].message.content
                logger.debug("Groq generate completed | output_chars=%d", len(output or ""))
                return output
            except Exception as ex:
                error_text = str(ex)
                is_too_large = (
                    "Request too large" in error_text
                    or "rate_limit_exceeded" in error_text
                    or "tokens per minute" in error_text
                    or "Error code: 413" in error_text
                )

                if not is_too_large or attempt == 3:
                    raise

                logger.warning(
                    "Groq request too large; shrinking payload | attempt=%d | max_tokens=%s",
                    attempt,
                    safe_max_tokens,
                )

                safe_messages = self._shrink_messages(safe_messages, factor=0.7)
                if safe_max_tokens is None:
                    safe_max_tokens = 800
                else:
                    safe_max_tokens = max(256, int(safe_max_tokens * 0.7))

        raise RuntimeError("Groq generation failed after retries")

    def _shrink_messages(self, messages, factor: float = 0.7):
        shrunk = []
        for message in messages:
            item = dict(message)
            content = str(item.get("content", ""))

            if item.get("role") == "system":
                shrunk.append(item)
                continue

            if len(content) > 500:
                keep = max(300, int(len(content) * factor))
                head = content[: int(keep * 0.8)]
                tail = content[-int(keep * 0.2):]
                item["content"] = (head + "\n\n[...truncated for token budget...]\n\n" + tail).strip()

            shrunk.append(item)

        return shrunk
  