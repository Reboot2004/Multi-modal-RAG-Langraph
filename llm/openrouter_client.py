import json
import os
from urllib import error, request

from config.settings import OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME
from pipeline_logger import get_logger


logger = get_logger("openrouter_client")


class OpenRouterClient:
    def __init__(self, model_name: str = None, api_key: str = None):
        self.api_key = (api_key or OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment variables or runtime input")

        self.model_name = (model_name or OPENROUTER_MODEL_NAME).strip()
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        logger.info("OpenRouterClient initialized | model=%s", self.model_name)

    def generate(self, messages, max_tokens=None, temperature=0.2):
        safe_messages = [dict(item) for item in messages]
        safe_max_tokens = int(max_tokens) if max_tokens is not None else None

        for attempt in range(1, 4):
            payload = {
                "model": self.model_name,
                "messages": safe_messages,
                "temperature": temperature,
            }
            if safe_max_tokens is not None:
                payload["max_tokens"] = int(safe_max_tokens)

            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.endpoint,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            try:
                with request.urlopen(req, timeout=90) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                    parsed = json.loads(body)

                choices = parsed.get("choices", [])
                if not choices:
                    raise RuntimeError("OpenRouter response missing choices")

                message = choices[0].get("message", {})
                content = message.get("content", "")

                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    output = "\n".join([p for p in text_parts if p])
                else:
                    output = content or ""

                logger.debug("OpenRouter generate completed | output_chars=%d", len(output))
                return output

            except error.HTTPError as ex:
                detail = ex.read().decode("utf-8", errors="ignore") if ex.fp else str(ex)
                error_text = f"HTTP {ex.code}: {detail}"
                is_too_large = ex.code in (413, 429) or "rate_limit" in detail.lower()

                if not is_too_large or attempt == 3:
                    raise RuntimeError(f"OpenRouter request failed: {error_text}") from ex

                logger.warning(
                    "OpenRouter request too large/limited; shrinking payload | attempt=%d | max_tokens=%s",
                    attempt,
                    safe_max_tokens,
                )

                safe_messages = self._shrink_messages(safe_messages, factor=0.7)
                if safe_max_tokens is None:
                    safe_max_tokens = 800
                else:
                    safe_max_tokens = max(256, int(safe_max_tokens * 0.7))

            except Exception as ex:
                if attempt == 3:
                    raise

                logger.warning("OpenRouter request retry | attempt=%d | error=%s", attempt, ex)

        raise RuntimeError("OpenRouter generation failed after retries")

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
