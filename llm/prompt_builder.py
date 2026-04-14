# prompt_builder.py
# llm/prompt_builder.py

from typing import List, Dict
from config.settings import (
    SYSTEM_PROMPT_TEMPLATE,
    PROMPT_MAX_CONTEXT_CHARS,
    PROMPT_MAX_CONTEXT_CHUNKS,
)
from pipeline_logger import get_logger
from utils.prompt_registry import get_active_system_prompt


logger = get_logger("prompt_builder")


class PromptBuilder:
    def __init__(self):
        registry_entry = get_active_system_prompt()
        self.system_prompt = registry_entry.get("system_prompt", SYSTEM_PROMPT_TEMPLATE)
        self.max_context_chars = PROMPT_MAX_CONTEXT_CHARS
        self.max_context_chunks = PROMPT_MAX_CONTEXT_CHUNKS

    def build_prompt(
        self,
        query: str,
        query_language: str,
        retrieved_chunks: List[Dict],
        response_language_instruction: str = "",
        response_language_name: str = "",
    ) -> List[Dict]:
        """
        Returns messages list formatted for Groq chat completion API.
        """

        # Combine retrieved context within budget
        context_blocks = []
        context_chars = 0

        for i, item in enumerate(retrieved_chunks[: self.max_context_chunks]):
            source = item["metadata"].get("source", "unknown")
            page = item["metadata"].get("page", "unknown")

            block = f"[Source: {source}, Page: {page}]\n{item['text']}"
            next_chars = len(block) + (2 if context_blocks else 0)
            if context_chars + next_chars > self.max_context_chars:
                remaining = self.max_context_chars - context_chars
                if remaining > 120:
                    clipped = block[:remaining].rstrip()
                    context_blocks.append(clipped)
                    context_chars += len(clipped)
                break

            context_blocks.append(block)
            context_chars += next_chars

        combined_context = "\n\n".join(context_blocks)
        logger.info(
            "Prompt build | query_chars=%d | retrieved_chunks=%d | used_chunks=%d | context_chars=%d",
            len(query or ""),
            len(retrieved_chunks),
            len(context_blocks),
            len(combined_context),
        )

        user_prompt = f"""
Context:
{combined_context}

Question:
{query}

Instructions:
- Answer strictly using the context above.
- If the answer is not present, say you do not know.
- Do not repeat or restate the question.
- Start directly with the answer.
- Respond ONLY in one language: {response_language_name or query_language} (code: {query_language}).
- Do NOT provide bilingual output.
- Do NOT provide an English version before or after the final answer.
- Do NOT add translations unless explicitly requested by the user.
{('- ' + response_language_instruction) if response_language_instruction else ''}
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]

        logger.debug("Prompt messages built | total_messages=%d", len(messages))

        return messages