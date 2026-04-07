from llm.groq_client import GroqClient
from llm.openrouter_client import OpenRouterClient


def build_llm_client(provider: str, model_name: str = None, api_key: str = None):
    provider_key = (provider or "groq").strip().lower()

    if provider_key == "groq":
        client = GroqClient(api_key=api_key)
        if model_name:
            client.model_name = model_name.strip()
        return client

    if provider_key == "openrouter":
        return OpenRouterClient(model_name=model_name, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")
