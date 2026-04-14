import json
import os
from typing import Dict, Any

from config.settings import (
    ENABLE_PROMPT_REGISTRY,
    PROMPT_REGISTRY_PATH,
    PROMPT_REGISTRY_STRICT_APPROVAL,
    SYSTEM_PROMPT_TEMPLATE,
)


def get_active_system_prompt() -> Dict[str, Any]:
    default = {
        "prompt_id": "default",
        "version": "1.0.0",
        "approved": True,
        "risk_tier": "standard",
        "system_prompt": SYSTEM_PROMPT_TEMPLATE,
    }
    if not ENABLE_PROMPT_REGISTRY or not os.path.exists(PROMPT_REGISTRY_PATH):
        return default

    try:
        with open(PROMPT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return default

    active = payload.get("active", {})
    approved = bool(active.get("approved", False))
    if PROMPT_REGISTRY_STRICT_APPROVAL and not approved:
        return default

    prompt = active.get("system_prompt") or SYSTEM_PROMPT_TEMPLATE
    return {
        "prompt_id": active.get("prompt_id", "registry"),
        "version": active.get("version", "1.0.0"),
        "approved": approved,
        "risk_tier": active.get("risk_tier", "standard"),
        "system_prompt": prompt,
    }
