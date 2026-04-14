import re
from typing import Any


class PIIGuard:
    EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
    PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}\b")
    CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")

    @classmethod
    def redact_text(cls, text: str) -> str:
        value = text or ""
        value = cls.EMAIL_RE.sub("[REDACTED_EMAIL]", value)
        value = cls.PHONE_RE.sub("[REDACTED_PHONE]", value)
        value = cls.CARD_RE.sub("[REDACTED_CARD]", value)
        value = cls.AADHAAR_RE.sub("[REDACTED_AADHAAR]", value)
        return value

    @classmethod
    def sanitize_payload(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls.sanitize_payload(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.sanitize_payload(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cls.sanitize_payload(v) for v in obj)
        if isinstance(obj, str):
            return cls.redact_text(obj)
        return obj
