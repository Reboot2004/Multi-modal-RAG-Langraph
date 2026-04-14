import hashlib
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

from config.settings import ENABLE_DATA_LIFECYCLE, DATA_MANIFEST_PATH, STALE_DATA_MAX_AGE_DAYS


class DataLifecycleManager:
    def __init__(self):
        self.enabled = bool(ENABLE_DATA_LIFECYCLE)
        self.path = DATA_MANIFEST_PATH

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"documents": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"documents": {}}

    def _write(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def register_ingestion(self, files: List[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        manifest = self._read()
        docs = manifest.setdefault("documents", {})
        now = datetime.now(timezone.utc).isoformat()

        for item in files:
            source = (item.get("source") or "unknown").strip()
            path = item.get("path") or source
            fingerprint = self._hash_file(path)
            current = docs.get(source, {})
            prev_hash = current.get("fingerprint")
            version = int(current.get("version", 0)) + (1 if prev_hash and prev_hash != fingerprint else 0)
            docs[source] = {
                "source": source,
                "path": path,
                "fingerprint": fingerprint,
                "version": version,
                "updated_at": now,
                "deleted": False,
            }

        manifest["last_ingestion_at"] = now
        self._write(manifest)

    def mark_deleted(self, source: str) -> None:
        if not self.enabled:
            return
        manifest = self._read()
        docs = manifest.setdefault("documents", {})
        if source in docs:
            docs[source]["deleted"] = True
            docs[source]["deleted_at"] = datetime.now(timezone.utc).isoformat()
            self._write(manifest)

    def get_source_timestamps(self) -> Dict[str, str]:
        manifest = self._read()
        docs = manifest.get("documents", {})
        return {name: meta.get("updated_at", "") for name, meta in docs.items()}

    def stale_sources(self) -> List[str]:
        manifest = self._read()
        docs = manifest.get("documents", {})
        threshold_days = int(STALE_DATA_MAX_AGE_DAYS)
        now = datetime.now(timezone.utc)
        stale = []
        for source, meta in docs.items():
            updated = meta.get("updated_at")
            if not updated:
                continue
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except Exception:
                continue
            age_days = (now - dt).days
            if age_days >= threshold_days and not meta.get("deleted", False):
                stale.append(source)
        return sorted(stale)

    def _hash_file(self, path: str) -> str:
        sha = hashlib.sha256()
        if not os.path.exists(path):
            return "missing"
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception:
            return "unreadable"
