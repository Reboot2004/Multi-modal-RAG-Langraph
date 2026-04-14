import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from config.settings import GRAPH_INDEX_PATH, GRAPH_COMMUNITY_TOP_K


class GraphRAGIndex:
    """Persisted graph-style index with entities, co-occurrence relations, and source communities."""

    def __init__(self, index_path: str = GRAPH_INDEX_PATH):
        self.index_path = index_path

    def build(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        entities = Counter()
        relations = Counter()
        by_source = defaultdict(list)

        for item in chunks:
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            source = meta.get("source", "unknown")
            text = (item.get("text", "") or "").strip()
            if not text:
                continue

            by_source[source].append(text)
            ent = self._extract_entities(text)
            for e in ent:
                entities[e] += 1
            for i in range(len(ent)):
                for j in range(i + 1, min(len(ent), i + 6)):
                    a, b = sorted([ent[i], ent[j]])
                    relations[(a, b)] += 1

        communities = []
        for source, texts in by_source.items():
            summary = self._summarize_texts(texts)
            source_entities = self._top_entities_for_texts(texts)
            communities.append(
                {
                    "source": source,
                    "summary": summary,
                    "entities": source_entities,
                    "size": len(texts),
                }
            )

        payload = {
            "entities": [{"name": k, "count": v} for k, v in entities.most_common(2000)],
            "relations": [
                {"a": a, "b": b, "count": c}
                for (a, b), c in relations.most_common(4000)
            ],
            "communities": communities,
        }
        self._write(payload)
        return payload

    def query(self, question: str, top_k: int = GRAPH_COMMUNITY_TOP_K) -> List[Dict[str, Any]]:
        payload = self._read()
        communities = payload.get("communities", [])
        q_entities = set(self._extract_entities(question))
        q_tokens = set(self._tokens(question))

        scored = []
        for c in communities:
            c_entities = set(c.get("entities", []))
            c_tokens = set(self._tokens(c.get("summary", "")))

            entity_overlap = len(q_entities & c_entities)
            token_overlap = len(q_tokens & c_tokens)
            score = (1.5 * entity_overlap) + token_overlap + (0.05 * float(c.get("size", 1)))
            if score <= 0:
                continue
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, c in scored[: max(1, int(top_k))]:
            results.append(
                {
                    "score": float(score),
                    "text": c.get("summary", ""),
                    "metadata": {
                        "source": c.get("source", "graph_community"),
                        "page": "graph",
                        "graph_mode": True,
                        "community_size": c.get("size", 0),
                    },
                }
            )
        return results

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.index_path):
            return {"entities": [], "relations": [], "communities": []}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"entities": [], "relations": [], "communities": []}

    def _write(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _extract_entities(self, text: str) -> List[str]:
        # Heuristic entities: title-cased words, ALLCAPS tokens, and alnum IDs.
        candidates = re.findall(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,}|[A-Za-z]+\d+)\b", text or "")
        cleaned = [c.strip() for c in candidates if len(c.strip()) >= 3]
        # Keep order while deduplicating.
        seen = set()
        out = []
        for item in cleaned:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out[:40]

    def _tokens(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower()) if len(t) > 2]

    def _summarize_texts(self, texts: List[str]) -> str:
        merged = " ".join(texts[:8])
        merged = re.sub(r"\s+", " ", merged).strip()
        return merged[:1200]

    def _top_entities_for_texts(self, texts: List[str]) -> List[str]:
        counter = Counter()
        for t in texts[:20]:
            for e in self._extract_entities(t):
                counter[e] += 1
        return [k for k, _ in counter.most_common(20)]
