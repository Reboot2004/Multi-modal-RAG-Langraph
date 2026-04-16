import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from config.settings import GRAPH_INDEX_PATH, GRAPH_COMMUNITY_TOP_K


class GraphRAGIndex:
    """Persisted graph-style index with entities, co-occurrence relations, and source communities."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "because", "been", "being", "but",
        "by", "can", "could", "do", "does", "done", "during", "each", "either", "for",
        "from", "had", "has", "have", "having", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "may",
        "might", "more", "most", "much", "must", "my", "near", "no", "not", "of",
        "on", "or", "our", "out", "over", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "up", "use", "used",
        "using", "very", "was", "we", "were", "what", "when", "where", "which", "who",
        "will", "with", "within", "without", "would", "you", "your", "about", "across",
        "after", "before", "between", "into", "onto", "than", "via",
    }

    FILLER_WORDS = {
        "actually", "basically", "clearly", "kind", "kinda", "like", "literally", "maybe",
        "mostly", "mostly", "obviously", "okay", "really", "simply", "stuff", "thing",
        "things", "unknown", "whatever", "wherever", "etc", "page", "source", "section",
        "document", "content", "chunk", "chunks", "graph", "community",
    }

    def __init__(self, index_path: str = GRAPH_INDEX_PATH):
        self.index_path = index_path

    def build(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        entities = Counter()
        relations = Counter()
        by_source = defaultdict(list)
        keywords_by_source = defaultdict(Counter)

        for index, item in enumerate(chunks):
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            source = self._normalize_source(meta, index=index)
            text = (item.get("text", "") or "").strip()
            if not text:
                continue

            by_source[source].append(text)
            for term in self._meaningful_tokens(text):
                keywords_by_source[source][term] += 1
            ent = self._extract_entities(text)
            for e in ent:
                entities[e] += 1
            for i in range(len(ent)):
                for j in range(i + 1, min(len(ent), i + 6)):
                    a, b = sorted([ent[i], ent[j]])
                    relations[(a, b)] += 1

        communities = []
        for source, texts in by_source.items():
            keywords = [term for term, _ in keywords_by_source[source].most_common(24)]
            source_entities = self._top_entities_for_texts(texts)
            lexicon = self._build_lexicon(source_entities, keywords)
            summary = self._build_summary(source, texts, keywords, source_entities)
            communities.append(
                {
                    "source": source,
                    "summary": summary,
                    "entities": source_entities,
                    "keywords": keywords,
                    "lexicon": lexicon,
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
        q_tokens = set(self._meaningful_tokens(question))

        scored = []
        for c in communities:
            c_entities = set(c.get("entities", []))
            c_tokens = set(c.get("keywords", [])) | set(c.get("lexicon", [])) | set(self._meaningful_tokens(c.get("summary", "")))

            entity_overlap = len(q_entities & c_entities)
            token_overlap = len(q_tokens & c_tokens)
            score = (1.8 * entity_overlap) + (1.2 * token_overlap) + (0.03 * float(c.get("size", 1)))
            if score <= 0:
                continue
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, c in scored[: max(1, int(top_k))]:
            source = c.get("source", "graph_community")
            keywords = c.get("keywords", [])[:10]
            results.append(
                {
                    "score": float(score),
                    "text": c.get("summary", ""),
                    "metadata": {
                        "source": source,
                        "display_source": source,
                        "page": "graph",
                        "graph_mode": True,
                        "community_size": c.get("size", 0),
                        "keywords": keywords,
                        "lexicon": c.get("lexicon", [])[:12],
                        "entities": c.get("entities", [])[:12],
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
        cleaned = [c.strip() for c in candidates if len(c.strip()) >= 3 and c.lower() not in self.STOP_WORDS]
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

    def _meaningful_tokens(self, text: str) -> List[str]:
        tokens = []
        for token in self._tokens(text):
            normalized = token.strip("_-.")
            if len(normalized) <= 2:
                continue
            if normalized in self.STOP_WORDS or normalized in self.FILLER_WORDS:
                continue
            if normalized.isdigit():
                continue
            tokens.append(normalized)

        seen = set()
        deduped = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def _normalize_source(self, meta: Dict[str, Any], index: int) -> str:
        source = str(meta.get("source", "") or "").strip()
        if source and source.lower() not in {"unknown", "none", "null"}:
            return source

        section_hint = str(meta.get("section_hint", "") or meta.get("chunk_type", "") or "").strip()
        page = meta.get("page")

        if section_hint and section_hint.lower() not in {"unknown", "mixed", "paragraph"}:
            safe_section = re.sub(r"[^A-Za-z0-9_\-]+", "_", section_hint).strip("_")
            if page not in {None, "", "graph"}:
                return f"{safe_section}_page_{page}"
            return safe_section or f"community_{index}"

        if page not in {None, "", "graph"}:
            return f"page_{page}"

        return f"community_{index}"

    def _build_lexicon(self, entities: List[str], keywords: List[str]) -> List[str]:
        merged = []
        seen = set()
        for term in list(entities) + list(keywords):
            key = term.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(term)
        return merged[:24]

    def _build_summary(self, source: str, texts: List[str], keywords: List[str], entities: List[str]) -> str:
        merged = " ".join(texts[:8])
        merged = re.sub(r"\s+", " ", merged).strip()
        leading = merged[:900]

        keyword_text = ", ".join(keywords[:8]) if keywords else ""
        entity_text = ", ".join(entities[:6]) if entities else ""

        parts = [f"{source}."]
        if keyword_text:
            parts.append(f"Key terms: {keyword_text}.")
        if entity_text:
            parts.append(f"Entities: {entity_text}.")
        if leading:
            parts.append(leading)

        return " ".join(parts)[:1200]

    def _top_entities_for_texts(self, texts: List[str]) -> List[str]:
        counter = Counter()
        for t in texts[:20]:
            for e in self._extract_entities(t):
                counter[e] += 1
        return [k for k, _ in counter.most_common(20)]
