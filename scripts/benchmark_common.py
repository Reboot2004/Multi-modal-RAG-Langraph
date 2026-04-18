import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np


def stable_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class BenchmarkExample:
    qid: str
    question: str
    answers: List[str]
    gold_doc_ids: List[str]


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


class InMemorySemanticIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def build(self, corpus: Dict[str, str]) -> None:
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]
        self.doc_embeddings = self.embedder.embed_texts(self.doc_texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.doc_embeddings.size == 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        scores = np.dot(self.doc_embeddings, query_embedding)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]


def recall_at_k(gold_doc_ids: List[str], ranked_doc_ids: List[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    top = set(ranked_doc_ids[:k])
    hits = sum(1 for g in gold_doc_ids if g in top)
    return hits / max(1, len(set(gold_doc_ids)))


def hit_at_k(gold_doc_ids: List[str], ranked_doc_ids: List[str], k: int) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return 1.0 if any(doc_id in gold for doc_id in ranked_doc_ids[:k]) else 0.0


def mrr(gold_doc_ids: List[str], ranked_doc_ids: List[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_doc_ids: List[str], ranked_doc_ids: List[str], k: int) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in gold else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)

    ideal_rels = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rels + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def answer_coverage_at_k(answers: List[str], ranked_texts: List[str], k: int) -> float:
    normalized_answers = [normalize_text(ans) for ans in (answers or []) if normalize_text(ans)]
    if not normalized_answers:
        return 0.0

    joined = "\n".join(normalize_text(text) for text in ranked_texts[:k])
    for answer in normalized_answers:
        if answer and answer in joined:
            return 1.0
    return 0.0


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "hit@5": 0.0,
            "hit@10": 0.0,
            "mrr": 0.0,
            "ndcg@10": 0.0,
            "answer_coverage@5": 0.0,
            "answer_coverage@10": 0.0,
        }

    keys = [k for k in rows[0].keys() if k != "qid"]
    metrics = {"count": len(rows)}
    for key in keys:
        metrics[key] = float(sum(row.get(key, 0.0) for row in rows) / len(rows))
    return metrics


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
