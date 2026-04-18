import argparse
import os
import sys
from typing import Dict, List

from datasets import load_dataset

# Ensure repo root imports work when running from scripts/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from embeddings.embedder import MultilingualEmbedder
from scripts.benchmark_common import (
    BenchmarkExample,
    InMemorySemanticIndex,
    aggregate_metrics,
    answer_coverage_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    save_json,
    stable_hash,
)


def build_quick_squad_benchmark(max_samples: int) -> (Dict[str, str], List[BenchmarkExample]):
    ds = load_dataset("squad_v2", split=f"validation[:{max_samples * 2}]")

    corpus: Dict[str, str] = {}
    examples: List[BenchmarkExample] = []

    for row in ds:
        answers = [a.strip() for a in row.get("answers", {}).get("text", []) if (a or "").strip()]
        if not answers:
            # Keep quick benchmark answerable only for coverage metrics.
            continue

        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        if not context or not question:
            continue

        doc_id = stable_hash(context)
        corpus.setdefault(doc_id, context)

        qid = str(row.get("id") or stable_hash(question + context)[:16])
        examples.append(
            BenchmarkExample(
                qid=qid,
                question=question,
                answers=answers,
                gold_doc_ids=[doc_id],
            )
        )

        if len(examples) >= max_samples:
            break

    return corpus, examples


def run_quick_benchmark(max_samples: int, top_k: int) -> Dict:
    corpus, examples = build_quick_squad_benchmark(max_samples=max_samples)

    embedder = MultilingualEmbedder()
    index = InMemorySemanticIndex(embedder=embedder)
    index.build(corpus)

    rows = []

    for ex in examples:
        ranked = index.search(ex.question, top_k=top_k)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        ranked_texts = [corpus[doc_id] for doc_id in ranked_doc_ids if doc_id in corpus]

        rows.append(
            {
                "qid": ex.qid,
                "recall@5": recall_at_k(ex.gold_doc_ids, ranked_doc_ids, k=5),
                "recall@10": recall_at_k(ex.gold_doc_ids, ranked_doc_ids, k=10),
                "hit@5": hit_at_k(ex.gold_doc_ids, ranked_doc_ids, k=5),
                "hit@10": hit_at_k(ex.gold_doc_ids, ranked_doc_ids, k=10),
                "mrr": mrr(ex.gold_doc_ids, ranked_doc_ids),
                "ndcg@10": ndcg_at_k(ex.gold_doc_ids, ranked_doc_ids, k=10),
                "answer_coverage@5": answer_coverage_at_k(ex.answers, ranked_texts, k=5),
                "answer_coverage@10": answer_coverage_at_k(ex.answers, ranked_texts, k=10),
            }
        )

    return {
        "benchmark": "quick_squad_v2",
        "dataset": "squad_v2",
        "samples": len(examples),
        "corpus_size": len(corpus),
        "top_k": top_k,
        "metrics": aggregate_metrics(rows),
        "per_query": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick open-source RAG benchmark (SQuAD v2 retrieval)")
    parser.add_argument("--samples", type=int, default=120, help="Number of benchmark questions")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K docs to retrieve")
    parser.add_argument(
        "--output",
        default="data/processed/benchmarks/quick_benchmark.json",
        help="Path to write benchmark report JSON",
    )
    args = parser.parse_args()

    report = run_quick_benchmark(max_samples=max(20, int(args.samples)), top_k=max(5, int(args.top_k)))
    save_json(args.output, report)

    print("Quick Benchmark Completed")
    print(f"- dataset: {report['dataset']}")
    print(f"- samples: {report['samples']}")
    print(f"- corpus_size: {report['corpus_size']}")
    print("- metrics:")
    for key, value in report["metrics"].items():
        print(f"  - {key}: {value}")
    print(f"- output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
