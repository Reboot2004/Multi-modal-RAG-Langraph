import argparse
import os
import sys
from typing import Dict, List, Tuple

from datasets import load_dataset

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


def load_squad_examples(max_samples: int) -> Tuple[Dict[str, str], List[BenchmarkExample]]:
    ds = load_dataset("squad_v2", split=f"validation[:{max_samples * 2}]")

    corpus: Dict[str, str] = {}
    examples: List[BenchmarkExample] = []

    for row in ds:
        answers = [a.strip() for a in row.get("answers", {}).get("text", []) if (a or "").strip()]
        if not answers:
            continue

        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        if not question or not context:
            continue

        doc_id = stable_hash(context)
        corpus.setdefault(doc_id, context)

        examples.append(
            BenchmarkExample(
                qid=f"squad::{row.get('id', stable_hash(question)[:12])}",
                question=question,
                answers=answers,
                gold_doc_ids=[doc_id],
            )
        )

        if len(examples) >= max_samples:
            break

    return corpus, examples


def load_hotpot_examples(max_samples: int) -> Tuple[Dict[str, str], List[BenchmarkExample]]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{max_samples}]")

    corpus: Dict[str, str] = {}
    examples: List[BenchmarkExample] = []

    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue

        context = row.get("context", {})
        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences_by_title = context.get("sentences", []) if isinstance(context, dict) else []

        if not titles or not sentences_by_title:
            continue

        local_doc_ids = []
        for title, sentences in zip(titles, sentences_by_title):
            text = " ".join(sentences or []).strip()
            if not text:
                continue

            doc_key = f"{title}::{text[:200]}"
            doc_id = stable_hash(doc_key)
            corpus.setdefault(doc_id, f"{title}\n{text}")
            local_doc_ids.append((title, doc_id))

        support = row.get("supporting_facts", {})
        support_titles = set(support.get("title", [])) if isinstance(support, dict) else set()
        gold_doc_ids = [doc_id for title, doc_id in local_doc_ids if title in support_titles]

        if not gold_doc_ids:
            # Fallback: keep at least one expected relevant doc.
            if local_doc_ids:
                gold_doc_ids = [local_doc_ids[0][1]]
            else:
                continue

        examples.append(
            BenchmarkExample(
                qid=f"hotpot::{row.get('id', stable_hash(question)[:12])}",
                question=question,
                answers=[answer],
                gold_doc_ids=gold_doc_ids,
            )
        )

        if len(examples) >= max_samples:
            break

    return corpus, examples


def evaluate_dataset(
    name: str,
    corpus: Dict[str, str],
    examples: List[BenchmarkExample],
    embedder,
    top_k: int,
) -> Dict:
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
        "dataset": name,
        "samples": len(examples),
        "corpus_size": len(corpus),
        "metrics": aggregate_metrics(rows),
        "per_query": rows,
    }


def run_comprehensive_benchmark(squad_samples: int, hotpot_samples: int, top_k: int) -> Dict:
    embedder = MultilingualEmbedder()

    squad_corpus, squad_examples = load_squad_examples(max_samples=squad_samples)
    hotpot_corpus, hotpot_examples = load_hotpot_examples(max_samples=hotpot_samples)

    squad_report = evaluate_dataset(
        name="squad_v2",
        corpus=squad_corpus,
        examples=squad_examples,
        embedder=embedder,
        top_k=top_k,
    )
    hotpot_report = evaluate_dataset(
        name="hotpot_qa_distractor",
        corpus=hotpot_corpus,
        examples=hotpot_examples,
        embedder=embedder,
        top_k=top_k,
    )

    aggregate_rows = []
    for block in (squad_report, hotpot_report):
        for row in block["per_query"]:
            aggregate_rows.append(row)

    return {
        "benchmark": "comprehensive_open_rag_benchmark",
        "datasets": [squad_report, hotpot_report],
        "overall_metrics": aggregate_metrics(aggregate_rows),
        "config": {
            "top_k": top_k,
            "squad_samples": squad_samples,
            "hotpot_samples": hotpot_samples,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive open-source RAG benchmark")
    parser.add_argument("--squad-samples", type=int, default=300)
    parser.add_argument("--hotpot-samples", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        default="data/processed/benchmarks/comprehensive_benchmark.json",
        help="Path to write comprehensive benchmark report JSON",
    )
    args = parser.parse_args()

    report = run_comprehensive_benchmark(
        squad_samples=max(50, int(args.squad_samples)),
        hotpot_samples=max(50, int(args.hotpot_samples)),
        top_k=max(5, int(args.top_k)),
    )
    save_json(args.output, report)

    print("Comprehensive Benchmark Completed")
    print("Overall metrics:")
    for key, value in report["overall_metrics"].items():
        print(f"- {key}: {value}")

    for dataset_report in report["datasets"]:
        print(f"Dataset: {dataset_report['dataset']}")
        print(f"- samples: {dataset_report['samples']}")
        print(f"- corpus_size: {dataset_report['corpus_size']}")
        for key, value in dataset_report["metrics"].items():
            print(f"  - {key}: {value}")

    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
