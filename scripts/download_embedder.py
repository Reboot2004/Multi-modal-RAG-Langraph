#!/usr/bin/env python
"""
Pre-download and cache embedding model locally.

Usage (PowerShell):
  python scripts/download_embedder.py
  python scripts/download_embedder.py --model BAAI/bge-m3 --cache-dir ./.hf_cache
  python scripts/download_embedder.py --local-only
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache embedder model locally")
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Hugging Face model id to cache (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional local cache directory (default: HF cache)",
    )
    parser.add_argument(
        "--etag-timeout",
        type=int,
        default=30,
        help="HF metadata timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=180,
        help="HF download timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only already-downloaded local cache; fail if missing",
    )
    parser.add_argument(
        "--install-hf-xet",
        action="store_true",
        help="Print suggestion to install hf_xet for better HF transfer performance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(max(5, int(args.etag_timeout)))
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(max(30, int(args.download_timeout)))

    if args.cache_dir:
        cache_path = Path(args.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)

    if args.install_hf_xet:
        print("Tip: install hf_xet for faster transfers: pip install huggingface_hub[hf_xet] hf_xet")

    print(f"[1/3] Preparing download for model: {args.model}")
    if args.cache_dir:
        print(f"      Cache dir: {os.environ.get('HF_HOME')}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as ex:
        print(f"Failed to import sentence_transformers: {ex}")
        return 2

    # Prefer constructor with local_files_only if available; fallback for older versions.
    print("[2/3] Loading model via sentence-transformers...")
    model = None
    try:
        model = SentenceTransformer(args.model, local_files_only=bool(args.local_only))
    except TypeError:
        # Older sentence-transformers may not support local_files_only.
        if args.local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        model = SentenceTransformer(args.model)
    except Exception as ex:
        print(f"Model load failed: {ex}")
        return 1

    print("[3/3] Verifying model with a tiny encode call...")
    try:
        vec = model.encode("cache warmup", convert_to_numpy=True, normalize_embeddings=True)
        dim = int(getattr(vec, "shape", [0])[-1]) if hasattr(vec, "shape") else -1
        print(f"Success. Model cached and usable. Embedding dim: {dim}")
        return 0
    except Exception as ex:
        print(f"Model loaded but encode test failed: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
