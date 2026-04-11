#!/usr/bin/env python3
# scripts/download_datasets.py
# =============================================================================
# Dataset Download & Ingestion Script
# ─────────────────────────────────────
# Downloads three open-source HuggingFace datasets and POSTs them to the
# Flask ingestion API.
#
# Datasets:
#   1. SciQ  — 13,679 crowdsourced science questions + support passages
#              https://huggingface.co/datasets/allenai/sciq
#              Best for: testing question generation on science passages
#
#   2. SQuAD v2  — 150,000 QA pairs from Wikipedia
#                  https://huggingface.co/datasets/rajpurkar/squad_v2
#                  Best for: evaluating semantic answer grading
#
#   3. arXiv abstracts (cs.AI subset, 2020-2024)
#              https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021
#              Best for: domain-specific knowledge graph (AI research)
#
# Usage:
#   python scripts/download_datasets.py [--dataset all|sciq|squad|arxiv]
#                                       [--limit 200]
#                                       [--api http://localhost:5001]
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("datasets")

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_API = "http://localhost:5001"
INGEST_ENDPOINT = "/api/ingest/dataset"
BATCH_SIZE = 50  # items per POST request


# ── Helpers ────────────────────────────────────────────────────────────────────

def post_batch(api_base: str, items: list[dict], source_label: str) -> dict:
    """POST a batch of {text, source} items to the Flask API."""
    url = f"{api_base}{INGEST_ENDPOINT}"
    payload = {"items": items}
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        log.error(
            "Cannot connect to API at %s — is the Flask server running?", api_base
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        log.error("API error: %s  body: %s", exc, resp.text[:200])
        return {}


def ingest_batched(
    api_base: str,
    items: list[dict],
    dataset_name: str,
    limit: int,
) -> int:
    """Send *items* to the API in batches. Returns total chunks added."""
    items = items[:limit]
    total_added = 0
    log.info("Ingesting %d items from '%s' in batches of %d …", len(items), dataset_name, BATCH_SIZE)

    for i in tqdm(range(0, len(items), BATCH_SIZE), desc=f"  ↳ {dataset_name}"):
        batch = items[i : i + BATCH_SIZE]
        result = post_batch(api_base, batch, dataset_name)
        added = result.get("chunksAdded", 0)
        total_added += added

    log.info("'%s': %d chunks added → total in index: %s",
             dataset_name, total_added, result.get("totalChunks", "?"))
    return total_added


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_sciq(limit: int) -> list[dict]:
    """
    SciQ: Allen AI science questions dataset.
    Direct link: https://huggingface.co/datasets/allenai/sciq

    We use the 'support' field (source passage) as the ingestion text.
    The question + correct_answer pairs are saved to processed/ for
    later use as evaluation baselines.
    """
    log.info("Downloading SciQ …")
    ds = load_dataset("allenai/sciq", split="train", trust_remote_code=False)
    items: list[dict] = []
    qa_pairs: list[dict] = []

    for row in ds:
        support = (row.get("support") or "").strip()
        if not support or len(support) < 80:
            continue

        items.append({"text": support, "source": "sciq"})
        qa_pairs.append(
            {
                "passage": support,
                "question": row.get("question", ""),
                "answer": row.get("correct_answer", ""),
                "distractors": [
                    row.get("distractor1", ""),
                    row.get("distractor2", ""),
                    row.get("distractor3", ""),
                ],
            }
        )

    # Persist QA pairs for evaluation baseline
    out_path = PROCESSED_DIR / "sciq_qa_pairs.json"
    with open(out_path, "w") as fh:
        json.dump(qa_pairs[:limit], fh, indent=2)
    log.info("SciQ QA pairs saved → %s (%d items)", out_path, min(len(qa_pairs), limit))

    return items[:limit]


def load_squad(limit: int) -> list[dict]:
    """
    SQuAD v2: Stanford Question Answering Dataset.
    Direct link: https://huggingface.co/datasets/rajpurkar/squad_v2

    We ingest the unique *contexts* (Wikipedia passages) so the knowledge
    graph captures the factual content.
    """
    log.info("Downloading SQuAD v2 …")
    ds = load_dataset("rajpurkar/squad_v2", split="train", trust_remote_code=False)

    seen_contexts: set[str] = set()
    items: list[dict] = []
    qa_pairs: list[dict] = []

    for row in ds:
        ctx = (row.get("context") or "").strip()
        if not ctx or ctx in seen_contexts or len(ctx) < 100:
            continue
        seen_contexts.add(ctx)

        title = row.get("title", "wikipedia")
        items.append({"text": ctx, "source": f"squad/{title}"})

        # Collect answerable QA pairs
        answers = row.get("answers", {}).get("text", [])
        if answers:
            qa_pairs.append(
                {
                    "context": ctx,
                    "question": row.get("question", ""),
                    "answer": answers[0],
                    "title": title,
                }
            )

        if len(items) >= limit:
            break

    out_path = PROCESSED_DIR / "squad_qa_pairs.json"
    with open(out_path, "w") as fh:
        json.dump(qa_pairs[:limit], fh, indent=2)
    log.info("SQuAD QA pairs saved → %s (%d items)", out_path, min(len(qa_pairs), limit))

    return items[:limit]


def load_arxiv(limit: int) -> list[dict]:
    """
    arXiv Abstracts (AI/ML papers 2021).
    Direct link: https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021

    We ingest title + abstract as a single passage.
    This gives the knowledge graph rich AI/CS domain content.
    """
    log.info("Downloading arXiv abstracts …")
    try:
        ds = load_dataset(
            "gfissore/arxiv-abstracts-2021",
            split="train",
            trust_remote_code=False,
        )
    except Exception:
        # Fallback: use a smaller, definitely-available dataset
        log.warning("Primary arXiv dataset unavailable — using 'ccdv/arxiv-summarization'")
        ds = load_dataset(
            "ccdv/arxiv-summarization",
            split="train",
            trust_remote_code=False,
        )

    items: list[dict] = []
    for row in ds:
        abstract = (row.get("abstract") or row.get("article") or "").strip()
        title = (row.get("title") or "").strip()
        if not abstract or len(abstract) < 100:
            continue

        # Combine title + abstract for richer embedding context
        text = f"{title}\n\n{abstract}" if title else abstract
        items.append({"text": text, "source": "arxiv"})

        if len(items) >= limit:
            break

    return items


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets and ingest them into the SKE API."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "sciq", "squad", "arxiv"],
        help="Which dataset(s) to ingest (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max items per dataset (default: 200 — good for testing). "
             "Use 1000+ for production-quality graphs.",
    )
    parser.add_argument(
        "--api",
        default=DEFAULT_API,
        help=f"Flask API base URL (default: {DEFAULT_API})",
    )
    args = parser.parse_args()

    # Verify API is reachable
    try:
        resp = requests.get(f"{args.api}/api/health", timeout=5)
        health = resp.json()
        log.info(
            "API health: Flask=%s  FAISS vectors=%d",
            health.get("flask"),
            health.get("faiss", {}).get("totalVectors", 0),
        )
    except requests.exceptions.ConnectionError:
        log.error("Flask API not reachable at %s — start it with: python backend/app.py", args.api)
        sys.exit(1)

    t_start = time.perf_counter()
    total_chunks = 0

    if args.dataset in ("all", "sciq"):
        items = load_sciq(args.limit)
        total_chunks += ingest_batched(args.api, items, "SciQ", args.limit)

    if args.dataset in ("all", "squad"):
        items = load_squad(args.limit)
        total_chunks += ingest_batched(args.api, items, "SQuAD v2", args.limit)

    if args.dataset in ("all", "arxiv"):
        items = load_arxiv(args.limit)
        total_chunks += ingest_batched(args.api, items, "arXiv", args.limit)

    elapsed = time.perf_counter() - t_start
    log.info(
        "Done! %d chunks added across all datasets in %.1fs",
        total_chunks,
        elapsed,
    )
    log.info(
        "\nDirect dataset links for reference:\n"
        "  SciQ:   https://huggingface.co/datasets/allenai/sciq\n"
        "  SQuAD:  https://huggingface.co/datasets/rajpurkar/squad_v2\n"
        "  arXiv:  https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021\n"
    )


if __name__ == "__main__":
    main()
