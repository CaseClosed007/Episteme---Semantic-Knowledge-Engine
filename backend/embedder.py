# backend/embedder.py
# =============================================================================
# Embedding Engine
# ─────────────────
# Responsibilities:
#   1. Load sentence-transformers model onto MPS (Apple Silicon GPU)
#   2. Chunk raw text using a token-aware sliding window
#   3. Batch-encode chunks into 384-dim vectors
#   4. Persist/load a FAISS IndexFlatIP (inner-product ≡ cosine on L2-normed vecs)
#   5. Expose similarity search and metadata retrieval
#
# Design notes:
#   • IndexFlatIP chosen over IndexIVFFlat — at < 1M chunks the exact search is
#     fast enough on M5 and avoids nprobe tuning complexity.
#   • Vectors are L2-normalised before insertion so inner product == cosine sim.
#   • Metadata (chunk text, source file, page, chunk index) is stored alongside
#     FAISS in a JSON sidecar so we never need to rebuild from scratch.
# =============================================================================

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import cfg

log = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single piece of source text with provenance metadata."""
    chunk_id: int
    text: str
    source: str          # filename or dataset name
    page: int = 0        # PDF page number (0 if N/A)
    chunk_idx: int = 0   # position within the source document
    token_count: int = 0


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float         # cosine similarity [0, 1]


@dataclass
class FAISSStore:
    """In-memory state of the FAISS index and its metadata sidecar."""
    index: faiss.Index
    chunks: List[TextChunk]
    # Lookup: faiss row id → chunk_id (they're the same after first build,
    # but tracked explicitly for future incremental updates)
    id_map: dict[int, int] = field(default_factory=dict)


# ── Chunker ────────────────────────────────────────────────────────────────────

class TokenAwareChunker:
    """
    Splits raw text into overlapping chunks using a simple whitespace tokeniser.
    A real production system would use tiktoken; we keep it dependency-light here
    while still being byte-accurate about overlap.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _clean(text: str) -> str:
        """Collapse excessive whitespace and remove control characters."""
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def chunk(self, text: str, source: str, page: int = 0) -> List[TextChunk]:
        """
        Split *text* into overlapping windows.

        Returns a list of TextChunk objects — each carrying its raw text and
        provenance metadata. Empty or very short texts are returned as-is.
        """
        text = self._clean(text)
        if not text:
            return []

        words = text.split()
        if len(words) <= self.chunk_size:
            return [
                TextChunk(
                    chunk_id=-1,  # assigned by the store
                    text=text,
                    source=source,
                    page=page,
                    chunk_idx=0,
                    token_count=len(words),
                )
            ]

        chunks: List[TextChunk] = []
        start = 0
        idx = 0
        step = self.chunk_size - self.overlap  # stride between windows

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            window_words = words[start:end]
            chunks.append(
                TextChunk(
                    chunk_id=-1,
                    text=" ".join(window_words),
                    source=source,
                    page=page,
                    chunk_idx=idx,
                    token_count=len(window_words),
                )
            )
            idx += 1
            if end == len(words):
                break
            start += step

        return chunks


# ── Embedding Engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps SentenceTransformer + FAISS.

    Thread-safety: The encode() call releases the GIL; FAISS index mutations
    are guarded by a simple lock (see app.py for Flask threading context).
    """

    def __init__(self):
        self._store: Optional[FAISSStore] = None
        self._model: Optional[SentenceTransformer] = None
        self.chunker = TokenAwareChunker(cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        """
        Lazy-load the embedding model.

        Device priority: MPS (Apple Silicon GPU) → CPU.
        MPS gives ~4-6× speedup over CPU for batch encoding on M5.
        """
        if self._model is not None:
            return self._model

        device = cfg.EMBEDDING_DEVICE
        # Validate MPS availability at runtime
        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS requested but not available — falling back to CPU")
            device = "cpu"

        log.info("Loading embedding model %s on %s …", cfg.EMBEDDING_MODEL, device)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        log.info(
            "Embedding model loaded in %.2fs  |  dim=%d  |  device=%s",
            time.perf_counter() - t0,
            cfg.EMBEDDING_DIM,
            device,
        )
        return self._model

    # ── FAISS helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_index() -> faiss.Index:
        """
        Create a new FAISS IndexFlatIP (exact inner-product search).
        When vectors are L2-normalised, inner product equals cosine similarity.
        """
        index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
        # Wrap in an IDMap so we can assign custom integer IDs
        id_index = faiss.IndexIDMap(index)
        return id_index

    def _ensure_store(self) -> FAISSStore:
        """Return existing store or create an empty one."""
        if self._store is None:
            self._store = FAISSStore(
                index=self._make_index(),
                chunks=[],
                id_map={},
            )
        return self._store

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, text: str, source: str, page: int = 0) -> List[TextChunk]:
        """
        Chunk *text*, encode all chunks with the embedding model, and insert
        into FAISS.

        Returns the list of TextChunk objects that were stored (with assigned
        chunk_ids). Persists the index to disk after each call.
        """
        store = self._ensure_store()
        model = self._load_model()

        raw_chunks = self.chunker.chunk(text, source=source, page=page)
        if not raw_chunks:
            log.warning("No chunks generated for source='%s'", source)
            return []

        # Assign stable integer IDs (global sequential)
        base_id = len(store.chunks)
        for i, c in enumerate(raw_chunks):
            c.chunk_id = base_id + i

        texts = [c.text for c in raw_chunks]
        log.info("Encoding %d chunks from '%s' …", len(texts), source)
        t0 = time.perf_counter()

        # Batch encode — returns (N, 384) float32 ndarray
        vectors: np.ndarray = model.encode(
            texts,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # ← L2-normalise for cosine similarity
            convert_to_numpy=True,
        )

        log.info(
            "Encoded %d chunks in %.2fs (%.0f chunks/s)",
            len(texts),
            time.perf_counter() - t0,
            len(texts) / (time.perf_counter() - t0),
        )

        # Insert into FAISS
        ids = np.array([c.chunk_id for c in raw_chunks], dtype=np.int64)
        store.index.add_with_ids(vectors, ids)

        # Update metadata
        store.chunks.extend(raw_chunks)
        for c in raw_chunks:
            store.id_map[c.chunk_id] = c.chunk_id

        self._persist()
        return raw_chunks

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Embed *query* and return the top-k most similar chunks.
        Scores are cosine similarities in [0, 1].
        """
        store = self._ensure_store()
        if store.index.ntotal == 0:
            return []

        model = self._load_model()
        q_vec: np.ndarray = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # shape: (1, 384)

        k = min(top_k, store.index.ntotal)
        scores, ids = store.index.search(q_vec, k)  # both (1, k)

        results: List[SearchResult] = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:  # FAISS sentinel for "not found"
                continue
            chunk = store.chunks[fid]
            results.append(SearchResult(chunk=chunk, score=float(score)))

        return results

    def get_all_chunks(self) -> List[TextChunk]:
        """Return every chunk stored in memory."""
        return list(self._ensure_store().chunks)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Batch-encode an arbitrary list of texts. Used by the graph builder
        to compute pairwise cosine similarities without re-fetching from FAISS.
        """
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Write FAISS index and metadata JSON to disk."""
        store = self._store
        if store is None:
            return

        cfg.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(store.index, str(cfg.FAISS_INDEX_PATH))

        # Save metadata sidecar
        meta = [asdict(c) for c in store.chunks]
        with open(cfg.FAISS_META_PATH, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        log.info(
            "Persisted FAISS index (%d vectors) → %s",
            store.index.ntotal,
            cfg.FAISS_INDEX_PATH,
        )

    def load_from_disk(self) -> bool:
        """
        Attempt to restore a previously saved FAISS index from disk.
        Returns True if successful, False if no saved index exists.
        """
        if not cfg.FAISS_INDEX_PATH.exists() or not cfg.FAISS_META_PATH.exists():
            log.info("No existing FAISS index found — starting fresh.")
            return False

        log.info("Restoring FAISS index from %s …", cfg.FAISS_INDEX_PATH)
        index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))

        with open(cfg.FAISS_META_PATH, "r", encoding="utf-8") as fh:
            raw_meta = json.load(fh)

        chunks = [TextChunk(**m) for m in raw_meta]
        id_map = {c.chunk_id: c.chunk_id for c in chunks}

        self._store = FAISSStore(index=index, chunks=chunks, id_map=id_map)
        log.info(
            "Restored %d vectors from disk  |  sources: %s",
            index.ntotal,
            list({c.source for c in chunks}),
        )
        return True

    @property
    def total_chunks(self) -> int:
        return self._ensure_store().index.ntotal


# ── Module-level singleton ─────────────────────────────────────────────────────
# Flask app imports this; model is lazy-loaded on first request.
engine = EmbeddingEngine()
