# backend/graph_builder.py
# =============================================================================
# Semantic Knowledge Graph Builder
# ──────────────────────────────────
# Transforms a flat list of text chunks into a directed semantic dependency
# graph (DAG) by:
#
#   1. Computing pairwise cosine similarities between all chunk embeddings.
#   2. Applying a similarity threshold to create "prerequisite" edges
#      (lower chunk index → higher chunk index when similarity > threshold).
#   3. Pruning to MAX_GRAPH_EDGES_PER_NODE top neighbours per node to keep
#      the graph navigable.
#   4. Extracting a "topic label" for each node via the first N words of the
#      chunk (can be replaced with an LLM-based title extraction in prod).
#   5. Serialising to a JSON structure compatible with react-force-graph-2d.
#
# Output JSON schema:
# {
#   "nodes": [{"id": 0, "label": "...", "source": "...", "text": "...", ...}],
#   "links": [{"source": 0, "target": 3, "weight": 0.72, "type": "prerequisite"}]
# }
#
# Performance note:
#   Pairwise cosine similarity is O(N²). For N < 5000 chunks this is fine on
#   M5 with NumPy (vectorised BLAS). For N > 50k, consider approximate nearest
#   neighbour (FAISS HNSW) instead of the dense matrix approach.
# =============================================================================

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from config import cfg
from embedder import EmbeddingEngine, TextChunk

log = logging.getLogger(__name__)


# ── Graph Node / Edge types ────────────────────────────────────────────────────

def _make_label(text: str, max_words: int = 8) -> str:
    """Extract a short human-readable label from the start of a chunk."""
    words = re.sub(r"\s+", " ", text).split()[:max_words]
    label = " ".join(words)
    if len(label) < len(text[:60]):
        label += "…"
    return label


def _topic_weight(chunk: TextChunk) -> float:
    """
    Simple heuristic importance score for a chunk — used for node sizing in
    the frontend.  Longer chunks in earlier positions get higher weight.
    Scale: [0, 1].
    """
    max_tokens = cfg.CHUNK_SIZE
    return min(chunk.token_count / max_tokens, 1.0)


# ── Graph Builder ──────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    Builds and maintains a semantic knowledge graph over all ingested chunks.

    The graph is rebuilt from scratch on each call to `build()`.  For a
    production system with millions of chunks you would implement an
    incremental update strategy (add new nodes, recompute edges only for
    affected nodes).
    """

    def __init__(self, embed_engine: EmbeddingEngine):
        self._engine = embed_engine
        self._graph: Optional[nx.DiGraph] = None
        self._json_cache: Optional[dict] = None

    # ── Core build ────────────────────────────────────────────────────────────

    def build(self, chunks: Optional[List[TextChunk]] = None) -> dict:
        """
        Build the knowledge graph from *chunks* (defaults to all ingested chunks).

        Returns the graph as a dict ready to be JSON-serialised and consumed by
        the React frontend.
        """
        if chunks is None:
            chunks = self._engine.get_all_chunks()

        if not chunks:
            log.warning("No chunks available — returning empty graph.")
            return {"nodes": [], "links": [], "stats": {"nodes": 0, "edges": 0}}

        log.info("Building knowledge graph for %d chunks …", len(chunks))
        t0 = time.perf_counter()

        # ── 1. Encode all chunks ──────────────────────────────────────────────
        texts = [c.text for c in chunks]
        vectors = self._engine.encode_texts(texts)   # (N, 384) float32, L2-normed

        # ── 2. Pairwise cosine similarity ─────────────────────────────────────
        # Since vectors are L2-normalised, cosine_sim = dot product.
        # Result: NxN matrix, values in [-1, 1] (realistically [0, 1] for text).
        sim_matrix: np.ndarray = vectors @ vectors.T  # (N, N)
        np.fill_diagonal(sim_matrix, 0.0)             # no self-loops

        # ── 3. Build NetworkX directed graph ──────────────────────────────────
        G = nx.DiGraph()

        # Add nodes
        for chunk in chunks:
            G.add_node(
                chunk.chunk_id,
                label=_make_label(chunk.text),
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                chunk_idx=chunk.chunk_idx,
                weight=_topic_weight(chunk),
            )

        # Add edges: iterate over upper triangle only (avoid duplicate edges)
        threshold = cfg.SIMILARITY_THRESHOLD
        max_edges = cfg.MAX_GRAPH_EDGES_PER_NODE

        # Collect candidate edges per node, then prune to top-k
        # using a dict: node_id → [(score, target_id), ...]
        out_edges: dict[int, list[tuple[float, int]]] = {
            c.chunk_id: [] for c in chunks
        }

        n = len(chunks)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    src_id = chunks[i].chunk_id
                    tgt_id = chunks[j].chunk_id
                    # Directed: lower position in corpus → higher (prerequisite)
                    out_edges[src_id].append((score, tgt_id))

        # Prune each node's outgoing edges to top max_edges by score
        edges_added = 0
        for src_id, candidates in out_edges.items():
            if not candidates:
                continue
            # Sort descending by score, take top-k
            top = sorted(candidates, key=lambda x: x[0], reverse=True)[:max_edges]
            for score, tgt_id in top:
                G.add_edge(
                    src_id,
                    tgt_id,
                    weight=round(score, 4),
                    edge_type="prerequisite",
                )
                edges_added += 1

        self._graph = G

        # ── 4. Compute graph-level analytics ──────────────────────────────────
        try:
            # PageRank gives a relevance/centrality score per node
            pr = nx.pagerank(G, alpha=0.85, weight="weight")
        except nx.exception.NetworkXError:
            pr = {n: 1.0 / len(G.nodes) for n in G.nodes}

        # Community detection via weakly connected components
        # (gives each node a "cluster" colour for the frontend)
        components = list(nx.weakly_connected_components(G))
        node_cluster: dict[int, int] = {}
        for cluster_id, component in enumerate(components):
            for node_id in component:
                node_cluster[node_id] = cluster_id

        # ── 5. Serialise ──────────────────────────────────────────────────────
        nodes_json = []
        for node_id, attrs in G.nodes(data=True):
            nodes_json.append(
                {
                    "id": node_id,
                    "label": attrs["label"],
                    "text": attrs["text"],
                    "source": attrs["source"],
                    "page": attrs["page"],
                    "chunkIdx": attrs["chunk_idx"],
                    "weight": round(attrs["weight"], 4),
                    "pagerank": round(pr.get(node_id, 0.0), 6),
                    "cluster": node_cluster.get(node_id, 0),
                    "degree": G.degree(node_id),
                    "inDegree": G.in_degree(node_id),
                    "outDegree": G.out_degree(node_id),
                }
            )

        links_json = [
            {
                "source": u,
                "target": v,
                "weight": data["weight"],
                "type": data["edge_type"],
            }
            for u, v, data in G.edges(data=True)
        ]

        elapsed = time.perf_counter() - t0
        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "clusters": len(components),
            "buildTimeMs": round(elapsed * 1000),
            "avgDegree": round(
                sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
            ),
        }

        log.info(
            "Graph built: %d nodes, %d edges, %d clusters  (%.0fms)",
            stats["nodes"],
            stats["edges"],
            stats["clusters"],
            stats["buildTimeMs"],
        )

        self._json_cache = {"nodes": nodes_json, "links": links_json, "stats": stats}
        return self._json_cache

    # ── Prerequisite chain ────────────────────────────────────────────────────

    def get_prerequisites(self, node_id: int, depth: int = 2) -> List[int]:
        """
        Return the prerequisite node IDs for *node_id* up to *depth* hops.
        Used by the assessment engine to suggest what to study before tackling
        a given node.
        """
        if self._graph is None or node_id not in self._graph:
            return []

        predecessors: set[int] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[int] = set()
            for n in frontier:
                preds = set(self._graph.predecessors(n))
                new = preds - predecessors - {node_id}
                predecessors.update(new)
                next_frontier.update(new)
            frontier = next_frontier
            if not frontier:
                break

        return list(predecessors)

    def get_subgraph_json(self, node_id: int, hops: int = 2) -> dict:
        """
        Return a small JSON subgraph centred on *node_id* — used by the
        frontend to render a focused neighbourhood view during assessment.
        """
        if self._graph is None:
            return {"nodes": [], "links": []}

        # BFS neighbourhood
        neighbors: set[int] = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_f: set[int] = set()
            for n in frontier:
                next_f.update(self._graph.successors(n))
                next_f.update(self._graph.predecessors(n))
            next_f -= neighbors
            neighbors.update(next_f)
            frontier = next_f

        sub = self._graph.subgraph(neighbors)
        if self._json_cache is None:
            return {"nodes": [], "links": []}

        node_set = set(neighbors)
        nodes = [n for n in self._json_cache["nodes"] if n["id"] in node_set]
        links = [
            lk
            for lk in self._json_cache["links"]
            if lk["source"] in node_set and lk["target"] in node_set
        ]
        return {"nodes": nodes, "links": links}

    @property
    def cached_graph(self) -> Optional[dict]:
        return self._json_cache
