# backend/app.py
# =============================================================================
# Semantic Knowledge Engine — Flask API
# ────────────────────────────────────────
# Endpoints:
#
#  POST /api/ingest/text        — Ingest raw text string
#  POST /api/ingest/pdf         — Ingest uploaded PDF file
#  POST /api/ingest/dataset     — Ingest pre-formatted JSON dataset chunks
#
#  GET  /api/graph              — Full knowledge graph JSON
#  GET  /api/graph/node/<id>    — Single node detail + subgraph
#  POST /api/graph/rebuild      — Force graph rebuild from all stored chunks
#
#  GET  /api/search?q=...&k=10  — Semantic search over the corpus
#
#  POST /api/assess/question    — Generate question for a node (blocking)
#  GET  /api/assess/question/stream  — SSE: stream question tokens
#  POST /api/assess/evaluate    — Evaluate a user's answer (blocking)
#  GET  /api/assess/evaluate/stream  — SSE: stream evaluation + result
#
#  GET  /api/health             — Service + Ollama health check
#  GET  /api/stats              — Index statistics
#
# All routes return JSON. SSE routes use text/event-stream.
# =============================================================================

from __future__ import annotations

import io
import json
import logging
import threading
import time
from functools import wraps
from typing import Any

import pdfplumber
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from config import cfg
from embedder import EmbeddingEngine
from graph_builder import KnowledgeGraphBuilder
from llm_router import LLMRouter, OllamaError

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = cfg.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

CORS(app, origins=cfg.CORS_ORIGINS, supports_credentials=True)

# ── Shared singletons ─────────────────────────────────────────────────────────
# All singletons are module-level so they persist across requests.
# Thread-safety: embedder.ingest() and graph.build() are protected by _ingest_lock.
_ingest_lock = threading.Lock()

embed_engine = EmbeddingEngine()
graph_builder = KnowledgeGraphBuilder(embed_engine)
llm_router = LLMRouter(embed_engine)

# Attempt to restore a previously saved index on startup
embed_engine.load_from_disk()


# ── Utility decorators ────────────────────────────────────────────────────────

def json_error(message: str, status: int = 400) -> tuple[Any, int]:
    return jsonify({"error": message}), status


def require_json(fn):
    """Decorator: return 415 if request body is not JSON."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return json_error("Request Content-Type must be application/json", 415)
        return fn(*args, **kwargs)
    return wrapper


def sse_event(data: dict | str, event: str = "message") -> str:
    """Format a Server-Sent Event string."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ── Ingest helpers ────────────────────────────────────────────────────────────

def _ingest_and_rebuild(text: str, source: str, page: int = 0) -> dict:
    """
    Thread-safe ingest + graph rebuild.
    Returns stats dict.
    """
    with _ingest_lock:
        chunks = embed_engine.ingest(text, source=source, page=page)
        graph_builder.build()  # rebuild graph with all chunks so far

    return {
        "chunksAdded": len(chunks),
        "totalChunks": embed_engine.total_chunks,
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/ingest/text", methods=["POST"])
@require_json
def ingest_text():
    """
    Ingest a raw text string.

    Request body:
      { "text": "...", "source": "optional-label" }

    Response:
      { "chunksAdded": N, "totalChunks": M, "source": "..." }
    """
    data = request.get_json()
    text = (data.get("text") or "").strip()
    source = (data.get("source") or "manual-input").strip()

    if not text:
        return json_error("'text' field is required and must not be empty.")

    if len(text) < 50:
        return json_error("Text is too short (minimum 50 characters).")

    log.info("Ingesting text  source='%s'  len=%d chars", source, len(text))
    result = _ingest_and_rebuild(text, source)
    return jsonify(result), 201


@app.route("/api/ingest/pdf", methods=["POST"])
def ingest_pdf():
    """
    Ingest an uploaded PDF file.
    Uses pdfplumber for high-quality text extraction (handles tables, columns).

    multipart/form-data fields:
      file  — PDF binary
      source (optional) — override the filename label
    """
    if "file" not in request.files:
        return json_error("No file part in request. Send as 'file' field.")

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return json_error("File must be a PDF (.pdf extension).")

    source = request.form.get("source") or f.filename
    raw_bytes = f.read()

    log.info("Extracting PDF  source='%s'  size=%.1f KB", source, len(raw_bytes) / 1024)
    t0 = time.perf_counter()

    # Extract text page-by-page so we preserve page metadata per chunk
    total_chunks = 0
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue  # skip blank/image-only pages
                with _ingest_lock:
                    chunks = embed_engine.ingest(
                        page_text,
                        source=source,
                        page=page_num,
                    )
                    total_chunks += len(chunks)

        # Rebuild graph once after all pages are ingested
        with _ingest_lock:
            graph_builder.build()

    except Exception as exc:
        log.exception("PDF parsing failed")
        return json_error(f"PDF parsing error: {exc}", 422)

    elapsed = time.perf_counter() - t0
    log.info(
        "PDF ingested: %d chunks in %.2fs  source='%s'", total_chunks, elapsed, source
    )
    return (
        jsonify(
            {
                "chunksAdded": total_chunks,
                "totalChunks": embed_engine.total_chunks,
                "source": source,
                "processingMs": round(elapsed * 1000),
            }
        ),
        201,
    )


@app.route("/api/ingest/dataset", methods=["POST"])
@require_json
def ingest_dataset():
    """
    Ingest a pre-formatted dataset.
    Accepts a list of { "text": "...", "source": "..." } objects.
    Designed to be called by the download_datasets.py script.

    Request body:
      { "items": [{"text": "...", "source": "..."}, ...] }
    """
    data = request.get_json()
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return json_error("'items' must be a non-empty list of {text, source} objects.")

    total_added = 0
    errors = []
    t0 = time.perf_counter()

    for i, item in enumerate(items):
        text = (item.get("text") or "").strip()
        source = (item.get("source") or f"dataset-item-{i}").strip()
        if not text or len(text) < 30:
            continue
        try:
            with _ingest_lock:
                chunks = embed_engine.ingest(text, source=source)
                total_added += len(chunks)
        except Exception as exc:
            errors.append({"item": i, "error": str(exc)})

    # Rebuild graph after batch
    with _ingest_lock:
        graph_builder.build()

    return (
        jsonify(
            {
                "chunksAdded": total_added,
                "totalChunks": embed_engine.total_chunks,
                "errors": errors[:10],  # cap error list
                "processingMs": round((time.perf_counter() - t0) * 1000),
            }
        ),
        201,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/graph", methods=["GET"])
def get_graph():
    """
    Return the full knowledge graph JSON.
    Uses the cached graph if available; rebuilds if the cache is empty.
    """
    cached = graph_builder.cached_graph
    if cached is None:
        chunks = embed_engine.get_all_chunks()
        if not chunks:
            return jsonify({"nodes": [], "links": [], "stats": {"nodes": 0, "edges": 0}})
        cached = graph_builder.build(chunks)

    return jsonify(cached)


@app.route("/api/graph/node/<int:node_id>", methods=["GET"])
def get_node(node_id: int):
    """
    Return detail for a single node plus its 2-hop neighbourhood subgraph.
    Query param: hops=<int> (default 2)
    """
    hops = min(int(request.args.get("hops", 2)), 4)
    cached = graph_builder.cached_graph

    if cached is None:
        return json_error("Graph not yet built. POST to /api/ingest/text first.", 404)

    # Find the node in the cached graph
    node_data = next((n for n in cached["nodes"] if n["id"] == node_id), None)
    if node_data is None:
        return json_error(f"Node {node_id} not found.", 404)

    subgraph = graph_builder.get_subgraph_json(node_id, hops=hops)
    prerequisites = graph_builder.get_prerequisites(node_id, depth=2)

    return jsonify(
        {
            "node": node_data,
            "subgraph": subgraph,
            "prerequisites": prerequisites,
        }
    )


@app.route("/api/graph/rebuild", methods=["POST"])
def rebuild_graph():
    """Force a full graph rebuild from all stored chunks."""
    chunks = embed_engine.get_all_chunks()
    if not chunks:
        return json_error("No chunks in the index — ingest some content first.", 404)

    with _ingest_lock:
        graph = graph_builder.build(chunks)

    return jsonify({"rebuilt": True, "stats": graph["stats"]})


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/search", methods=["GET"])
def search():
    """
    Semantic search over the ingested corpus.
    Query params: q (required), k (optional, default 10)
    """
    query = (request.args.get("q") or "").strip()
    if not query:
        return json_error("Query parameter 'q' is required.")

    k = min(int(request.args.get("k", 10)), 50)
    results = embed_engine.search(query, top_k=k)

    return jsonify(
        {
            "query": query,
            "results": [
                {
                    "chunkId": r.chunk.chunk_id,
                    "text": r.chunk.text,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "score": round(r.score, 4),
                }
                for r in results
            ],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/assess/question", methods=["POST"])
@require_json
def generate_question():
    """
    Generate a comprehension question for a graph node.

    Request body:
      { "nodeId": 42 }         OR
      { "text": "raw passage" }

    The LLM is given both the node text and prerequisite chunk texts
    so it can ask a question that differentiates this node from its
    prerequisites (avoiding trivially easy questions).
    """
    data = request.get_json()
    node_id = data.get("nodeId")
    raw_text = (data.get("text") or "").strip()

    if node_id is not None:
        # Resolve from graph cache
        cached = graph_builder.cached_graph
        if cached is None:
            return json_error("Graph not built yet.", 404)
        node_data = next(
            (n for n in cached["nodes"] if n["id"] == int(node_id)), None
        )
        if node_data is None:
            return json_error(f"Node {node_id} not found in graph.", 404)
        node_text = node_data["text"]

        # Gather prerequisite context
        prereqs = graph_builder.get_prerequisites(int(node_id), depth=2)
        all_chunks = {c.chunk_id: c for c in embed_engine.get_all_chunks()}
        context_texts = [
            all_chunks[pid].text for pid in prereqs if pid in all_chunks
        ][:3]

    elif raw_text:
        node_text = raw_text
        context_texts = []
    else:
        return json_error("Either 'nodeId' or 'text' must be provided.")

    result = llm_router.generate_question(node_text, context_texts)
    return jsonify(result)


@app.route("/api/assess/question/stream", methods=["GET"])
def stream_question():
    """
    SSE endpoint: streams question generation tokens.
    Query params: nodeId=<int>  OR  text=<str>
    """
    node_id = request.args.get("nodeId")
    raw_text = (request.args.get("text") or "").strip()

    if node_id is not None:
        cached = graph_builder.cached_graph
        if cached is None:
            return json_error("Graph not built yet.", 404)
        node_data = next(
            (n for n in cached["nodes"] if n["id"] == int(node_id)), None
        )
        if node_data is None:
            return json_error(f"Node {node_id} not found.", 404)
        node_text = node_data["text"]
        prereqs = graph_builder.get_prerequisites(int(node_id), depth=2)
        all_chunks = {c.chunk_id: c for c in embed_engine.get_all_chunks()}
        context_texts = [
            all_chunks[pid].text for pid in prereqs if pid in all_chunks
        ][:3]
    elif raw_text:
        node_text = raw_text
        context_texts = []
    else:
        return json_error("Either 'nodeId' or 'text' query param required.")

    def generate():
        yield sse_event({"type": "start"}, event="start")
        try:
            for token in llm_router.stream_question(node_text, context_texts):
                yield sse_event({"type": "token", "content": token}, event="token")
        except OllamaError as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="error")
        yield sse_event({"type": "done"}, event="done")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.route("/api/assess/evaluate", methods=["POST"])
@require_json
def evaluate_response():
    """
    Evaluate a user's free-text answer.

    Request body:
      {
        "question": "...",
        "nodeId": 42,          // to resolve reference text from graph
        "userAnswer": "..."
      }
    """
    data = request.get_json()
    question = (data.get("question") or "").strip()
    node_id = data.get("nodeId")
    user_answer = (data.get("userAnswer") or "").strip()

    if not question:
        return json_error("'question' is required.")
    if not user_answer:
        return json_error("'userAnswer' is required.")
    if len(user_answer) < 10:
        return json_error("Answer is too short (minimum 10 characters).")

    # Resolve reference text
    if node_id is not None:
        cached = graph_builder.cached_graph
        if cached is None:
            return json_error("Graph not built yet.", 404)
        node_data = next(
            (n for n in cached["nodes"] if n["id"] == int(node_id)), None
        )
        if node_data is None:
            return json_error(f"Node {node_id} not found.", 404)
        reference_text = node_data["text"]
    else:
        reference_text = (data.get("referenceText") or "").strip()
        if not reference_text:
            return json_error("Either 'nodeId' or 'referenceText' is required.")

    result = llm_router.evaluate_response(question, reference_text, user_answer)
    return jsonify(result)


@app.route("/api/assess/evaluate/stream", methods=["GET"])
def stream_evaluation():
    """
    SSE endpoint: streams the LLM evaluation then sends the final JSON result.
    Query params: question, nodeId, userAnswer (all URL-encoded)
    """
    question = (request.args.get("question") or "").strip()
    node_id = request.args.get("nodeId")
    user_answer = (request.args.get("userAnswer") or "").strip()

    if not question or not user_answer:
        return json_error("'question' and 'userAnswer' query params required.")

    if node_id is not None:
        cached = graph_builder.cached_graph
        node_data = next(
            (n for n in (cached["nodes"] if cached else []) if n["id"] == int(node_id)),
            None,
        )
        reference_text = node_data["text"] if node_data else ""
    else:
        reference_text = (request.args.get("referenceText") or "").strip()

    if not reference_text:
        return json_error("Cannot resolve reference text for evaluation.", 404)

    def generate():
        yield sse_event({"type": "start"}, event="start")
        try:
            eval_buffer = ""
            for token in llm_router.stream_evaluation(question, reference_text, user_answer):
                # The router appends __EVAL_RESULT__<json> as a final sentinel
                if "__EVAL_RESULT__" in token:
                    parts = token.split("__EVAL_RESULT__", 1)
                    if parts[0]:
                        yield sse_event(
                            {"type": "token", "content": parts[0]}, event="token"
                        )
                    try:
                        extra = json.loads(parts[1])
                        yield sse_event(
                            {"type": "result", "data": extra}, event="result"
                        )
                    except json.JSONDecodeError:
                        pass
                else:
                    yield sse_event({"type": "token", "content": token}, event="token")
        except OllamaError as exc:
            yield sse_event({"type": "error", "message": str(exc)}, event="error")
        yield sse_event({"type": "done"}, event="done")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & STATS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """Combined health check — Flask + Ollama + FAISS."""
    ollama_status = llm_router.health_check()
    return jsonify(
        {
            "flask": "online",
            "ollama": ollama_status,
            "faiss": {
                "totalVectors": embed_engine.total_chunks,
                "indexPath": str(cfg.FAISS_INDEX_PATH),
            },
            "graph": (
                graph_builder.cached_graph["stats"]
                if graph_builder.cached_graph
                else None
            ),
            "embeddingModel": cfg.EMBEDDING_MODEL,
            "embeddingDevice": cfg.EMBEDDING_DEVICE,
        }
    )


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return index and graph statistics."""
    graph_stats = (
        graph_builder.cached_graph["stats"] if graph_builder.cached_graph else {}
    )
    chunks = embed_engine.get_all_chunks()
    sources = {}
    for c in chunks:
        sources[c.source] = sources.get(c.source, 0) + 1

    return jsonify(
        {
            "totalChunks": embed_engine.total_chunks,
            "sources": sources,
            "graph": graph_stats,
            "config": {
                "chunkSize": cfg.CHUNK_SIZE,
                "chunkOverlap": cfg.CHUNK_OVERLAP,
                "similarityThreshold": cfg.SIMILARITY_THRESHOLD,
                "maxEdgesPerNode": cfg.MAX_GRAPH_EDGES_PER_NODE,
                "embeddingModel": cfg.EMBEDDING_MODEL,
                "llmModel": cfg.LLM_MODEL,
            },
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info(
        "Starting Semantic Knowledge Engine API on port %d  (debug=%s)",
        cfg.FLASK_PORT,
        cfg.FLASK_DEBUG,
    )
    # Use threaded=True so each request gets its own thread and SSE connections
    # don't block each other.  In production replace with gunicorn + eventlet.
    app.run(
        host="0.0.0.0",
        port=cfg.FLASK_PORT,
        debug=cfg.FLASK_DEBUG,
        threaded=True,
        use_reloader=False,  # disable reloader to prevent double model loading
    )
