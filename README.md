# Autonomous Semantic Knowledge Engine & Generative Assessor
### Runs 100% locally on Apple Silicon (M5) · Metal/MPS · Ollama · FAISS

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  React Frontend (Vite + react-force-graph-2d)                │   │
│  │                                                              │   │
│  │  KnowledgeContext ──── KnowledgeGraph (canvas/WebGL)        │   │
│  │  AssessmentContext ─── AssessmentPanel (SSE streaming)      │   │
│  │                                                              │   │
│  │  useReducer + useCallback + useMemo + useContext + useEffect │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ REST + SSE                                 │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │  Flask API (app.py)                                          │   │
│  │                                                              │   │
│  │  /api/ingest/*  →  EmbeddingEngine (embedder.py)            │   │
│  │                     ├── SentenceTransformers on MPS         │   │
│  │                     └── FAISS IndexIDMap (cosine, L2-norm)  │   │
│  │                                                              │   │
│  │  /api/graph/*   →  KnowledgeGraphBuilder (graph_builder.py) │   │
│  │                     ├── Pairwise cosine sim (NumPy BLAS)    │   │
│  │                     ├── NetworkX DiGraph                    │   │
│  │                     └── PageRank + community detection      │   │
│  │                                                              │   │
│  │  /api/assess/*  →  LLMRouter (llm_router.py)               │   │
│  │                     ├── Ollama /api/chat (Metal GPU)        │   │
│  │                     ├── SSE token streaming                 │   │
│  │                     └── Dual-layer eval (embed + LLM)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Local Models:                                                       │
│  • all-MiniLM-L6-v2  (384-dim, runs on MPS)                        │
│  • llama3.1:8b-instruct-q4_K_M  (4.7GB, Metal GPU via Ollama)      │
│                                                                      │
│  Storage:  FAISS binary index + JSON metadata sidecar               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Step 1 — Run setup (one-time)
```bash
chmod +x setup.sh && ./setup.sh
```
This installs: pyenv, Python 3.11, PyTorch MPS, all Python deps, Ollama,
pulls `llama3.1:8b-instruct-q4_K_M` (~4.7GB), Node.js 20, and npm packages.

### Step 2 — Start the backend
```bash
cd backend
source .venv/bin/activate
python app.py
# → http://localhost:5001
```

### Step 3 — Start the frontend
```bash
cd frontend
npm run dev
# → http://localhost:5173
```

### Step 4 — Load test data (optional but recommended)
```bash
source backend/.venv/bin/activate
# Load 200 items from each dataset (~3 min on M5, builds a rich graph)
python scripts/download_datasets.py --dataset all --limit 200

# Or load just SciQ for a quick test:
python scripts/download_datasets.py --dataset sciq --limit 50
```

---

## File Structure

```
ske/
├── setup.sh                          # One-command environment setup
├── backend/
│   ├── app.py                        # Flask API — all endpoints
│   ├── config.py                     # Typed config from .env
│   ├── embedder.py                   # SentenceTransformers + FAISS
│   ├── graph_builder.py              # Cosine sim → NetworkX DAG
│   ├── llm_router.py                 # Ollama client + SSE streaming
│   ├── requirements.txt
│   └── .env                          # Created by setup.sh
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # Root layout + TopBar
│   │   ├── main.jsx                  # React entry point
│   │   ├── styles.css                # Dark industrial theme
│   │   ├── contexts/
│   │   │   ├── KnowledgeContext.jsx  # Graph + ingest state
│   │   │   └── AssessmentContext.jsx # SSE assessment state machine
│   │   └── components/
│   │       ├── KnowledgeGraph.jsx    # Force-directed canvas graph
│   │       ├── AssessmentPanel.jsx   # Phase-driven assessment UI
│   │       └── IngestPanel.jsx       # Text/PDF ingest form
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── scripts/
│   └── download_datasets.py          # HuggingFace dataset downloader
└── data/
    ├── raw/                          # Downloaded source files
    ├── processed/                    # Formatted QA pairs
    └── embeddings/                   # FAISS index + metadata JSON
```

---

## Dataset Links

| Dataset | Link | Content | Use Case |
|---------|------|---------|----------|
| **SciQ** | https://huggingface.co/datasets/allenai/sciq | 13,679 science QA + passages | Question generation testing |
| **SQuAD v2** | https://huggingface.co/datasets/rajpurkar/squad_v2 | 150K Wikipedia QA pairs | Semantic eval grading |
| **arXiv Abstracts** | https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021 | AI/ML paper abstracts | Domain knowledge graph |

---

## M5 Performance Notes

| Operation | Expected Time (M5 16GB) |
|-----------|------------------------|
| Model load (first request) | ~3–5s |
| Embed 100 chunks (MPS) | ~0.8s |
| FAISS search (10k vectors) | <5ms |
| Graph build (500 nodes) | ~2s |
| LLM question gen (Q4_K_M) | ~3–6s |
| LLM evaluation | ~5–10s |

**Memory usage:** ~6GB unified memory at full load (model + index + graph).
Leaves ~10GB free for macOS on a 16GB M5 Air.

---

## API Reference

### Ingest
```
POST /api/ingest/text       { text, source }
POST /api/ingest/pdf        multipart: file, source
POST /api/ingest/dataset    { items: [{text, source}] }
```

### Graph
```
GET  /api/graph                     → full graph JSON
GET  /api/graph/node/<id>?hops=2    → node detail + subgraph
POST /api/graph/rebuild             → force rebuild
```

### Assessment
```
POST /api/assess/question           { nodeId }  → { question }
GET  /api/assess/question/stream    ?nodeId=N   → SSE tokens
POST /api/assess/evaluate           { question, nodeId, userAnswer }
GET  /api/assess/evaluate/stream    ?question=...&nodeId=N&userAnswer=...
```

### Utility
```
GET  /api/search?q=...&k=10         → semantic search results
GET  /api/health                    → service status
GET  /api/stats                     → index statistics
```

---

## Tuning `.env` for your M5

```bash
# More aggressive chunking for technical PDFs
CHUNK_SIZE=256
CHUNK_OVERLAP=32

# Lower threshold = denser graph (more edges)
SIMILARITY_THRESHOLD=0.35

# Raise for deeper prerequisite chains
MAX_GRAPH_EDGES_PER_NODE=8

# Switch to Llama 3.2 3B for faster responses on 8GB M5 Air
LLM_MODEL=llama3.2:3b-instruct-q4_K_M
```
