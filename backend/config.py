# backend/config.py
# =============================================================================
# Centralised configuration — loads from .env with typed defaults.
# All tuneable constants live here so app.py stays declarative.
# =============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (one level above this file)
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")


class Config:
    # ── Flask ─────────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", 5001))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "1") == "1"
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" as fallback
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "mps")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))
    EMBEDDING_DIM: int = 384  # fixed for all-MiniLM-L6-v2

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))      # tokens
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))  # tokens

    # ── FAISS Persistence ──────────────────────────────────────────────────────
    FAISS_INDEX_PATH: Path = ROOT_DIR / os.getenv(
        "FAISS_INDEX_PATH", "data/embeddings/knowledge.index"
    )
    FAISS_META_PATH: Path = ROOT_DIR / os.getenv(
        "FAISS_META_PATH", "data/embeddings/metadata.json"
    )

    # ── Knowledge Graph ────────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))
    MAX_GRAPH_EDGES_PER_NODE: int = int(os.getenv("MAX_GRAPH_EDGES_PER_NODE", 5))

    # ── LLM (Ollama) ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))

    # ── Data Paths ────────────────────────────────────────────────────────────
    RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
    EMBEDDINGS_DIR: Path = ROOT_DIR / "data" / "embeddings"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they don't exist."""
        for d in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EMBEDDINGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
Config.ensure_dirs()
