#!/usr/bin/env bash
# =============================================================================
# Autonomous Semantic Knowledge Engine — Apple Silicon (M5) Setup Script
# Run: chmod +x setup.sh && ./setup.sh
# =============================================================================
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[✓]${RESET} $1"; }
info() { echo -e "${CYAN}[→]${RESET} $1"; }
warn() { echo -e "${YELLOW}[!]${RESET} $1"; }
die()  { echo -e "${RED}[✗]${RESET} $1"; exit 1; }

echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo -e "║   Semantic Knowledge Engine — M5 MacBook Air Setup          ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${RESET}"

# ── 0. Verify Apple Silicon ───────────────────────────────────────────────────
ARCH=$(uname -m)
[[ "$ARCH" == "arm64" ]] || die "This script requires Apple Silicon (arm64). Got: $ARCH"
log "Apple Silicon confirmed: $ARCH"

# ── 1. Homebrew ────────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for the current session (Apple Silicon path)
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
log "Homebrew ready"

# ── 2. System dependencies ────────────────────────────────────────────────────
info "Installing system dependencies..."
brew install cmake libomp pkg-config hdf5 poppler || true
log "System deps installed"

# ── 3. Python 3.11 via pyenv (ensures clean arm64 build) ─────────────────────
if ! command -v pyenv &>/dev/null; then
  info "Installing pyenv..."
  brew install pyenv
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
fi

PYTHON_VERSION="3.11.9"
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
  info "Installing Python $PYTHON_VERSION (native arm64)..."
  PYTHON_CONFIGURE_OPTS="--enable-optimizations" pyenv install "$PYTHON_VERSION"
fi
pyenv local "$PYTHON_VERSION"
log "Python $PYTHON_VERSION set as local version"

# ── 4. Python virtual environment ─────────────────────────────────────────────
VENV_DIR="$(pwd)/backend/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at $VENV_DIR..."
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
log "Virtual environment activated"

# ── 5. PyTorch with MPS backend (Apple Silicon native) ───────────────────────
# NOTE: We use the official Apple-recommended install path for MPS acceleration.
# Do NOT use conda-forge torch; it may disable MPS.
info "Installing PyTorch (MPS-enabled for M5)..."
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cpu
# Verify MPS availability
python - <<'PYCHECK'
import torch
assert torch.backends.mps.is_available(), "MPS not available — check macOS 12.3+ and Xcode CLI tools"
print(f"  PyTorch {torch.__version__} | MPS available ✓")
PYCHECK
log "PyTorch with MPS backend verified"

# ── 6. Python dependencies ─────────────────────────────────────────────────────
info "Installing Python packages..."
pip install \
  flask==3.0.3 \
  flask-cors==4.0.1 \
  flask-socketio==5.3.6 \
  sentence-transformers==3.0.1 \
  faiss-cpu==1.8.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  networkx==3.3 \
  pypdf2==3.0.1 \
  pdfplumber==0.11.4 \
  python-dotenv==1.0.1 \
  requests==2.32.3 \
  tqdm==4.66.4 \
  datasets==2.20.0 \
  huggingface-hub==0.24.2 \
  tiktoken==0.7.0 \
  gunicorn==22.0.0 \
  eventlet==0.36.1
log "Python packages installed"

# ── 7. Ollama (local LLM runner with Metal GPU acceleration) ──────────────────
if ! command -v ollama &>/dev/null; then
  info "Installing Ollama (Metal-accelerated LLM runner)..."
  brew install ollama
fi
log "Ollama installed: $(ollama --version 2>/dev/null || echo 'installed')"

# Start Ollama server in background if not running
if ! pgrep -x "ollama" > /dev/null; then
  info "Starting Ollama server..."
  ollama serve &>/tmp/ollama.log &
  sleep 3
fi

# Pull quantized Llama 3 8B (Q4_K_M — best quality/speed ratio for 16GB unified memory)
info "Pulling llama3.1:8b-instruct-q4_K_M (~4.7GB, optimized for M5 16GB)..."
ollama pull llama3.1:8b-instruct-q4_K_M
log "Llama 3.1 8B Q4_K_M pulled"

# Pull nomic-embed-text as a fallback embedding model
info "Pulling nomic-embed-text for embedding fallback..."
ollama pull nomic-embed-text
log "nomic-embed-text pulled"

# ── 8. Node.js & Frontend ─────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  info "Installing Node.js via nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
  nvm install 20
  nvm use 20
fi
log "Node.js $(node --version) | npm $(npm --version)"

info "Installing frontend dependencies..."
cd frontend && npm install && cd ..
log "Frontend dependencies installed"

# ── 9. Create data directories ────────────────────────────────────────────────
mkdir -p data/{raw,processed,embeddings} backend/logs
log "Data directories created"

# ── 10. Environment file ───────────────────────────────────────────────────────
if [[ ! -f backend/.env ]]; then
  cat > backend/.env <<'ENV'
# ── Backend Configuration ─────────────────────────────────────────────
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5001

# ── Embedding Model ────────────────────────────────────────────────────
# Runs natively on MPS — no CUDA needed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=mps
EMBEDDING_BATCH_SIZE=64
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# ── FAISS Index ────────────────────────────────────────────────────────
FAISS_INDEX_PATH=data/embeddings/knowledge.index
FAISS_META_PATH=data/embeddings/metadata.json

# ── LLM (Ollama) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b-instruct-q4_K_M
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# ── Graph ─────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD=0.45
MAX_GRAPH_EDGES_PER_NODE=5

# ── Security ──────────────────────────────────────────────────────────
SECRET_KEY=dev-secret-change-in-production
CORS_ORIGINS=http://localhost:5173
ENV
  log ".env created"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════"
echo -e "  Setup complete! Start the system:"
echo -e ""
echo -e "  Terminal 1 (Backend):   cd backend && source .venv/bin/activate"
echo -e "                          python app.py"
echo -e ""
echo -e "  Terminal 2 (Frontend):  cd frontend && npm run dev"
echo -e ""
echo -e "  Terminal 3 (Datasets):  source backend/.venv/bin/activate"
echo -e "                          python scripts/download_datasets.py"
echo -e "═══════════════════════════════════════════════════════════════${RESET}"
