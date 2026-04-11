// frontend/src/App.jsx
// =============================================================================
// Root application component.
// Layout:
//
//   ┌─────────────────────────────────────────────────────┐
//   │  TopBar (health status + stats)                     │
//   ├──────────────────┬──────────────────────────────────┤
//   │  Left sidebar    │  Centre: KnowledgeGraph          │
//   │  ─ IngestPanel   │                                  │
//   │  ─ SearchPanel   ├──────────────────────────────────┤
//   │                  │  Right: AssessmentPanel          │
//   └──────────────────┴──────────────────────────────────┘
//
// All panels communicate exclusively through context — no prop drilling.
// =============================================================================

import { useCallback, useEffect, useRef, useState } from "react";
import AssessmentPanel from "./components/AssessmentPanel";
import IngestPanel from "./components/IngestPanel";
import KnowledgeGraph from "./components/KnowledgeGraph";
import { AssessmentProvider,useAssessment } from "./contexts/AssessmentContext";
import { KnowledgeProvider, useKnowledge } from "./contexts/KnowledgeContext";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:5001";

// ── Search panel (inline — small enough not to warrant its own file) ──────────

function SearchPanel() {
  const { search, searchResults, searchLoading, searchQuery, selectNode } =
    useKnowledge();
  const [query, setQuery] = useState("");
  const timerRef = useRef(null);

  const handleChange = useCallback(
    (e) => {
      const val = e.target.value;
      setQuery(val);
      clearTimeout(timerRef.current);
      if (val.trim().length >= 3) {
        timerRef.current = setTimeout(() => search(val.trim()), 400);
      }
    },
    [search]
  );

  useEffect(() => () => clearTimeout(timerRef.current), []);

  return (
    <div className="search-panel">
      <input
        className="search-input"
        placeholder="Semantic search…"
        value={query}
        onChange={handleChange}
      />
      {searchLoading && <div className="search-spinner" />}
      <div className="search-results">
        {searchResults.slice(0, 8).map((r) => (
          <button
            key={r.chunkId}
            className="search-result-item"
            onClick={() => selectNode(r.chunkId)}
          >
            <span className="result-score">{(r.score * 100).toFixed(0)}%</span>
            <span className="result-text">{r.text.slice(0, 80)}…</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Top bar ───────────────────────────────────────────────────────────────────

function TopBar() {
  const { health, graph, checkHealth, fetchGraph } = useKnowledge();
  const ollamaOnline = health?.ollama?.status === "online";
  const modelLoaded  = health?.ollama?.target_loaded;

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="brand-mark">◈</span>
        <span className="brand-name">Semantic Knowledge Engine</span>
        <span className="brand-sub">M5 · Local · Private</span>
      </div>

      <div className="topbar-status">
        <span
          className={`status-pill ${ollamaOnline ? "status-online" : "status-offline"}`}
          title={`Ollama: ${health?.ollama?.status ?? "unknown"}`}
        >
          {ollamaOnline ? "●" : "○"} Ollama
        </span>

        <span
          className={`status-pill ${modelLoaded ? "status-online" : "status-warning"}`}
          title={health?.ollama?.target_model}
        >
          {modelLoaded ? "●" : "◌"} {health?.ollama?.target_model?.split(":")[0] ?? "LLM"}
        </span>

        <span className="status-pill status-neutral" title="Embedded vectors">
          {health?.faiss?.totalVectors ?? "–"} vectors
        </span>

        {graph && (
          <span className="status-pill status-neutral">
            {graph.stats.nodes}N / {graph.stats.edges}E
          </span>
        )}

        <button className="btn-icon" onClick={checkHealth} title="Refresh health">↺</button>
        <button className="btn-icon" onClick={fetchGraph} title="Refresh graph">⬡</button>
      </div>
    </header>
  );
}

// ── Node detail tooltip ───────────────────────────────────────────────────────

function NodeDetail() {
  const { selectedNodeData, selectedNodeId } = useKnowledge();
  const { generateQuestion } = useAssessment();

  if (!selectedNodeData || !selectedNodeId) return null;
  const { node } = selectedNodeData;

  return (
    <div className="node-detail-card">
      <div className="node-detail-header">
        <span className="node-detail-id">#{node.id}</span>
        <span className="node-detail-source">{node.source}</span>
        {node.page > 0 && <span className="node-detail-page">p.{node.page}</span>}
      </div>
      <p className="node-detail-label">{node.label}</p>
      <div className="node-detail-meta">
        <span>PR: {node.pagerank?.toFixed(5)}</span>
        <span>deg: {node.degree}</span>
        <span>cluster: {node.cluster}</span>
      </div>
      <p className="node-detail-text">{node.text?.slice(0, 200)}…</p>
    </div>
  );
}

// ── Root layout ───────────────────────────────────────────────────────────────

function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="app-root">
      <TopBar />

      <div className="app-body">
        {/* Left sidebar */}
        <div className={`sidebar ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? "◂" : "▸"}
          </button>

          {sidebarOpen && (
            <>
              <section className="sidebar-section">
                <h3 className="sidebar-section-title">Ingest</h3>
                <IngestPanel />
              </section>

              <section className="sidebar-section">
                <h3 className="sidebar-section-title">Search</h3>
                <SearchPanel />
              </section>
            </>
          )}
        </div>

        {/* Main graph area */}
        <main className="main-area">
          <KnowledgeGraph />
          <NodeDetail />
        </main>

        {/* Assessment panel */}
        <AssessmentPanel />
      </div>
    </div>
  );
}

// ── App entry ─────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <KnowledgeProvider>
      <AssessmentProvider>
        <AppLayout />
      </AssessmentProvider>
    </KnowledgeProvider>
  );
}
