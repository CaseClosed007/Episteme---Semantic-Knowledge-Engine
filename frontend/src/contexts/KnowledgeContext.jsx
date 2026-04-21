// frontend/src/contexts/KnowledgeContext.jsx
// =============================================================================
// Global Knowledge State Context
// ────────────────────────────────
// Manages:
//   • The full knowledge graph (nodes + links)
//   • Selected node (drives assessment panel)
//   • Ingest state (loading / error)
//   • Search results
//
// Architecture:
//   useReducer handles state transitions deterministically.
//   useCallback memoizes all dispatch-wrapped action creators so child
//   components never get stale closures or trigger unnecessary re-renders.
//
//   The graph data itself can be large (1000s of nodes) so we gate all
//   derived computations behind useMemo.
// =============================================================================

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:5001";

// ── State shape ───────────────────────────────────────────────────────────────

const initialState = {
  // Graph
  graph: null,               // { nodes: [], links: [], stats: {} }
  graphLoading: false,
  graphError: null,
  lastBuiltAt: null,

  // Graph identity
  graphName: "",             // display name for the current graph

  // Selection
  selectedNodeId: null,      // currently focused node id
  selectedNodeData: null,    // full node detail + subgraph from API

  // Ingest
  ingestLoading: false,
  ingestError: null,
  ingestResult: null,        // { chunksAdded, totalChunks, source }

  // Search
  searchQuery: "",
  searchResults: [],
  searchLoading: false,

  // Health
  health: null,
};

// ── Action types ──────────────────────────────────────────────────────────────

const A = {
  GRAPH_LOADING:      "GRAPH_LOADING",
  GRAPH_SUCCESS:      "GRAPH_SUCCESS",
  GRAPH_ERROR:        "GRAPH_ERROR",

  NODE_LOADING:       "NODE_LOADING",
  NODE_SUCCESS:       "NODE_SUCCESS",

  INGEST_LOADING:     "INGEST_LOADING",
  INGEST_SUCCESS:     "INGEST_SUCCESS",
  INGEST_ERROR:       "INGEST_ERROR",

  SEARCH_LOADING:     "SEARCH_LOADING",
  SEARCH_SUCCESS:     "SEARCH_SUCCESS",

  HEALTH_SUCCESS:     "HEALTH_SUCCESS",
  SELECT_NODE:        "SELECT_NODE",
  SET_SEARCH_QUERY:   "SET_SEARCH_QUERY",
  SET_GRAPH_NAME:     "SET_GRAPH_NAME",
};

// ── Pure reducer ──────────────────────────────────────────────────────────────

function reducer(state, action) {
  switch (action.type) {
    case A.GRAPH_LOADING:
      return { ...state, graphLoading: true, graphError: null };

    case A.GRAPH_SUCCESS:
      return {
        ...state,
        graphLoading: false,
        graph: action.payload,
        lastBuiltAt: Date.now(),
      };

    case A.GRAPH_ERROR:
      return { ...state, graphLoading: false, graphError: action.payload };

    case A.NODE_LOADING:
      // Keep previous selectedNodeData during load to avoid UI flash
      return { ...state };

    case A.NODE_SUCCESS:
      return {
        ...state,
        selectedNodeId: action.payload.node.id,
        selectedNodeData: action.payload,
      };

    case A.SELECT_NODE:
      // Immediate selection — data loaded asynchronously
      return { ...state, selectedNodeId: action.payload };

    case A.INGEST_LOADING:
      return { ...state, ingestLoading: true, ingestError: null, ingestResult: null };

    case A.INGEST_SUCCESS:
      return {
        ...state,
        ingestLoading: false,
        ingestResult: action.payload,
        graphName: action.payload.source ?? state.graphName,
      };

    case A.SET_GRAPH_NAME:
      return { ...state, graphName: action.payload };

    case A.INGEST_ERROR:
      return { ...state, ingestLoading: false, ingestError: action.payload };

    case A.SEARCH_LOADING:
      return { ...state, searchLoading: true };

    case A.SEARCH_SUCCESS:
      return { ...state, searchLoading: false, searchResults: action.payload };

    case A.HEALTH_SUCCESS:
      return { ...state, health: action.payload };

    case A.SET_SEARCH_QUERY:
      return { ...state, searchQuery: action.payload };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────

const KnowledgeContext = createContext(null);

export function KnowledgeProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // ── Fetch graph ─────────────────────────────────────────────────────────────
  const fetchGraph = useCallback(async () => {
    dispatch({ type: A.GRAPH_LOADING });
    try {
      const res = await fetch(`${API}/api/graph`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      dispatch({ type: A.GRAPH_SUCCESS, payload: data });
    } catch (err) {
      dispatch({ type: A.GRAPH_ERROR, payload: err.message });
    }
  }, []);

  // ── Fetch node detail ───────────────────────────────────────────────────────
  const fetchNodeDetail = useCallback(async (nodeId) => {
    dispatch({ type: A.NODE_LOADING });
    try {
      const res = await fetch(`${API}/api/graph/node/${nodeId}?hops=2`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      dispatch({ type: A.NODE_SUCCESS, payload: data });
    } catch (err) {
      console.error("fetchNodeDetail failed:", err);
    }
  }, []);

  // ── Select node ─────────────────────────────────────────────────────────────
  const selectNode = useCallback(
    (nodeId) => {
      dispatch({ type: A.SELECT_NODE, payload: nodeId });
      if (nodeId !== null) {
        fetchNodeDetail(nodeId);
      }
    },
    [fetchNodeDetail]
  );

  // ── Ingest text ─────────────────────────────────────────────────────────────
  const ingestText = useCallback(
    async (text, source = "manual") => {
      dispatch({ type: A.INGEST_LOADING });
      try {
        const res = await fetch(`${API}/api/ingest/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, source }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error ?? `HTTP ${res.status}`);
        }
        const data = await res.json();
        dispatch({ type: A.INGEST_SUCCESS, payload: data });
        // Auto-refresh graph after ingest
        await fetchGraph();
      } catch (err) {
        dispatch({ type: A.INGEST_ERROR, payload: err.message });
      }
    },
    [fetchGraph]
  );

  // ── Set graph name manually ─────────────────────────────────────────────────
  const setGraphName = useCallback((name) => {
    dispatch({ type: A.SET_GRAPH_NAME, payload: name });
  }, []);

  // ── Ingest PDF ──────────────────────────────────────────────────────────────
  const ingestPDF = useCallback(
    async (file, displayName) => {
      dispatch({ type: A.INGEST_LOADING });
      const form = new FormData();
      form.append("file", file);
      // Use displayName as source if provided; falls back to raw filename
      form.append("source", displayName || file.name);
      try {
        const res = await fetch(`${API}/api/ingest/pdf`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error ?? `HTTP ${res.status}`);
        }
        const data = await res.json();
        dispatch({ type: A.INGEST_SUCCESS, payload: data });
        await fetchGraph();
      } catch (err) {
        dispatch({ type: A.INGEST_ERROR, payload: err.message });
      }
    },
    [fetchGraph]
  );

  // ── Semantic search ─────────────────────────────────────────────────────────
  const search = useCallback(async (query, k = 10) => {
    dispatch({ type: A.SEARCH_LOADING });
    dispatch({ type: A.SET_SEARCH_QUERY, payload: query });
    try {
      const res = await fetch(
        `${API}/api/search?q=${encodeURIComponent(query)}&k=${k}`
      );
      const data = await res.json();
      dispatch({ type: A.SEARCH_SUCCESS, payload: data.results ?? [] });
    } catch {
      dispatch({ type: A.SEARCH_SUCCESS, payload: [] });
    }
  }, []);

  // ── Health check ────────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/health`);
      const data = await res.json();
      dispatch({ type: A.HEALTH_SUCCESS, payload: data });
    } catch {
      dispatch({
        type: A.HEALTH_SUCCESS,
        payload: { flask: "offline", ollama: { status: "offline" } },
      });
    }
  }, []);

  // ── Initial load ─────────────────────────────────────────────────────────────
  useEffect(() => {
    checkHealth();
    fetchGraph();
  }, [checkHealth, fetchGraph]);

  // ── Derived data (memoised to avoid recalculation on every render) ───────────
  const derivedStats = useMemo(() => {
    if (!state.graph) return null;
    const { nodes, stats } = state.graph;
    const sources = [...new Set(nodes.map((n) => n.source))];
    const clusterCount = stats?.clusters ?? 0;
    const avgPagerank =
      nodes.length > 0
        ? nodes.reduce((s, n) => s + n.pagerank, 0) / nodes.length
        : 0;
    return { sources, clusterCount, avgPagerank: avgPagerank.toFixed(6) };
  }, [state.graph]);

  // Cluster colour map — stable reference, recomputed only when graph changes
  const clusterColours = useMemo(() => {
    if (!state.graph) return {};
    const palette = [
      "#60a5fa", "#34d399", "#f59e0b", "#f87171",
      "#a78bfa", "#38bdf8", "#fb923c", "#4ade80",
      "#e879f9", "#facc15", "#2dd4bf", "#f472b6",
    ];
    const map = {};
    for (const node of state.graph.nodes) {
      if (!(node.cluster in map)) {
        map[node.cluster] = palette[node.cluster % palette.length];
      }
    }
    return map;
  }, [state.graph]);

  const value = useMemo(
    () => ({
      ...state,
      derivedStats,
      clusterColours,
      // Actions
      fetchGraph,
      selectNode,
      ingestText,
      ingestPDF,
      search,
      checkHealth,
      setGraphName,
    }),
    [
      state,
      derivedStats,
      clusterColours,
      fetchGraph,
      selectNode,
      ingestText,
      ingestPDF,
      search,
      checkHealth,
      setGraphName,
    ]
  );

  return (
    <KnowledgeContext.Provider value={value}>
      {children}
    </KnowledgeContext.Provider>
  );
}

export function useKnowledge() {
  const ctx = useContext(KnowledgeContext);
  if (!ctx) throw new Error("useKnowledge must be used inside <KnowledgeProvider>");
  return ctx;
}
