// frontend/src/components/KnowledgeGraph.jsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useAssessment } from "../contexts/AssessmentContext";
import { useKnowledge } from "../contexts/KnowledgeContext";

const PREREQ_COLOUR    = "#fbbf24";
const DEFAULT_LINK_CLR = "rgba(148,163,184,0.35)";
const ACTIVE_LINK_CLR  = "rgba(240,171,252,0.75)";

function paintNode(node, colour, ctx, globalScale, isSelected) {
  const r = Math.max(5, Math.sqrt(node.weight ?? 0.5) * 12 + (isSelected ? 4 : 0));

  // Glow ring for selected node
  if (isSelected) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(240,171,252,0.20)";
    ctx.fill();
  }

  // Main circle
  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = colour;
  ctx.fill();

  // Border
  ctx.strokeStyle = isSelected ? "#f0abfc" : "rgba(255,255,255,0.25)";
  ctx.lineWidth = isSelected ? 2.5 / globalScale : 1 / globalScale;
  ctx.stroke();

  // Label — show when zoomed in or selected
  if (globalScale > 1.4 || isSelected) {
    const fontSize = Math.max(5, 10 / globalScale);
    ctx.font = `${fontSize}px monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = isSelected ? "#ffffff" : "rgba(255,255,255,0.80)";
    const label = (node.label ?? `#${node.id}`).slice(0, 28);
    ctx.fillText(label, node.x, node.y + r + 2);
  }
}

export default function KnowledgeGraph() {
  const {
    graph,
    graphLoading,
    graphError,
    clusterColours,
    selectedNodeId,
    selectNode,
    selectedNodeData,
    graphName,
    derivedStats,
  } = useKnowledge();

  const { generateQuestion } = useAssessment();
  const fgRef        = useRef(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // null = show all sources; string = show only that source
  const [activeSource, setActiveSource] = useState(null);

  // If the stored source is no longer present in the graph (e.g. after a
  // full rebuild), treat the selection as "all" without triggering setState.
  const sources = useMemo(() => derivedStats?.sources ?? [], [derivedStats]);
  const effectiveSource = sources.includes(activeSource) ? activeSource : null;

  // ── Responsive sizing ──────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        setDimensions({ width, height });
      }
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // ── Graph data — filtered by effectiveSource when set ────────────────────
  const graphData = useMemo(() => {
    if (!graph || !graph.nodes?.length) return { nodes: [], links: [] };

    const nodes = effectiveSource
      ? graph.nodes.filter((n) => n.source === effectiveSource).map((n) => ({ ...n }))
      : graph.nodes.map((n) => ({ ...n }));

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links = graph.links
      .filter((l) => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        return nodeIds.has(s) && nodeIds.has(t);
      })
      .map((l) => ({ ...l }));

    return { nodes, links };
  }, [graph, effectiveSource]);

  // Zoom-to-fit after switching source so the user always sees the subgraph
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    const t = setTimeout(() => fg.zoomToFit(400, 40), 300);
    return () => clearTimeout(t);
  }, [effectiveSource]);

  // Set of prerequisite node IDs for the selected node
  const prerequisiteSet = useMemo(() => {
    return new Set(selectedNodeData?.prerequisites ?? []);
  }, [selectedNodeData]);

  // ── Node painter ───────────────────────────────────────────────────────
  const nodeCanvasObject = useCallback(
    (node, ctx, globalScale) => {
      const isSelected = node.id === selectedNodeId;
      const isPrereq   = prerequisiteSet.has(node.id);
      const colour     = isPrereq
        ? PREREQ_COLOUR
        : (clusterColours[node.cluster] ?? "#60a5fa");
      paintNode(node, colour, ctx, globalScale, isSelected);
    },
    [selectedNodeId, prerequisiteSet, clusterColours]
  );

  // ── Hit area (FIXED: uses a literal colour, not an undefined variable) ─
  const nodePointerAreaPaint = useCallback((node, _colour, ctx) => {
    // Use a fixed hit-area radius regardless of visual size
    const r = Math.max(8, Math.sqrt(node.weight ?? 0.5) * 14);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = _colour;   // _colour is provided by react-force-graph-2d
    ctx.fill();
  }, []);

  // ── Click handler ──────────────────────────────────────────────────────
  const handleNodeClick = useCallback(
    (node) => {
      selectNode(node.id);
      // Smooth pan + zoom to the clicked node
      if (fgRef.current) {
        fgRef.current.centerAt(node.x, node.y, 500);
        fgRef.current.zoom(3.5, 500);
      }
    },
    [selectNode]
  );

  // Right-click = instant assessment
  const handleNodeRightClick = useCallback(
    (node, event) => {
      event.preventDefault();
      selectNode(node.id);
      generateQuestion(node.id, node.label ?? "");
    },
    [selectNode, generateQuestion]
  );

  // Cursor change on hover
  const handleNodeHover = useCallback((node) => {
    const canvas = fgRef.current?.renderer?.()?.domElement
      ?? containerRef.current?.querySelector("canvas");
    if (canvas) canvas.style.cursor = node ? "pointer" : "grab";
  }, []);

  // ── Link styles ────────────────────────────────────────────────────────
  const linkColour = useCallback(
    (link) => {
      const s = typeof link.source === "object" ? link.source.id : link.source;
      const t = typeof link.target === "object" ? link.target.id : link.target;
      return s === selectedNodeId || t === selectedNodeId
        ? ACTIVE_LINK_CLR
        : DEFAULT_LINK_CLR;
    },
    [selectedNodeId]
  );

  const linkWidth = useCallback(
    (link) => {
      const s = typeof link.source === "object" ? link.source.id : link.source;
      const t = typeof link.target === "object" ? link.target.id : link.target;
      if (s === selectedNodeId || t === selectedNodeId) return 2;
      return Math.max(0.5, (link.weight ?? 0.5) * 2);
    },
    [selectedNodeId]
  );

  // ── Render guards ──────────────────────────────────────────────────────
  if (graphLoading && !graph) {
    return (
      <div style={styles.placeholder}>
        <div style={styles.spinner} />
        <p style={{ color: "#94a3b8", marginTop: 12 }}>Building knowledge graph…</p>
      </div>
    );
  }

  if (graphError) {
    return (
      <div style={styles.placeholder}>
        <p style={{ color: "#f87171" }}>⚠ {graphError}</p>
      </div>
    );
  }

  if (!graph || !graph.nodes?.length) {
    return (
      <div style={styles.placeholder}>
        <div style={{ fontSize: 48, opacity: 0.2 }}>⬡</div>
        <p style={{ color: "#94a3b8", marginTop: 8 }}>No graph yet — ingest a document to begin.</p>
        <p style={{ color: "#475569", fontSize: 12, marginTop: 4 }}>
          Use the Ingest panel on the left.
        </p>
      </div>
    );
  }

  return (
    // IMPORTANT: position:relative + overflow:visible so the canvas
    // receives all pointer events without being clipped
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "visible",   // ← was "hidden" in styles.css — this was blocking drag
        cursor: "grab",
      }}
    >
      {/* Graph navigator — name + clickable source tabs */}
      <div style={styles.navigator}>
        {/* Current graph title */}
        <div style={styles.nameLabel}>
          {effectiveSource ?? graphName ?? "All Graphs"}
        </div>

        {/* Source tabs (only shown when there are sources) */}
        {sources.length > 0 && (
          <div style={styles.sourceTabs}>
            {/* "All" tab */}
            <button
              style={{
                ...styles.sourceTab,
                ...(effectiveSource === null ? styles.sourceTabActive : {}),
              }}
              onClick={() => setActiveSource(null)}
            >
              All
            </button>

            {sources.map((s) => (
              <button
                key={s}
                title={s}
                style={{
                  ...styles.sourceTab,
                  ...(effectiveSource === s ? styles.sourceTabActive : {}),
                }}
                onClick={() => setActiveSource((prev) => (prev === s ? null : s))}
              >
                {s.length > 22 ? s.slice(0, 20) + "…" : s}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Stats overlay — pointer-events:none so it never blocks the canvas */}
      <div style={styles.statsOverlay}>
        <span>{graphData.nodes.length} nodes</span>
        <span>{graphData.links.length} edges</span>
        {effectiveSource
          ? <span style={{ color: "#f0abfc" }}>filtered</span>
          : <span>{graph.stats?.clusters ?? 0} clusters</span>
        }
        <span style={{ color: "#475569" }}>Right-click node → assess</span>
      </div>

      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        // ── Node rendering ─────────────────────────────────────────
        nodeCanvasObject={nodeCanvasObject}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={nodePointerAreaPaint}
        // ── Link rendering ─────────────────────────────────────────
        linkColor={linkColour}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={5}
        linkDirectionalArrowRelPos={0.88}
        linkCurvature={0.12}
        // ── Interaction ────────────────────────────────────────────
        onNodeClick={handleNodeClick}
        onNodeRightClick={handleNodeRightClick}
        onNodeHover={handleNodeHover}
        enableNodeDrag={true}          // ← drag individual nodes
        enableZoomInteraction={true}   // ← scroll to zoom
        enablePanInteraction={true}    // ← drag background to pan
        // ── Simulation ─────────────────────────────────────────────
        cooldownTicks={150}
        d3AlphaDecay={0.015}
        d3VelocityDecay={0.25}
        backgroundColor="transparent"
      />
    </div>
  );
}

// Inline styles used here instead of class names to avoid CSS conflicts
const styles = {
  placeholder: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    gap: 8,
  },
  spinner: {
    width: 40,
    height: 40,
    borderRadius: "50%",
    border: "2px solid rgba(99,110,131,0.3)",
    borderTopColor: "#60a5fa",
    animation: "spin 1s linear infinite",
  },
  statsOverlay: {
    position: "absolute",
    bottom: 16,
    left: 16,
    display: "flex",
    gap: 12,
    background: "rgba(13,17,23,0.85)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(99,110,131,0.25)",
    borderRadius: 6,
    padding: "4px 12px",
    fontSize: 11,
    color: "#94a3b8",
    pointerEvents: "none",
    zIndex: 10,
  },
  navigator: {
    position: "absolute",
    top: 16,
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
    zIndex: 10,
  },
  nameLabel: {
    background: "rgba(13,17,23,0.88)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(240,171,252,0.35)",
    borderRadius: 20,
    padding: "4px 16px",
    fontSize: 13,
    fontWeight: 700,
    color: "#f0abfc",
    letterSpacing: "0.06em",
    whiteSpace: "nowrap",
    maxWidth: 440,
    overflow: "hidden",
    textOverflow: "ellipsis",
    pointerEvents: "none",
  },
  sourceTabs: {
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 4,
    maxWidth: 560,
  },
  sourceTab: {
    background: "rgba(13,17,23,0.80)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(99,110,131,0.3)",
    borderRadius: 4,
    padding: "3px 10px",
    fontSize: 10,
    color: "#94a3b8",
    cursor: "pointer",
    whiteSpace: "nowrap",
    maxWidth: 160,
    overflow: "hidden",
    textOverflow: "ellipsis",
    transition: "border-color 0.15s, color 0.15s, background 0.15s",
  },
  sourceTabActive: {
    background: "rgba(240,171,252,0.12)",
    border: "1px solid rgba(240,171,252,0.55)",
    color: "#f0abfc",
    fontWeight: 700,
  },
};