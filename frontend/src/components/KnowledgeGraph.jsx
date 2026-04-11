// frontend/src/components/KnowledgeGraph.jsx
// =============================================================================
// Knowledge Graph Visualisation
// ────────────────────────────────
// Renders the semantic dependency graph using react-force-graph-2d.
//
// Performance optimisations:
//   • Graph data passed as a stable useMemo reference so react-force-graph-2d
//     doesn't reinitialise the simulation on every parent render.
//   • Node colours are resolved via the clusterColours map from context
//     (also memoised) — no per-render palette computation.
//   • onNodeClick is wrapped in useCallback so the canvas doesn't re-subscribe
//     to the click handler on every render.
//   • The canvas is rendered at devicePixelRatio for crisp text on Retina/XDR.
//   • Node paint uses CanvasRenderingContext2D directly (no DOM nodes per node)
//     so 1000-node graphs stay smooth at 60fps.
// =============================================================================

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useAssessment } from "../contexts/AssessmentContext";
import { useKnowledge } from "../contexts/KnowledgeContext";

// ── Colour helpers ─────────────────────────────────────────────────────────────

const SELECTED_COLOUR   = "#f0abfc";   // fuchsia-300
const PREREQUISITE_CLR  = "#fbbf24";   // amber-400
const DEFAULT_LINK_CLR  = "rgba(148,163,184,0.35)";
const SELECTED_LINK_CLR = "rgba(240,171,252,0.7)";

// ── Custom node painter ────────────────────────────────────────────────────────

function paintNode(node, colour, ctx, globalScale, isSelected) {
  const r = Math.max(4, Math.sqrt(node.weight ?? 0.3) * 12 + (isSelected ? 3 : 0));

  // Glow for selected node
  if (isSelected) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 5, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(240,171,252,0.25)";
    ctx.fill();
  }

  // Node circle
  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = colour;
  ctx.fill();

  // Node border
  ctx.strokeStyle = isSelected ? "#f0abfc" : "rgba(255,255,255,0.3)";
  ctx.lineWidth = isSelected ? 2 / globalScale : 0.8 / globalScale;
  ctx.stroke();

  // Label — only draw when zoomed in enough
  if (globalScale > 1.5 || isSelected) {
    const fontSize = Math.max(6, 10 / globalScale);
    ctx.font = `${fontSize}px 'IBM Plex Mono', monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = isSelected ? "#fff" : "rgba(255,255,255,0.85)";

    const label = node.label?.slice(0, 24) ?? `node ${node.id}`;
    ctx.fillText(label, node.x, node.y + r + fontSize * 0.7);
  }
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function KnowledgeGraph() {
  const {
    graph,
    graphLoading,
    graphError,
    clusterColours,
    selectedNodeId,
    selectNode,
    selectedNodeData,
  } = useKnowledge();
  const { generateQuestion } = useAssessment();
  const fgRef = useRef(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // ── Responsive sizing ───────────────────────────────────────────────────────
  useEffect(() => {
    const obs = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setDimensions({ width, height });
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // ── Stable graph data ───────────────────────────────────────────────────────
  // react-force-graph-2d mutates node objects with x/y/vx/vy so we cannot
  // pass the raw context graph directly — we deep-clone the nodes/links once
  // per graph change and let the simulation own those objects.
  const graphData = useMemo(() => {
    if (!graph) return { nodes: [], links: [] };
    return {
      nodes: graph.nodes.map((n) => ({ ...n })),
      links: graph.links.map((l) => ({ ...l })),
    };
  }, [graph]);

  // Set of prerequisite node IDs for the selected node
  const prerequisiteSet = useMemo(() => {
    if (!selectedNodeData) return new Set();
    return new Set(selectedNodeData.prerequisites ?? []);
  }, [selectedNodeData]);

  // ── Callbacks ───────────────────────────────────────────────────────────────

  const handleNodeClick = useCallback(
    (node) => {
      selectNode(node.id);
      // Smooth zoom to the clicked node
      fgRef.current?.centerAt(node.x, node.y, 600);
      fgRef.current?.zoom(3, 600);
    },
    [selectNode]
  );

  const handleNodeHover = useCallback((node) => {
    // Change cursor on hover — react-force-graph-2d exposes the canvas element
    const canvas = fgRef.current?.renderer()?.domElement;
    if (canvas) canvas.style.cursor = node ? "pointer" : "default";
  }, []);

  const handleNodeRightClick = useCallback(
    (node) => {
      // Right-click directly starts an assessment for that node
      selectNode(node.id);
      generateQuestion(node.id, node.label ?? "");
    },
    [selectNode, generateQuestion]
  );

  const nodeCanvasObject = useCallback(
    (node, ctx, globalScale) => {
      const isSelected = node.id === selectedNodeId;
      const isPrereq   = prerequisiteSet.has(node.id);
      const base       = clusterColours[node.cluster] ?? "#60a5fa";
      const colour     = isPrereq ? PREREQUISITE_CLR : base;
      paintNode(node, colour, ctx, globalScale, isSelected);
    },
    [selectedNodeId, prerequisiteSet, clusterColours]
  );

  const linkColour = useCallback(
    (link) => {
      const srcId = link.source?.id ?? link.source;
      const tgtId = link.target?.id ?? link.target;
      if (srcId === selectedNodeId || tgtId === selectedNodeId) {
        return SELECTED_LINK_CLR;
      }
      return DEFAULT_LINK_CLR;
    },
    [selectedNodeId]
  );

  const linkWidth = useCallback(
    (link) => {
      const srcId = link.source?.id ?? link.source;
      const tgtId = link.target?.id ?? link.target;
      const w = link.weight ?? 0.5;
      if (srcId === selectedNodeId || tgtId === selectedNodeId) return 2;
      return Math.max(0.5, w * 2);
    },
    [selectedNodeId]
  );

  // ── Render ───────────────────────────────────────────────────────────────────

  if (graphLoading && !graph) {
    return (
      <div className="graph-placeholder">
        <div className="loading-pulse" />
        <p>Building knowledge graph…</p>
      </div>
    );
  }

  if (graphError) {
    return (
      <div className="graph-placeholder graph-error">
        <p>⚠ Graph error: {graphError}</p>
      </div>
    );
  }

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="graph-placeholder graph-empty">
        <div className="empty-icon">⬡</div>
        <p>No knowledge graph yet.</p>
        <p className="hint">Ingest a document or dataset to begin.</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="graph-container">
      {/* Stats overlay */}
      <div className="graph-stats-overlay">
        <span>{graph.stats.nodes} nodes</span>
        <span>{graph.stats.edges} edges</span>
        <span>{graph.stats.clusters} clusters</span>
        <span className="stat-hint">Right-click node → assess</span>
      </div>

      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeCanvasObject={nodeCanvasObject}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={(node, colour, ctx) => {
          // Larger hit area than the visible circle for easier clicking
          const r = Math.max(6, Math.sqrt(node.weight ?? 0.3) * 14);
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
          ctx.fillStyle = colour;
          ctx.fill();
        }}
        linkColor={linkColour}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={0.85}
        linkCurvature={0.15}
        onNodeClick={handleNodeClick}
        onNodeRightClick={handleNodeRightClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={120}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        backgroundColor="transparent"
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
      />
    </div>
  );
}
