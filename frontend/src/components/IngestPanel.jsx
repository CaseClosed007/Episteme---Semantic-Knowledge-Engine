// frontend/src/components/IngestPanel.jsx
import { useCallback, useRef, useState } from "react";
import { useKnowledge } from "../contexts/KnowledgeContext";

export default function IngestPanel() {
  const { ingestText, ingestPDF, ingestLoading, ingestError, ingestResult } =
    useKnowledge();

  const [mode, setMode] = useState("text"); // "text" | "pdf"
  const [textInput, setTextInput] = useState("");
  const [sourceLabel, setSourceLabel] = useState("");
  const fileRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  // ── Text submit ────────────────────────────────────────────────────────────
  const handleTextSubmit = useCallback(() => {
    if (!textInput.trim()) return;
    ingestText(textInput.trim(), sourceLabel || "manual");
    setTextInput("");
  }, [textInput, sourceLabel, ingestText]);

  // ── File processing ────────────────────────────────────────────────────────
  const processFile = useCallback(
    (file) => {
      if (!file?.name.toLowerCase().endsWith(".pdf")) {
        alert("Please upload a PDF file.");
        return;
      }
      ingestPDF(file);
    },
    [ingestPDF]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleFileChange = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="ingest-panel">
      {/* Mode tabs */}
      <div className="ingest-mode-tabs">
        <button
          className={`tab ${mode === "text" ? "tab-active" : ""}`}
          onClick={() => setMode("text")}
        >
          Text
        </button>
        <button
          className={`tab ${mode === "pdf" ? "tab-active" : ""}`}
          onClick={() => setMode("pdf")}
        >
          PDF
        </button>
      </div>

      {/* Text mode */}
      {mode === "text" && (
        <div className="ingest-text-form">
          <input
            className="source-input"
            placeholder="Source label (optional)"
            value={sourceLabel}
            onChange={(e) => setSourceLabel(e.target.value)}
          />
          <textarea
            className="ingest-textarea"
            placeholder="Paste text to ingest… (min 50 characters)"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            rows={5}
          />
          <button
            className="btn-primary btn-full"
            disabled={ingestLoading || textInput.trim().length < 50}
            onClick={handleTextSubmit}
          >
            {ingestLoading ? "Ingesting…" : "Ingest Text"}
          </button>
        </div>
      )}

      {/* PDF mode */}
      {mode === "pdf" && (
        <div
          className={`drop-zone ${dragOver ? "drop-zone--over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          {ingestLoading ? (
            <p>Extracting &amp; indexing PDF…</p>
          ) : (
            <>
              <div className="drop-icon">⊕</div>
              <p>Drop a PDF here or click to browse</p>
            </>
          )}
        </div>
      )}

      {/* Feedback messages */}
      {ingestError && (
        <div className="ingest-feedback ingest-error">
          ⚠ {ingestError}
        </div>
      )}
      {ingestResult && !ingestLoading && (
        <div className="ingest-feedback ingest-success">
          ✓ {ingestResult.chunksAdded} chunks added from "{ingestResult.source}"
          — {ingestResult.totalChunks} total in index
        </div>
      )}
    </div>
  );
}