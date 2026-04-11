// frontend/src/components/AssessmentPanel.jsx
// =============================================================================
// Assessment Panel
// ─────────────────
// Renders the generative assessment interface.  Driven entirely by
// AssessmentContext state — this component contains zero local state
// except for the textarea focus ref.
//
// Phases rendered:
//   IDLE               → "Select a node to start"
//   GENERATING_QUESTION → Streaming question with cursor blink
//   AWAITING_ANSWER    → Textarea + submit button
//   EVALUATING         → Streaming eval reasoning + spinner
//   COMPLETE           → Score card with strengths/gaps
//   ERROR              → Error message + retry
// =============================================================================

import { useCallback, useEffect, useRef } from "react";
import { Phase, useAssessment } from "../contexts/AssessmentContext";
import { useKnowledge } from "../contexts/KnowledgeContext";

// ── Score ring (SVG) ──────────────────────────────────────────────────────────

function ScoreRing({ score, max = 5 }) {
  const pct = Math.max(0, Math.min(score / max, 1));
  const r = 36;
  const circ = 2 * Math.PI * r;
  const dash = pct * circ;
  const grade = score >= 4.5 ? "A" : score >= 3.5 ? "B" : score >= 2.5 ? "C" : score >= 1.5 ? "D" : "F";
  const colour = score >= 4 ? "#34d399" : score >= 3 ? "#60a5fa" : score >= 2 ? "#f59e0b" : "#f87171";

  return (
    <svg className="score-ring" viewBox="0 0 90 90" width={90} height={90}>
      <circle cx={45} cy={45} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={8} />
      <circle
        cx={45} cy={45} r={r}
        fill="none"
        stroke={colour}
        strokeWidth={8}
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeDashoffset={circ / 4}  /* start at 12 o'clock */
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.34,1.56,0.64,1)" }}
      />
      <text x={45} y={40} textAnchor="middle" dominantBaseline="middle"
        fill={colour} fontSize={20} fontWeight={700} fontFamily="'IBM Plex Mono', monospace">
        {grade}
      </text>
      <text x={45} y={58} textAnchor="middle" dominantBaseline="middle"
        fill="rgba(255,255,255,0.5)" fontSize={10} fontFamily="'IBM Plex Mono', monospace">
        {score?.toFixed(1)}/{max}
      </text>
    </svg>
  );
}

// ── Streaming cursor ──────────────────────────────────────────────────────────

function StreamCursor() {
  return <span className="stream-cursor" aria-hidden="true">▋</span>;
}

// ── Phase label pill ──────────────────────────────────────────────────────────

const PHASE_LABELS = {
  [Phase.IDLE]:                 { text: "Idle",            cls: "phase-idle"     },
  [Phase.GENERATING_QUESTION]:  { text: "Generating…",     cls: "phase-active"   },
  [Phase.AWAITING_ANSWER]:      { text: "Your Turn",        cls: "phase-answer"   },
  [Phase.EVALUATING]:           { text: "Evaluating…",     cls: "phase-active"   },
  [Phase.COMPLETE]:             { text: "Complete",         cls: "phase-complete" },
  [Phase.ERROR]:                { text: "Error",            cls: "phase-error"    },
};

// ── Main component ─────────────────────────────────────────────────────────────

export default function AssessmentPanel() {
  const {
    phase,
    nodeId,
    nodePreview,
    streamingQuestion,
    finalQuestion,
    userAnswer,
    streamingEval,
    evaluationResult,
    history,
    isStreaming,
    canSubmit,
    sessionScore,
    error,
    generateQuestion,
    setUserAnswer,
    submitAnswer,
    reset,
  } = useAssessment();

  const { selectedNodeData, selectedNodeId } = useKnowledge();
  const textareaRef = useRef(null);

  // Auto-focus textarea when entering AWAITING_ANSWER
  useEffect(() => {
    if (phase === Phase.AWAITING_ANSWER && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [phase]);

  // Keyboard shortcut: Ctrl+Enter to submit
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canSubmit) {
        submitAnswer();
      }
    },
    [canSubmit, submitAnswer]
  );

  const handleStartAssessment = useCallback(() => {
    const id = nodeId ?? selectedNodeId;
    if (id == null) return;
    const preview = selectedNodeData?.node?.label ?? "";
    generateQuestion(id, preview);
  }, [nodeId, selectedNodeId, selectedNodeData, generateQuestion]);

  // ── Phase renders ──────────────────────────────────────────────────────────

  const { text: phaseText, cls: phaseCls } = PHASE_LABELS[phase];

  const renderBody = () => {
    switch (phase) {
      case Phase.IDLE:
        return (
          <div className="assess-idle">
            <div className="idle-icon">◈</div>
            {selectedNodeId != null ? (
              <>
                <p className="idle-node-preview">
                  {selectedNodeData?.node?.label ?? `Node #${selectedNodeId}`}
                </p>
                <button className="btn-primary" onClick={handleStartAssessment}>
                  Start Assessment
                </button>
                <p className="idle-hint">
                  Or right-click any graph node to assess it instantly.
                </p>
              </>
            ) : (
              <p className="idle-hint">
                Click a node in the graph to select it, then start an assessment.
              </p>
            )}
          </div>
        );

      case Phase.GENERATING_QUESTION:
        return (
          <div className="assess-generating">
            <p className="section-label">Generating question…</p>
            {nodePreview && (
              <p className="node-preview-text">
                Topic: <em>{nodePreview}</em>
              </p>
            )}
            <div className="streaming-text question-stream">
              {streamingQuestion}
              <StreamCursor />
            </div>
          </div>
        );

      case Phase.AWAITING_ANSWER:
        return (
          <div className="assess-answer">
            <p className="section-label">Question</p>
            <div className="question-box">{finalQuestion}</div>

            <p className="section-label" style={{ marginTop: "1.5rem" }}>
              Your Answer
              <span className="char-count">
                {userAnswer.length} chars
              </span>
            </p>
            <textarea
              ref={textareaRef}
              className="answer-textarea"
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Write your answer here… (⌘+Enter to submit)"
              rows={6}
            />
            <div className="answer-actions">
              <button
                className="btn-primary"
                disabled={!canSubmit}
                onClick={submitAnswer}
              >
                Submit Answer
              </button>
              <button className="btn-ghost" onClick={reset}>
                Skip
              </button>
            </div>
            {!canSubmit && userAnswer.length > 0 && (
              <p className="hint-text">
                Answer must be at least 10 characters.
              </p>
            )}
          </div>
        );

      case Phase.EVALUATING:
        return (
          <div className="assess-evaluating">
            <p className="section-label">Question</p>
            <div className="question-box question-box--muted">{finalQuestion}</div>

            <p className="section-label" style={{ marginTop: "1rem" }}>
              Your Answer
            </p>
            <div className="answer-readonly">{userAnswer}</div>

            <p className="section-label" style={{ marginTop: "1.5rem" }}>
              Evaluating…
            </p>
            <div className="streaming-text eval-stream">
              {streamingEval}
              <StreamCursor />
            </div>
          </div>
        );

      case Phase.COMPLETE:
        return (
          <div className="assess-complete">
            <p className="section-label">Question</p>
            <div className="question-box question-box--muted">{finalQuestion}</div>

            <p className="section-label" style={{ marginTop: "1rem" }}>
              Your Answer
            </p>
            <div className="answer-readonly">{userAnswer}</div>

            {evaluationResult && (
              <div className="eval-result">
                <div className="eval-header">
                  <ScoreRing score={evaluationResult.finalScore} />
                  <div className="eval-summary">
                    <p className="eval-summary-text">{evaluationResult.summary}</p>
                    <div className="eval-scores">
                      <span title="Combined score">
                        Overall: {evaluationResult.finalScore?.toFixed(2)}/5
                      </span>
                      <span title="LLM rubric score">
                        Rubric: {evaluationResult.llmScore}/5
                      </span>
                      <span title="Embedding cosine similarity × 5">
                        Semantic: {evaluationResult.embeddingScore?.toFixed(2)}/5
                      </span>
                    </div>
                  </div>
                </div>

                {evaluationResult.strengths?.length > 0 && (
                  <div className="eval-section">
                    <p className="eval-section-label strengths-label">✓ Strengths</p>
                    <ul>
                      {evaluationResult.strengths.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {evaluationResult.gaps?.length > 0 && (
                  <div className="eval-section">
                    <p className="eval-section-label gaps-label">✗ Gaps</p>
                    <ul>
                      {evaluationResult.gaps.map((g, i) => (
                        <li key={i}>{g}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {evaluationResult.correctiveHint && (
                  <div className="eval-hint">
                    💡 {evaluationResult.correctiveHint}
                  </div>
                )}
              </div>
            )}

            <div className="complete-actions">
              <button className="btn-primary" onClick={handleStartAssessment}>
                Try Another Question
              </button>
              <button className="btn-ghost" onClick={reset}>
                Reset
              </button>
            </div>
          </div>
        );

      case Phase.ERROR:
        return (
          <div className="assess-error">
            <p className="error-icon">⚠</p>
            <p className="error-text">{error}</p>
            <p className="error-hint">
              Make sure Ollama is running: <code>ollama serve</code>
            </p>
            <button className="btn-primary" onClick={reset}>
              Reset
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <aside className="assessment-panel">
      {/* Header */}
      <div className="assessment-header">
        <h2 className="assessment-title">Assessment</h2>
        <span className={`phase-pill ${phaseCls}`}>{phaseText}</span>
      </div>

      {/* Session stats */}
      {history.length > 0 && (
        <div className="session-stats">
          <span>{history.length} round{history.length !== 1 ? "s" : ""}</span>
          {sessionScore && <span>Avg: {sessionScore}/5</span>}
        </div>
      )}

      {/* Body */}
      <div className="assessment-body">{renderBody()}</div>

      {/* History mini-list */}
      {history.length > 0 && phase === Phase.IDLE && (
        <div className="history-list">
          <p className="section-label">Recent Rounds</p>
          {[...history].reverse().slice(0, 5).map((h, i) => (
            <div key={i} className="history-item">
              <span
                className="history-grade"
                style={{ color: h.result?.grade === "A" || h.result?.grade === "B" ? "#34d399" : "#f87171" }}
              >
                {h.result?.grade ?? "?"}
              </span>
              <span className="history-q">{h.question?.slice(0, 60)}…</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
