// frontend/src/contexts/AssessmentContext.jsx
// =============================================================================
// Assessment State Context
// ─────────────────────────
// Manages the complete generative assessment lifecycle:
//
//   IDLE → GENERATING_QUESTION → AWAITING_ANSWER → EVALUATING → COMPLETE
//                                       ↑_____________________________|
//
// Key engineering decisions:
//
//   1. SSE streaming is managed via a ref (sseRef) to hold the EventSource
//      object.  This avoids placing a non-serialisable object in state and
//      ensures we can close the connection from any callback without stale
//      closure issues.
//
//   2. useCallback wraps every action so AssessmentPanel only re-renders
//      when the slice of state it consumes changes — not on every parent
//      render.
//
//   3. The streaming token buffer is accumulated in a ref (tokenBufferRef)
//      and flushed to state in a requestAnimationFrame callback so the
//      DOM is updated at 60fps regardless of how fast the LLM produces
//      tokens.  This prevents the browser from lagging during streaming.
//
//   4. useEffect cleans up the EventSource on unmount/node-change so we
//      never leak connections.
// =============================================================================

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:5001";

// ── Assessment phase enum ─────────────────────────────────────────────────────

export const Phase = Object.freeze({
  IDLE:                "IDLE",
  GENERATING_QUESTION: "GENERATING_QUESTION",
  AWAITING_ANSWER:     "AWAITING_ANSWER",
  EVALUATING:          "EVALUATING",
  COMPLETE:            "COMPLETE",
  ERROR:               "ERROR",
});

// ── State ─────────────────────────────────────────────────────────────────────

const initialState = {
  phase: Phase.IDLE,
  nodeId: null,
  nodePreview: "",

  // Question generation
  streamingQuestion: "",      // tokens arriving via SSE
  finalQuestion: "",          // assembled question after streaming

  // User input
  userAnswer: "",

  // Evaluation result
  evaluationResult: null,     // { finalScore, grade, summary, strengths, gaps, ... }
  streamingEval: "",          // streamed reasoning text

  // Session history (array of completed rounds)
  history: [],

  error: null,
};

// ── Reducer ────────────────────────────────────────────────────────────────────

const A = {
  START_QUESTION:      "START_QUESTION",
  QUESTION_TOKEN:      "QUESTION_TOKEN",
  QUESTION_DONE:       "QUESTION_DONE",
  SET_ANSWER:          "SET_ANSWER",
  START_EVAL:          "START_EVAL",
  EVAL_TOKEN:          "EVAL_TOKEN",
  EVAL_DONE:           "EVAL_DONE",
  RESET:               "RESET",
  ERROR:               "ERROR",
};

function reducer(state, action) {
  switch (action.type) {
    case A.START_QUESTION:
      return {
        ...state,
        phase: Phase.GENERATING_QUESTION,
        nodeId: action.payload.nodeId,
        nodePreview: action.payload.nodePreview ?? "",
        streamingQuestion: "",
        finalQuestion: "",
        userAnswer: "",
        evaluationResult: null,
        streamingEval: "",
        error: null,
      };

    case A.QUESTION_TOKEN:
      // Accumulate tokens — the rAF flush in the hook handles batching
      return {
        ...state,
        streamingQuestion: state.streamingQuestion + action.payload,
      };

    case A.QUESTION_DONE:
      return {
        ...state,
        phase: Phase.AWAITING_ANSWER,
        finalQuestion: state.streamingQuestion.trim(),
      };

    case A.SET_ANSWER:
      return { ...state, userAnswer: action.payload };

    case A.START_EVAL:
      return {
        ...state,
        phase: Phase.EVALUATING,
        streamingEval: "",
        evaluationResult: null,
      };

    case A.EVAL_TOKEN:
      return { ...state, streamingEval: state.streamingEval + action.payload };

    case A.EVAL_DONE:
      return {
        ...state,
        phase: Phase.COMPLETE,
        evaluationResult: action.payload,
        // Append to history
        history: [
          ...state.history,
          {
            nodeId: state.nodeId,
            question: state.finalQuestion,
            answer: state.userAnswer,
            result: action.payload,
            timestamp: Date.now(),
          },
        ],
      };

    case A.RESET:
      return { ...initialState, history: state.history };

    case A.ERROR:
      return { ...state, phase: Phase.ERROR, error: action.payload };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────

const AssessmentContext = createContext(null);

export function AssessmentProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Non-reactive refs — mutated without triggering re-renders
  const sseRef = useRef(null);             // active EventSource
  const tokenBufferRef = useRef("");       // accumulated tokens not yet flushed
  const rafRef = useRef(null);             // requestAnimationFrame handle
  const evalBufferRef = useRef("");        // eval streaming buffer

  // ── SSE helpers ───────────────────────────────────────────────────────────

  const closeSse = useCallback(() => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    tokenBufferRef.current = "";
    evalBufferRef.current = "";
  }, []);

  // rAF flush: dispatches accumulated tokens at 60fps
  const scheduleFlush = useCallback((actionType) => {
    if (rafRef.current) return; // already scheduled
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const buf = tokenBufferRef.current;
      if (buf) {
        tokenBufferRef.current = "";
        dispatch({ type: actionType, payload: buf });
      }
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => () => closeSse(), [closeSse]);

  // ── Generate question (streaming SSE) ────────────────────────────────────

  const generateQuestion = useCallback(
    (nodeId, nodePreview = "") => {
      closeSse();
      dispatch({ type: A.START_QUESTION, payload: { nodeId, nodePreview } });

      const url = `${API}/api/assess/question/stream?nodeId=${nodeId}`;
      const es = new EventSource(url);
      sseRef.current = es;

      es.addEventListener("token", (e) => {
        const data = JSON.parse(e.data);
        // Buffer tokens and flush via rAF to avoid per-token DOM updates
        tokenBufferRef.current += data.content ?? "";
        scheduleFlush(A.QUESTION_TOKEN);
      });

      es.addEventListener("done", () => {
        // Flush any remaining buffer immediately
        if (tokenBufferRef.current) {
          dispatch({ type: A.QUESTION_TOKEN, payload: tokenBufferRef.current });
          tokenBufferRef.current = "";
        }
        dispatch({ type: A.QUESTION_DONE });
        closeSse();
      });

      es.addEventListener("error", (e) => {
        const data = e.data ? JSON.parse(e.data) : {};
        dispatch({ type: A.ERROR, payload: data.message ?? "SSE error" });
        closeSse();
      });

      es.onerror = () => {
        dispatch({ type: A.ERROR, payload: "Lost connection to API" });
        closeSse();
      };
    },
    [closeSse, scheduleFlush]
  );

  // ── Blocking question (fallback when SSE not desired) ────────────────────

  const generateQuestionBlocking = useCallback(async (nodeId, nodePreview = "") => {
    dispatch({ type: A.START_QUESTION, payload: { nodeId, nodePreview } });
    try {
      const res = await fetch(`${API}/api/assess/question`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId }),
      });
      const data = await res.json();
      // Simulate streaming for UI consistency
      dispatch({ type: A.QUESTION_TOKEN, payload: data.question });
      dispatch({ type: A.QUESTION_DONE });
    } catch (err) {
      dispatch({ type: A.ERROR, payload: err.message });
    }
  }, []);

  // ── Update user answer ────────────────────────────────────────────────────

  const setUserAnswer = useCallback((text) => {
    dispatch({ type: A.SET_ANSWER, payload: text });
  }, []);

  // ── Submit for evaluation (streaming SSE) ────────────────────────────────

  const submitAnswer = useCallback(() => {
    const { finalQuestion, nodeId, userAnswer } = state;
    if (!finalQuestion || !userAnswer?.trim()) return;

    closeSse();
    dispatch({ type: A.START_EVAL });

    const params = new URLSearchParams({
      question: finalQuestion,
      nodeId: String(nodeId),
      userAnswer,
    });
    const url = `${API}/api/assess/evaluate/stream?${params}`;
    const es = new EventSource(url);
    sseRef.current = es;

    es.addEventListener("token", (e) => {
      const data = JSON.parse(e.data);
      tokenBufferRef.current += data.content ?? "";
      scheduleFlush(A.EVAL_TOKEN);
    });

    es.addEventListener("result", (e) => {
      // Merge the cosine similarity from the sentinel into the full result
      const partial = JSON.parse(e.data).data ?? {};
      evalBufferRef.current = partial;
    });

    es.addEventListener("done", async () => {
      // Flush remaining eval tokens
      if (tokenBufferRef.current) {
        dispatch({ type: A.EVAL_TOKEN, payload: tokenBufferRef.current });
        tokenBufferRef.current = "";
      }

      // Now do a blocking evaluate call to get the full structured result
      // (we get the rich JSON from the blocking endpoint; the streaming was
      //  for the "thinking aloud" UX only)
      try {
        const res = await fetch(`${API}/api/assess/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: finalQuestion,
            nodeId,
            userAnswer,
          }),
        });
        const fullResult = await res.json();
        dispatch({ type: A.EVAL_DONE, payload: fullResult });
      } catch {
        dispatch({
          type: A.EVAL_DONE,
          payload: {
            finalScore: null,
            summary: "Evaluation result unavailable",
            ...evalBufferRef.current,
          },
        });
      }
      closeSse();
    });

    es.onerror = () => {
      dispatch({ type: A.ERROR, payload: "Lost connection during evaluation" });
      closeSse();
    };
  }, [state, closeSse, scheduleFlush]);

  // ── Reset ─────────────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    closeSse();
    dispatch({ type: A.RESET });
  }, [closeSse]);

  // ── Derived values ────────────────────────────────────────────────────────

  const isStreaming = useMemo(
    () =>
      state.phase === Phase.GENERATING_QUESTION ||
      state.phase === Phase.EVALUATING,
    [state.phase]
  );

  const canSubmit = useMemo(
    () =>
      state.phase === Phase.AWAITING_ANSWER &&
      (state.userAnswer?.trim().length ?? 0) >= 10,
    [state.phase, state.userAnswer]
  );

  const sessionScore = useMemo(() => {
    if (state.history.length === 0) return null;
    const scores = state.history
      .map((h) => h.result?.finalScore)
      .filter((s) => s !== null && s !== undefined);
    return scores.length > 0
      ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2)
      : null;
  }, [state.history]);

  const value = useMemo(
    () => ({
      ...state,
      isStreaming,
      canSubmit,
      sessionScore,
      // Actions
      generateQuestion,
      generateQuestionBlocking,
      setUserAnswer,
      submitAnswer,
      reset,
    }),
    [
      state,
      isStreaming,
      canSubmit,
      sessionScore,
      generateQuestion,
      generateQuestionBlocking,
      setUserAnswer,
      submitAnswer,
      reset,
    ]
  );

  return (
    <AssessmentContext.Provider value={value}>
      {children}
    </AssessmentContext.Provider>
  );
}

export function useAssessment() {
  const ctx = useContext(AssessmentContext);
  if (!ctx) throw new Error("useAssessment must be used inside <AssessmentProvider>");
  return ctx;
}
