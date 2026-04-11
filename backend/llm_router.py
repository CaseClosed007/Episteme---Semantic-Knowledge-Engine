# backend/llm_router.py
# =============================================================================
# Local LLM Router (Ollama backend)
# ───────────────────────────────────
# Three capabilities exposed to the Flask API:
#
#   1. generate_question(node_text)
#      → Constructs a domain-specific free-text question about the chunk.
#
#   2. evaluate_response(question, reference_text, user_answer)
#      → Semantically grades the user's free-text answer.
#      Uses TWO layers of evaluation:
#        a. Embedding cosine similarity (fast, local, no LLM call)
#        b. LLM rubric-based qualitative evaluation (richer feedback)
#
#   3. stream_generate(prompt) — generator
#      → Raw SSE-compatible streaming of LLM tokens.
#
# Design: All Ollama calls are made via the /api/chat endpoint with
# stream=True so the Flask SSE endpoint can forward tokens in real time
# without blocking the event loop.
# =============================================================================

from __future__ import annotations

import json
import logging
import time
from typing import Generator, Optional

import numpy as np
import requests
from requests.exceptions import ConnectionError, Timeout

from config import cfg
from embedder import EmbeddingEngine

log = logging.getLogger(__name__)


# ── System prompt templates ────────────────────────────────────────────────────

_QUESTION_SYSTEM = """\
You are an expert educational assessor. Your sole task is to generate ONE precise,
open-ended comprehension question that tests deep understanding of the provided
passage. The question must:
- Be answerable from the passage alone (no outside knowledge required)
- Require synthesis, not just recall
- Be one to two sentences, ending with a question mark
- NOT include the answer or any hints
Respond with ONLY the question text. No preamble, no numbering."""

_EVALUATE_SYSTEM = """\
You are a strict but fair academic grader. Evaluate the student's answer against
the reference material using this rubric:

Score 5 — Excellent: Accurate, complete, uses correct terminology, shows synthesis
Score 4 — Good: Mostly accurate, minor omissions or imprecision
Score 3 — Adequate: Core concept captured but significant gaps or imprecision
Score 2 — Partial: Some relevant content but major misunderstandings
Score 1 — Poor: Largely incorrect or off-topic

Respond ONLY with valid JSON in this exact schema:
{
  "score": <1-5>,
  "grade": "<A/B/C/D/F>",
  "summary": "<one sentence overall verdict>",
  "strengths": ["<point>", ...],
  "gaps": ["<point>", ...],
  "corrective_hint": "<one sentence pointing toward what was missed>"
}
No markdown, no explanation outside the JSON."""


# ── Ollama client ─────────────────────────────────────────────────────────────

class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class LLMRouter:
    """
    Routes LLM requests to a locally running Ollama instance.

    All public methods handle the case where Ollama is not running and return
    a graceful fallback rather than crashing the Flask server.
    """

    def __init__(self, embed_engine: EmbeddingEngine):
        self._engine = embed_engine
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return Ollama server status and available models."""
        try:
            resp = self._session.get(
                f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            target_loaded = any(cfg.LLM_MODEL in m for m in models)
            return {
                "status": "online",
                "models": models,
                "target_model": cfg.LLM_MODEL,
                "target_loaded": target_loaded,
            }
        except (ConnectionError, Timeout):
            return {
                "status": "offline",
                "error": f"Cannot reach Ollama at {cfg.OLLAMA_BASE_URL}. "
                         "Run: ollama serve",
                "models": [],
                "target_loaded": False,
            }

    # ── Core streaming call ───────────────────────────────────────────────────

    def _stream_chat(
        self,
        messages: list[dict],
        temperature: float = cfg.LLM_TEMPERATURE,
        max_tokens: int = cfg.LLM_MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """
        Low-level streaming generator.  Yields decoded token strings as they
        arrive from Ollama.  Raises OllamaError on connection failure.
        """
        payload = {
            "model": cfg.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with self._session.post(
                f"{cfg.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if obj.get("done"):
                        break

        except (ConnectionError, Timeout) as exc:
            raise OllamaError(
                f"Ollama unreachable at {cfg.OLLAMA_BASE_URL}: {exc}"
            ) from exc

    def _complete(
        self,
        messages: list[dict],
        temperature: float = cfg.LLM_TEMPERATURE,
        max_tokens: int = cfg.LLM_MAX_TOKENS,
    ) -> str:
        """Blocking version of _stream_chat — accumulates all tokens."""
        return "".join(self._stream_chat(messages, temperature, max_tokens))

    # ── Public capabilities ───────────────────────────────────────────────────

    def generate_question(
        self, node_text: str, context_texts: Optional[list[str]] = None
    ) -> dict:
        """
        Generate a comprehension question for *node_text*.

        *context_texts* — optional list of prerequisite chunk texts to provide
        richer context to the model without inflating the question difficulty.

        Returns:
          {
            "question": str,
            "node_preview": str,   # first 120 chars of the node text
            "model": str,
          }
        """
        ctx_block = ""
        if context_texts:
            ctx_block = "\n\nBackground context (do NOT ask about this):\n" + "\n---\n".join(
                t[:400] for t in context_texts[:3]
            )

        messages = [
            {"role": "system", "content": _QUESTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Generate a comprehension question about this passage:\n\n"
                    f"{node_text[:1500]}"
                    f"{ctx_block}"
                ),
            },
        ]

        t0 = time.perf_counter()
        try:
            question = self._complete(messages, temperature=0.6, max_tokens=256).strip()
        except OllamaError as exc:
            log.error("Question generation failed: %s", exc)
            # Fallback: extract a simple question from the text
            question = self._heuristic_question(node_text)

        log.info("Question generated in %.2fs", time.perf_counter() - t0)
        return {
            "question": question,
            "node_preview": node_text[:120] + ("…" if len(node_text) > 120 else ""),
            "model": cfg.LLM_MODEL,
        }

    def stream_question(
        self, node_text: str, context_texts: Optional[list[str]] = None
    ) -> Generator[str, None, None]:
        """
        Streaming version of generate_question — yields SSE-formatted strings.
        Used by the /api/assess/question/stream endpoint.
        """
        ctx_block = ""
        if context_texts:
            ctx_block = "\n\nBackground context:\n" + "\n---\n".join(
                t[:400] for t in context_texts[:3]
            )

        messages = [
            {"role": "system", "content": _QUESTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Generate a comprehension question about this passage:\n\n"
                    f"{node_text[:1500]}{ctx_block}"
                ),
            },
        ]
        for token in self._stream_chat(messages, temperature=0.6, max_tokens=256):
            yield token

    def evaluate_response(
        self,
        question: str,
        reference_text: str,
        user_answer: str,
    ) -> dict:
        """
        Dual-layer evaluation of the user's answer:
          Layer 1: Embedding cosine similarity (fast, device-local)
          Layer 2: LLM rubric evaluation (richer feedback)

        Returns a merged result dict with both scores.
        """
        # ── Layer 1: Semantic similarity ────────────────────────────────────
        vecs = self._engine.encode_texts([reference_text, user_answer])
        cos_sim = float(np.dot(vecs[0], vecs[1]))  # already L2-normed
        embedding_score = round(cos_sim * 5, 2)    # scale to 0–5

        # ── Layer 2: LLM rubric ──────────────────────────────────────────────
        messages = [
            {"role": "system", "content": _EVALUATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"REFERENCE TEXT:\n{reference_text[:1200]}\n\n"
                    f"QUESTION ASKED:\n{question}\n\n"
                    f"STUDENT ANSWER:\n{user_answer}"
                ),
            },
        ]

        llm_eval: dict = {}
        try:
            raw = self._complete(messages, temperature=0.1, max_tokens=512).strip()
            # Strip possible markdown code fences
            raw = raw.strip("`").lstrip("json").strip()
            llm_eval = json.loads(raw)
        except (OllamaError, json.JSONDecodeError) as exc:
            log.warning("LLM evaluation failed (%s) — using embedding score only", exc)
            llm_eval = self._fallback_eval(embedding_score)

        # ── Merge ────────────────────────────────────────────────────────────
        llm_score = llm_eval.get("score", round(embedding_score))
        final_score = round(0.35 * embedding_score + 0.65 * llm_score, 2)

        return {
            "finalScore": final_score,      # 0–5
            "llmScore": llm_score,
            "embeddingScore": round(embedding_score, 2),
            "cosineSimilarity": round(cos_sim, 4),
            "grade": llm_eval.get("grade", self._score_to_grade(final_score)),
            "summary": llm_eval.get("summary", ""),
            "strengths": llm_eval.get("strengths", []),
            "gaps": llm_eval.get("gaps", []),
            "correctiveHint": llm_eval.get("corrective_hint", ""),
            "model": cfg.LLM_MODEL,
        }

    def stream_evaluation(
        self,
        question: str,
        reference_text: str,
        user_answer: str,
    ) -> Generator[str, None, None]:
        """
        Streaming evaluation — yields raw tokens so the frontend can show
        the grading reasoning in real time before the final JSON lands.
        The final token block is `__EVAL_RESULT__<json>` so the client can
        parse the structured result when streaming ends.
        """
        messages = [
            {"role": "system", "content": _EVALUATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"REFERENCE TEXT:\n{reference_text[:1200]}\n\n"
                    f"QUESTION ASKED:\n{question}\n\n"
                    f"STUDENT ANSWER:\n{user_answer}"
                ),
            },
        ]

        buffer = ""
        for token in self._stream_chat(messages, temperature=0.1, max_tokens=512):
            buffer += token
            yield token

        # After streaming, compute embedding score and attach to client
        vecs = self._engine.encode_texts([reference_text, user_answer])
        cos_sim = float(np.dot(vecs[0], vecs[1]))
        yield f"\n__EVAL_RESULT__{json.dumps({'cosineSimilarity': round(cos_sim, 4)})}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic_question(text: str) -> str:
        """Fallback question when Ollama is unavailable."""
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
        if sentences:
            return f"Based on the passage, explain the main concept described in: '{sentences[0][:100]}...'"
        return "Summarize the main idea of the passage in your own words."

    @staticmethod
    def _fallback_eval(embedding_score: float) -> dict:
        score = round(embedding_score)
        return {
            "score": max(1, min(5, score)),
            "grade": LLMRouter._score_to_grade(embedding_score),
            "summary": "Evaluation based on semantic similarity only (LLM unavailable).",
            "strengths": [],
            "gaps": [],
            "corrective_hint": "Review the reference material for more details.",
        }

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 4.5:
            return "A"
        if score >= 3.5:
            return "B"
        if score >= 2.5:
            return "C"
        if score >= 1.5:
            return "D"
        return "F"
