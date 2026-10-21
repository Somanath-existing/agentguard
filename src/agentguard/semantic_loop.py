"""
agentguard/semantic_loop.py
---------------------------
Evasion-Resistant Semantic Loop Detection (v2).

Detects stuck / looping agents by modelling their behaviour as a trajectory
through embedding space.  Three complementary detection mechanisms:

  1. **Semantic Velocity** — EMA of step-over-step cosine distance.  When
     velocity collapses below `velocity_floor` for `min_stall_steps`
     consecutive steps, the agent is stuck.

  2. **Tool-Stratified Similarity** — per-tool embedding histories so that
     legitimate multi-tool pipelines (search → summarize → search) are not
     penalised for semantic overlap across different tools.

  3. **Graduated Confidence** — confidence scales with consecutive stall
     count, giving the recovery router better signal quality.

Embedding backend:
  - Default: sentence-transformers (all-MiniLM-L6-v2) — 384-dim, <10ms,
    real semantic understanding.
  - Fallback: character n-gram TF-IDF vectorizer with a conservative 0.95
    threshold (fires only on near-identical text).

Thread-safe: threading.Lock on all state mutations.
O(1) rolling window storage per tool (deque of pre-computed vectors).

Usage:
    detector = SemanticLoopDetector()
    result = detector.check("Search for AI safety papers", tool_name="web_search")
    result = detector.check("Search for AI safety papers", tool_name="web_search")
    # result.is_loop == True, result.consecutive_stalls == 2
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context normalization: extract active turn from multi-turn buffers
# ---------------------------------------------------------------------------

def extract_active_prompt(raw: str) -> str:
    """
    Extract the active / latest prompt or message from a potentially cumulative
    chat history buffer (e.g. OpenTelemetry gen_ai.input.messages, LangChain buffers).

    Autonomous agents and multi-turn chatbots accumulate prior conversation turns,
    tool responses, and system instructions in their message arrays. Comparing the
    entire cumulative buffer results in >95% artificial similarity and false loop
    signals. This function extracts the latest active turn for accurate trajectory
    and velocity tracking.
    """
    if not raw or not isinstance(raw, str):
        return str(raw or "").strip()

    s = raw.strip()

    # 1. Structured JSON (OpenTelemetry GenAI semantic conventions)
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                if "messages" in data and isinstance(data["messages"], list):
                    data = data["messages"]
                elif "prompt" in data and isinstance(data["prompt"], str):
                    return data["prompt"].strip()
                elif "input" in data and isinstance(data["input"], str):
                    return data["input"].strip()

            if isinstance(data, list) and data:
                # Walk backwards to find the last message with content
                # Prioritize 'user', 'tool', or 'human' messages if available
                for msg in reversed(data):
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "")).lower()
                        content = _extract_msg_content(msg)
                        if content and role in ("user", "tool", "human"):
                            return content
                # Fall back to the very last message with content
                for msg in reversed(data):
                    if isinstance(msg, dict):
                        content = _extract_msg_content(msg)
                        if content:
                            return content
                    elif isinstance(msg, str) and msg.strip():
                        return msg.strip()
        except Exception:
            pass

    # 2. Text with multi-turn role markers (e.g. "User: ... \nAssistant: ...")
    role_pattern = r"(?:^|\n)(?:User|Human|user|human)\s*:\s*(.+?)(?=(?:\n(?:Assistant|AI|System|User|Human|system|assistant)\s*:)|$)"
    matches = list(re.finditer(role_pattern, s, flags=re.DOTALL | re.IGNORECASE))
    if matches:
        last_match = matches[-1].group(1).strip()
        if last_match:
            return last_match

    # 3. ChatML markers: <|im_start|>user\n...<|im_end|>
    chatml_pattern = r"<\|im_start\|>user\s*\n?(.*?)(?:<\|im_end\|>|$)"
    chatml_matches = list(re.finditer(chatml_pattern, s, flags=re.DOTALL | re.IGNORECASE))
    if chatml_matches:
        last_match = chatml_matches[-1].group(1).strip()
        if last_match:
            return last_match

    # 4. Llama [INST] markers: [INST] ... [/INST]
    inst_pattern = r"\[INST\]\s*(.*?)\s*\[/INST\]"
    inst_matches = list(re.finditer(inst_pattern, s, flags=re.DOTALL))
    if inst_matches:
        last_match = inst_matches[-1].group(1).strip()
        if last_match:
            return last_match

    return s


def _extract_msg_content(msg: dict) -> str:
    """Helper to extract text from a message dict across providers."""
    # OpenTelemetry / Anthropic parts format: [{"content": "...", "type": "text"}]
    if "parts" in msg and isinstance(msg["parts"], list):
        text_parts = []
        for p in msg["parts"]:
            if isinstance(p, dict) and "content" in p:
                text_parts.append(str(p["content"]))
            elif isinstance(p, str):
                text_parts.append(p)
        joined = " ".join(text_parts).strip()
        if joined:
            return joined

    # Standard OpenAI format: {"content": "..."}
    if "content" in msg and msg["content"] is not None:
        if isinstance(msg["content"], str):
            return msg["content"].strip()
        elif isinstance(msg["content"], list):
            # OpenAI multimodal or multipart list
            parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts).strip()

    # Tool calls
    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
        calls = []
        for tc in msg["tool_calls"]:
            if isinstance(tc, dict) and "function" in tc:
                fn = tc["function"]
                calls.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
        if calls:
            return " ".join(calls)

    return ""


# ---------------------------------------------------------------------------
# Result type — richer than a bare Optional[float]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopCheckResult:
    """Immutable result from SemanticLoopDetector.check()."""
    is_loop: bool
    similarity: float              # max cosine similarity to recent same-tool steps
    consecutive_stalls: int        # how many consecutive steps had velocity < floor
    velocity: float                # current semantic velocity (1 - sim to previous)
    ema_velocity: float            # smoothed velocity (EMA)
    confidence: float              # graduated confidence score (0.0 – 1.0)
    tool_name: str | None       # tool context this was evaluated against


# ---------------------------------------------------------------------------
# Lightweight fallback: character n-gram TF-IDF
# ---------------------------------------------------------------------------

def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse term-frequency dicts."""
    if not vec1 or not vec2:
        return 0.0

    intersection = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in intersection)

    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _char_ngrams(text: str, n: int = 3) -> dict[str, float]:
    """Generate normalized character n-gram frequencies for fast text comparison."""
    clean = text.lower().strip()
    if len(clean) < n:
        return {clean: 1.0} if clean else {}

    ngrams = [clean[i : i + n] for i in range(len(clean) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Per-tool embedding history
# ---------------------------------------------------------------------------

@dataclass
class _ToolHistory:
    """Rolling window of embeddings for a single tool context."""
    embeddings: deque = field(default_factory=lambda: deque(maxlen=10))
    texts: deque = field(default_factory=lambda: deque(maxlen=10))
    last_embedding: object = None  # most recent embedding for velocity calc


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class SemanticLoopDetector:
    """
    Detects stuck / looping agents via embedding-space trajectory analysis.

    Thread-safe — all state mutations are protected by a lock.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        window: int = 10,
        use_sentence_transformers: bool = True,
        velocity_floor: float = 0.08,
        min_stall_steps: int = 2,
        ema_alpha: float = 0.3,
        fallback_threshold: float = 0.95,
    ):
        """
        Args:
            similarity_threshold: Cosine similarity above which two embeddings
                are considered semantically equivalent.
            window: Rolling window size per tool.
            use_sentence_transformers: If True (default), attempt to load
                sentence-transformers for real semantic embeddings.
            velocity_floor: EMA velocity below this triggers stall detection.
            min_stall_steps: Number of consecutive stalls before flagging a loop.
            ema_alpha: Smoothing factor for EMA velocity (0 < α ≤ 1).
                Higher = more responsive, lower = more stable.
            fallback_threshold: Similarity threshold for the n-gram fallback
                (should be high — 0.95 — to avoid false positives).
        """
        self._threshold = similarity_threshold
        self._window = window
        self._velocity_floor = velocity_floor
        self._min_stall_steps = min_stall_steps
        self._ema_alpha = ema_alpha
        self._fallback_threshold = fallback_threshold

        self._lock = threading.Lock()

        # Tool-stratified histories
        self._tool_histories: dict[str, _ToolHistory] = {}

        # Global velocity tracking (across all tools, for overall stuck detection)
        self._ema_velocity: float = 1.0  # start assuming healthy
        self._consecutive_stalls: int = 0
        self._last_global_embedding: object = None

        # Sentence-transformers backend
        self._use_st = False
        self._st_model = None

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._use_st = True
                logger.info(
                    "SemanticLoopDetector v2: initialized with sentence-transformers "
                    "(all-MiniLM-L6-v2) — velocity_floor=%.2f, min_stall=%d",
                    velocity_floor, min_stall_steps,
                )
            except Exception as exc:
                logger.debug(
                    "SemanticLoopDetector v2: sentence-transformers unavailable (%s), "
                    "falling back to n-gram TF-IDF (threshold=%.2f)",
                    exc, fallback_threshold,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        input_text: str,
        tool_name: str | None = None,
    ) -> LoopCheckResult | None:
        """
        Analyse a new agent step and return a LoopCheckResult if a loop is
        detected, or None if the agent's trajectory is healthy.

        Args:
            input_text: The raw input / prompt for this step.
            tool_name: Optional tool context. When provided, similarity is
                only compared against prior invocations of the same tool.

        Returns:
            LoopCheckResult if a loop/stall is detected, None otherwise.
        """
        if not input_text or not input_text.strip():
            return None

        # Normalize input: extract active prompt to eliminate cumulative history dilution
        active_text = extract_active_prompt(input_text)
        if not active_text:
            return None

        with self._lock:
            if self._use_st and self._st_model is not None:
                return self._check_with_embeddings(active_text, tool_name)
            else:
                return self._check_with_ngrams(active_text, tool_name)

    def reset(self) -> None:
        """Clear all stored window state."""
        with self._lock:
            self._tool_histories.clear()
            self._ema_velocity = 1.0
            self._consecutive_stalls = 0
            self._last_global_embedding = None

    # ------------------------------------------------------------------
    # Embedding-based detection (primary path)
    # ------------------------------------------------------------------

    def _check_with_embeddings(
        self, input_text: str, tool_name: str | None
    ) -> LoopCheckResult | None:
        """
        Full detection pipeline using sentence-transformer embeddings.

        Steps:
          1. Encode the input
          2. Compute velocity (distance from previous same-tool or global step)
          3. Update EMA velocity
          4. Check tool-stratified similarity
          5. Update consecutive stall counter
          6. Return LoopCheckResult if stalled
        """
        import numpy as np

        try:
            curr_emb = self._st_model.encode(input_text, normalize_embeddings=True)
        except Exception as exc:
            logger.warning("SemanticLoopDetector: encoding failed (%s), skipping", exc)
            return None

        tool_key = tool_name or "__global__"
        history = self._tool_histories.setdefault(
            tool_key, _ToolHistory(
                embeddings=deque(maxlen=self._window),
                texts=deque(maxlen=self._window),
            )
        )

        # --- Step 1: Velocity against prior same-tool (or global) step ---
        prev_emb = history.last_embedding if history.last_embedding is not None else self._last_global_embedding
        if prev_emb is not None:
            sim_to_prev = float(np.dot(curr_emb, prev_emb))
            velocity = max(0.0, 1.0 - sim_to_prev)
            self._ema_velocity = (
                self._ema_alpha * velocity
                + (1.0 - self._ema_alpha) * self._ema_velocity
            )
        else:
            velocity = 1.0
            self._ema_velocity = 1.0

        # --- Step 2: Tool-stratified similarity ---
        max_sim = 0.0
        for prior_emb in history.embeddings:
            sim = float(np.dot(curr_emb, prior_emb))
            if sim > max_sim:
                max_sim = sim

        # Record this step
        history.embeddings.append(curr_emb)
        history.texts.append(input_text)
        history.last_embedding = curr_emb
        self._last_global_embedding = curr_emb

        # --- Step 3: Stall detection ---
        is_stalled = (
            (velocity < self._velocity_floor or self._ema_velocity < self._velocity_floor)
            and max_sim >= self._threshold
        )

        if is_stalled:
            self._consecutive_stalls += 1
        else:
            self._consecutive_stalls = 0

        # --- Step 4: Graduated confidence ---
        if self._consecutive_stalls >= self._min_stall_steps:
            confidence = min(
                0.98,
                0.57 + (self._consecutive_stalls * 0.08),
            )
            return LoopCheckResult(
                is_loop=True,
                similarity=round(max_sim, 4),
                consecutive_stalls=self._consecutive_stalls,
                velocity=round(velocity, 4),
                ema_velocity=round(self._ema_velocity, 4),
                confidence=round(confidence, 4),
                tool_name=tool_name,
            )

        return None

    # ------------------------------------------------------------------
    # N-gram fallback (no sentence-transformers)
    # ------------------------------------------------------------------

    def _check_with_ngrams(
        self, input_text: str, tool_name: str | None
    ) -> LoopCheckResult | None:
        """
        Conservative n-gram fallback with higher threshold (0.95).

        This path fires only on near-identical text to avoid false positives
        from common English word overlap. It still does tool-stratified
        comparison and velocity tracking.
        """
        curr_vec = _char_ngrams(input_text)

        tool_key = tool_name or "__global__"
        history = self._tool_histories.setdefault(
            tool_key, _ToolHistory(
                embeddings=deque(maxlen=self._window),
                texts=deque(maxlen=self._window),
            )
        )

        # Velocity from previous step (using n-gram vectors)
        prev_emb = history.last_embedding if history.last_embedding is not None else self._last_global_embedding
        if prev_emb is not None and isinstance(prev_emb, dict):
            sim_to_prev = _cosine_similarity(curr_vec, prev_emb)
            velocity = max(0.0, 1.0 - sim_to_prev)
            self._ema_velocity = (
                self._ema_alpha * velocity
                + (1.0 - self._ema_alpha) * self._ema_velocity
            )
        else:
            velocity = 1.0
            self._ema_velocity = 1.0

        # Tool-stratified max similarity
        max_sim = 0.0
        for prior_vec in history.embeddings:
            if isinstance(prior_vec, dict):
                sim = _cosine_similarity(curr_vec, prior_vec)
                if sim > max_sim:
                    max_sim = sim

        # Record
        history.embeddings.append(curr_vec)
        history.texts.append(input_text)
        history.last_embedding = curr_vec
        self._last_global_embedding = curr_vec

        # Stall detection — higher threshold for n-grams
        is_stalled = (
            (velocity < self._velocity_floor or self._ema_velocity < self._velocity_floor)
            and max_sim >= self._fallback_threshold
        )

        if is_stalled:
            self._consecutive_stalls += 1
        else:
            self._consecutive_stalls = 0

        if self._consecutive_stalls >= self._min_stall_steps:
            confidence = min(
                0.98,
                0.57 + (self._consecutive_stalls * 0.08),
            )
            return LoopCheckResult(
                is_loop=True,
                similarity=round(max_sim, 4),
                consecutive_stalls=self._consecutive_stalls,
                velocity=round(velocity, 4),
                ema_velocity=round(self._ema_velocity, 4),
                confidence=round(confidence, 4),
                tool_name=tool_name,
            )

        return None
