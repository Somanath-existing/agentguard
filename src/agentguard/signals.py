"""
agentguard/signals.py
---------------------
Signal engine: three deterministic detectors that run synchronously after
every intercepted span.  No LLM calls — pure math on step history.

Detectors:
  1. Loop detector      — same input hash ≥ 2x in last 10 steps
  2. Info-gain collapse — output size monotonically shrinking ≥ 70% over 4 steps
  3. Error streak       — ≥ 2 consecutive errors

Security notes:
  - All inputs come from StepEvent fields that the interceptor already
    validated/hashed — no raw LLM text enters here.
  - Per-run state is isolated per SignalEngine instance (one per watch() call),
    so concurrent runs can’t bleed state into each other.
  - All mutable state is protected by a threading.Lock to prevent race
    conditions in multi-threaded / async agent environments.
  - The engine exposes a reset() for testing without constructing a new instance.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

from .models import FailureSignal, StepEvent
from .semantic_loop import LoopCheckResult, SemanticLoopDetector

if TYPE_CHECKING:
    from .baseline import BaselineStore


class SignalEngine:
    """
    Stateful per-run signal evaluator.

    Instantiate one per Guard.watch() call — never share across runs.
    """

    def __init__(
        self,
        run_id: str,
        window: int = 10,
        baseline: BaselineStore | None = None,  # Adaptive baseline
        enable_semantic_loop: bool = True,
    ):
        self.run_id = run_id
        self._window = window
        self._baseline = baseline
        self._lock = threading.Lock()

        # Per-run state — completely isolated from other SignalEngine instances
        self._input_hashes: deque[str] = deque(maxlen=window)
        self._output_sizes: deque[int] = deque(maxlen=6)
        self._error_streak: int = 0
        self._recent_events: deque[StepEvent] = deque(maxlen=6)

        # Semantic loop detector v2 — embedding-space trajectory analysis
        self._semantic_loop: SemanticLoopDetector | None = (
            SemanticLoopDetector(
                similarity_threshold=0.88,
                window=window,
                use_sentence_transformers=True,
                velocity_floor=0.08,
                min_stall_steps=2,
            )
            if enable_semantic_loop
            else None
        )

        # Severity ordering for tiebreaking
        self._severity: dict[str, int] = {
            "continue": 0,
            "warn": 1,
            "redirect": 2,
            "abort": 3,
        }

    def evaluate(self, event: StepEvent, raw_input: str | None = None) -> FailureSignal | None:
        """
        Run all detectors. Return the highest-severity signal, or None.
        Called from interceptor.on_end() after every span.
        Thread-safe: acquires _lock for the duration of evaluation.
        """
        with self._lock:
            self._recent_events.append(event)
            candidates = [
                self._check_loop(event, raw_input=raw_input),
                self._check_info_gain(event),
                self._check_error_streak(event),
                self._check_baseline_anomaly(event),
                self._check_predictive_failure(event),
            ]
            real = [s for s in candidates if s is not None]
            if not real:
                return None
            return max(real, key=lambda s: self._severity[s.recommended_action])

    # ------------------------------------------------------------------
    # Detector 1: Loop detection (Exact Hash + Semantic Vector)
    # ------------------------------------------------------------------

    def _check_loop(self, event: StepEvent, raw_input: str | None = None) -> FailureSignal | None:
        # 1. Exact hash check (fast path — catches verbatim repeats)
        if event.input_hash:
            self._input_hashes.append(event.input_hash)
            recent = list(self._input_hashes)
            count = recent.count(event.input_hash)

            if count >= 3:
                return FailureSignal(
                    run_id=self.run_id,
                    step_number=event.step_number,
                    signal_type="loop",
                    confidence=0.95,
                    detail=f"Input hash seen {count}x in last {len(recent)} steps — tool: {event.tool_name}",
                    recommended_action="abort",
                )
            if count == 2:
                return FailureSignal(
                    run_id=self.run_id,
                    step_number=event.step_number,
                    signal_type="loop",
                    confidence=0.65,
                    detail=f"Possible loop: same input seen twice for {event.tool_name}",
                    recommended_action="warn",
                )

        # 2. Semantic trajectory analysis (catches paraphrased / evasion loops)
        if raw_input and self._semantic_loop:
            result: LoopCheckResult | None = self._semantic_loop.check(
                raw_input,
                tool_name=event.tool_name or event.span_name,
            )
            if result is not None and result.is_loop:
                return FailureSignal(
                    run_id=self.run_id,
                    step_number=event.step_number,
                    signal_type="loop",
                    confidence=result.confidence,
                    detail=(
                        f"Semantic loop detected: similarity={result.similarity:.2f}, "
                        f"velocity={result.ema_velocity:.3f}, "
                        f"stalls={result.consecutive_stalls} "
                        f"— tool: {event.tool_name or event.span_name}"
                    ),
                    recommended_action="abort" if result.confidence >= 0.80 else "warn",
                )

        return None

    # ------------------------------------------------------------------
    # Detector 2: Info-gain collapse
    # ------------------------------------------------------------------

    def _check_info_gain(self, event: StepEvent) -> FailureSignal | None:
        self._output_sizes.append(event.output_size)
        sizes = list(self._output_sizes)

        if len(sizes) < 4:
            return None

        last4 = sizes[-4:]
        shrinking = all(last4[i] > last4[i + 1] for i in range(3))
        if not shrinking:
            return None

        first, last = last4[0], last4[-1]
        if first == 0:
            return None  # avoid division-by-zero; can't compute collapse ratio

        collapse_ratio = 1.0 - (last / first)
        if collapse_ratio >= 0.70:
            return FailureSignal(
                run_id=self.run_id,
                step_number=event.step_number,
                signal_type="info_collapse",
                confidence=0.78,
                detail=(
                    f"Output collapsed {collapse_ratio:.0%} over 4 steps "
                    f"({first} → {last} chars) — likely diminishing returns"
                ),
                recommended_action="redirect",
            )
        return None

    # ------------------------------------------------------------------
    # Detector 3: Consecutive error streak
    # ------------------------------------------------------------------

    def _check_error_streak(self, event: StepEvent) -> FailureSignal | None:
        if event.status == "error":
            self._error_streak += 1
        else:
            self._error_streak = 0

        if self._error_streak >= 3:
            return FailureSignal(
                run_id=self.run_id,
                step_number=event.step_number,
                signal_type="tool_error",
                confidence=0.90,
                detail=f"{self._error_streak} consecutive errors — last: {event.error_type}",
                recommended_action="abort",
            )
        if self._error_streak == 2:
            return FailureSignal(
                run_id=self.run_id,
                step_number=event.step_number,
                signal_type="tool_error",
                confidence=0.60,
                detail=f"2 consecutive tool errors — escalation likely needed. Last: {event.error_type}",
                recommended_action="warn",
            )
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Detector 4: Adaptive baseline anomaly
    # ------------------------------------------------------------------

    def _check_baseline_anomaly(self, event: StepEvent) -> FailureSignal | None:
        """
        Check whether any metric in this step is statistically anomalous
        compared to this agent type's historical baseline.

        Calls check_anomaly() BEFORE record() so the current step is tested
        against prior history only.  The baseline is updated after detection.
        Requires _lock to already be held (called from evaluate()).
        """
        if not self._baseline:
            return None

        # NOTE: _lock is already held by evaluate() — BaselineStore has its own
        # internal lock so there is no deadlock risk here.
        anomaly = self._baseline.check_anomaly(event)
        self._baseline.record(event)  # update stats regardless of anomaly

        if not anomaly:
            return None

        # Pick the most extreme metric to drive confidence + action
        worst = max(anomaly.values(), key=lambda x: x["sigma"])
        sigma = worst["sigma"]

        # confidence: scales from 0.5 at threshold up toward 0.95 at 5σ
        confidence = min(0.95, 0.5 + (sigma - self._baseline._threshold) / 10.0)

        return FailureSignal(
            run_id=self.run_id,
            step_number=event.step_number,
            signal_type="drift",
            confidence=round(confidence, 4),
            detail=(
                "Behavioral anomaly detected: "
                + ", ".join(
                    f"{m}={v['value']:.0f} ({v['sigma']:.1f}σ above mean {v['mean']:.0f})"
                    for m, v in anomaly.items()
                )
            ),
            recommended_action="warn" if sigma < 4.0 else "redirect",
        )

    # ------------------------------------------------------------------
    # Detector 5: Predictive Failure Forecasting
    # ------------------------------------------------------------------

    def _check_predictive_failure(self, event: StepEvent) -> FailureSignal | None:
        """
        Predicts agent failures 1-2 steps ahead using trajectory analysis.
        Fires a 'predictive_failure' signal when a critical risk threshold is exceeded.
        """
        # We need at least 3 steps of history to forecast
        events = list(self._recent_events)
        if len(events) < 3:
            return None

        # --- Indicator 1: Loop Convergence (Hash Entropy) ---
        # If hashes are becoming repetitive in the rolling window, a loop is forming.
        hashes = list(self._input_hashes)
        hash_entropy = 1.0
        if len(hashes) >= 4:
            unique_hashes = len(set(hashes))
            hash_entropy = unique_hashes / len(hashes)

        # --- Indicator 2: Imminent Info-gain Collapse ---
        # If output sizes are steadily shrinking, but haven't hit the 70% threshold yet.
        sizes = [e.output_size for e in events[-4:]]
        is_shrinking = len(sizes) >= 3 and all(sizes[i] > sizes[i+1] for i in range(len(sizes)-1))
        approaching_collapse = False
        collapse_ratio = 0.0
        if is_shrinking and sizes[0] > 0:
            collapse_ratio = 1.0 - (sizes[-1] / sizes[0])
            # If we've collapsed between 40% and 70%, we are approaching collapse
            if 0.40 <= collapse_ratio < 0.70:
                approaching_collapse = True

        # --- Indicator 3: Token Burn Acceleration ---
        # Calculate step-over-step token usage changes to spot exponential expansion
        step_tokens = [e.tokens_in + e.tokens_out for e in events]
        token_rates = [step_tokens[i] - step_tokens[i-1] for i in range(1, len(step_tokens))]
        accelerating_burn = False
        if len(token_rates) >= 2 and all(r > 0 for r in token_rates):
            # The rate of token usage growth is positive and increasing
            if token_rates[-1] > token_rates[-2]:
                accelerating_burn = True

        # --- Risk Scoring & Forecasting Decision ---
        risk_score = 0.0
        reasons = []

        if hash_entropy <= 0.6:  # High repetition
            risk_score += 0.45
            reasons.append(f"high input repetition (entropy {hash_entropy:.2f})")

        if approaching_collapse:
            risk_score += 0.35
            reasons.append(f"approaching output collapse ({collapse_ratio:.0%})")

        if accelerating_burn:
            risk_score += 0.20
            reasons.append("exponential token consumption rate")

        if risk_score >= 0.50:
            return FailureSignal(
                run_id=self.run_id,
                step_number=event.step_number,
                signal_type="predictive_failure",
                confidence=round(risk_score, 2),
                detail=f"Trajectory warning: {', '.join(reasons)}",
                recommended_action="warn",  # proactive caution
            )

        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state. Useful for testing without creating a new instance."""
        with self._lock:
            self._input_hashes.clear()
            self._output_sizes.clear()
            self._error_streak = 0
            self._recent_events.clear()
            if self._semantic_loop:
                self._semantic_loop.reset()
