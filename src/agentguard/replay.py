"""
agentguard/replay.py
--------------------
Causal replay: binary-search through a stored run to find the root-cause step.

When a run aborts you can see the signal (e.g. "loop at step 8") but not which
earlier tool call first caused it.  CausalReplayer re-runs the signal detection
logic over stored CheckpointStore steps to binary-search for the minimal prefix
that reproduces the failure.

Complexity: O(log N) signal-engine passes instead of O(N).

Usage:
    from agentguard.replay import CausalReplayer

    # After an aborted session:
    replayer = CausalReplayer(store)
    result = replayer.find_root_cause()
    # {
    #   'root_cause_step': 4,
    #   'tool_name': 'web_search',
    #   'signal_type': 'loop',
    #   'replay_confidence': 0.95,
    #   'explanation': "Removing steps after step 4 eliminates the 'loop' signal."
    # }
"""
from __future__ import annotations

import logging
from typing import Optional

from .signals import SignalEngine
from .state import CheckpointStore

logger = logging.getLogger(__name__)


class CausalReplayer:
    """
    Given a CheckpointStore with recorded steps, finds the minimal step-prefix
    that reproduces the failure signal observed in the full run.

    Binary search strategy:
        - lo = 0, hi = len(steps) - 1
        - At each midpoint, replay steps[0..mid] through a fresh SignalEngine
        - If a signal fires → try a shorter prefix (hi = mid - 1)
        - If no signal fires → need more steps (lo = mid + 1)
        - Smallest prefix that fires = root cause

    Thread safety: CausalReplayer is stateless — all state lives in the local
    SignalEngine instances created per _replay() call.  Safe to call from any
    thread after the run has completed.
    """

    def __init__(self, store: CheckpointStore, baseline=None):
        self._store = store
        self._baseline = baseline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_root_cause(self) -> dict:
        """
        Perform causal replay. Returns a result dict describing the root-cause
        step, or an error dict if replay cannot reproduce the signal.

        This should be called AFTER the agent run has ended (inside or after
        the with-block's except handler).
        """
        steps = self._store.all_steps()
        if not steps:
            return {"error": "No steps recorded in CheckpointStore"}

        if len(steps) == 1:
            # Only one step — trivially the root cause if it produced a signal
            signal = self._replay(steps)
            if signal:
                return self._build_result(steps[0], signal)
            return {"error": "Single step recorded but no signal reproduced"}

        # Sanity check: does a full replay even produce any signal?
        full_signal = self._replay(steps)
        if not full_signal:
            return {
                "error": (
                    "No signal reproduced on full replay. "
                    "The failure may be non-deterministic or data-dependent."
                )
            }

        logger.debug(
            "Full replay produced signal '%s'. Beginning binary search over %d steps.",
            full_signal.signal_type,
            len(steps),
        )

        # Binary search: find smallest prefix that triggers a signal
        lo, hi = 0, len(steps) - 1
        root_step_idx = hi  # default: last step

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_signal = self._replay(steps[: mid + 1])

            if candidate_signal:
                # This prefix triggers a signal — try shorter
                root_step_idx = mid
                hi = mid - 1
            else:
                # This prefix is clean — need more steps
                lo = mid + 1

        root = steps[root_step_idx]
        logger.info(
            "Root cause found: step %d (tool=%s, signal=%s)",
            root.step_number,
            root.tool_name,
            full_signal.signal_type,
        )

        return self._build_result(root, full_signal)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replay(self, steps: list) -> Optional[object]:
        """
        Run a fresh SignalEngine over a step sequence.
        Returns the first FailureSignal that would trigger a guard action,
        or None if no signal fires across the entire sequence.
        """
        if not steps:
            return None

        engine = SignalEngine(run_id=steps[0].run_id, baseline=self._baseline)
        for step in steps:
            result = engine.evaluate(step)
            if result and result.recommended_action in ("warn", "redirect", "abort"):
                return result
        return None

    @staticmethod
    def _build_result(root_step, signal) -> dict:
        return {
            "root_cause_step": root_step.step_number,
            "tool_name": root_step.tool_name,
            "span_name": root_step.span_name,
            "signal_type": signal.signal_type,
            "recommended_action": signal.recommended_action,
            "input_hash": root_step.input_hash,  # hash only — no raw content
            "replay_confidence": signal.confidence,
            "explanation": (
                f"Removing steps after step {root_step.step_number} "
                f"eliminates the '{signal.signal_type}' signal. "
                f"Tool involved: {root_step.tool_name or 'unknown'}."
            ),
        }
