"""
agentguard/taint.py
-------------------
Step-level semantic taint tracking for LLM agent tool chains.

Builds a provenance DAG: each step records which prior steps' outputs
appear in its input (detected via content hash overlap).  When an injection
or anomaly is detected at any step, the tracker can:
  1. Trace forward to find ALL downstream steps that consumed tainted data.
  2. Return the clean prefix of steps unaffected by the taint source.
  3. Provide a targeted rollback boundary instead of aborting the full session.

This addresses the #1 unsolved problem in agent security: tracking how
untrusted data (from tool responses) propagates through an agent's
decision chain.  Traditional taint tracking relies on memory addresses;
LLMs have no addresses, but we CAN track data flow at the step/span
boundary using output hash fingerprints.

Security notes:
  - Raw content is NEVER stored.  Only 64-char SHA-256 hex digests of
    output chunks are retained for matching.
  - The DAG is append-only — no mutation of prior entries.
  - Thread-safe: single threading.Lock on all mutations.

Usage:
    tracker = TaintTracker()
    tracker.on_step_output(step=1, output_text="result from web search...")
    tracker.on_step_input(step=2, input_text="...contains step 1 result...")
    # Step 2 is now tainted by step 1

    # When injection detected at step 1:
    tainted = tracker.trace_forward(source_step=1)
    # tainted = [2, 3, 5]  — all steps that consumed data from step 1

    clean_prefix = tracker.clean_prefix(source_step=1)
    # Returns step numbers that are NOT tainted
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Chunk size for fingerprinting output text.
# We hash overlapping sliding windows of this size to detect partial
# inclusion of one step's output in another step's input.
_CHUNK_SIZE = 64


@dataclass(frozen=True)
class TaintLabel:
    """Immutable record of a taint relationship."""
    source_step: int         # which step introduced external data
    source_tool: str         # which tool produced the tainted output
    confidence: float        # 1.0 = exact chunk match; lower for partial
    propagation_chain: tuple[int, ...]  # ordered path from source to this step


class TaintTracker:
    """
    Step-level information flow tracking via content fingerprinting.

    For each step, we store a set of SHA-256 digests of fixed-size chunks
    from its output.  When a subsequent step's input contains any of those
    same chunks, we record a taint edge: source_step → consumer_step.

    This forms a DAG that allows forward tracing (from a compromised step
    to everything it influenced) and backward tracing (from any step to
    its data sources).
    """

    def __init__(self, chunk_size: int = _CHUNK_SIZE):
        self._chunk_size = max(16, chunk_size)
        self._lock = threading.Lock()

        # step_number → set of content chunk hashes (from output)
        self._output_fingerprints: dict[int, set[str]] = {}

        # step_number → tool name that produced the output
        self._step_tools: dict[int, str] = {}

        # forward edges: source_step → set of consumer steps
        self._forward_edges: dict[int, set[int]] = defaultdict(set)

        # backward edges: consumer_step → set of source steps
        self._backward_edges: dict[int, set[int]] = defaultdict(set)

        # all known step numbers in order
        self._steps: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_step_output(
        self,
        step: int,
        output_text: str,
        tool_name: str = "unknown",
    ) -> None:
        """
        Record the output fingerprints for a step.
        Call this when a tool or LLM produces output text.

        The raw text is NOT stored — only chunk hashes.
        """
        if not output_text:
            return

        fingerprints = self._compute_fingerprints(output_text)

        with self._lock:
            self._output_fingerprints[step] = fingerprints
            self._step_tools[step] = tool_name
            if step not in self._steps:
                self._steps.append(step)

        logger.debug(
            "TaintTracker: recorded %d fingerprints for step %d (tool=%s)",
            len(fingerprints), step, tool_name,
        )

    def on_step_input(self, step: int, input_text: str) -> list[TaintLabel]:
        """
        Check whether this step's input contains content from any prior step's
        output.  Records taint edges for any matches found.

        Returns a list of TaintLabel objects describing the taint sources.
        The raw text is NOT stored.
        """
        if not input_text:
            return []

        input_fingerprints = self._compute_fingerprints(input_text)
        taint_labels: list[TaintLabel] = []

        with self._lock:
            for prior_step, prior_fps in self._output_fingerprints.items():
                if prior_step >= step:
                    continue  # only look at prior steps

                overlap = input_fingerprints & prior_fps
                if overlap:
                    # Confidence: proportion of prior step's fingerprints found in input
                    confidence = min(1.0, len(overlap) / max(len(prior_fps), 1))

                    # Record edges
                    self._forward_edges[prior_step].add(step)
                    self._backward_edges[step].add(prior_step)

                    # Build propagation chain: trace backward from prior_step
                    chain = self._trace_backward_chain(prior_step)

                    label = TaintLabel(
                        source_step=prior_step,
                        source_tool=self._step_tools.get(prior_step, "unknown"),
                        confidence=round(confidence, 4),
                        propagation_chain=tuple(chain + [step]),
                    )
                    taint_labels.append(label)

                    logger.debug(
                        "TaintTracker: step %d tainted by step %d (%d/%d chunks matched, conf=%.2f)",
                        step, prior_step, len(overlap), len(prior_fps), confidence,
                    )

        return taint_labels

    def trace_forward(self, source_step: int) -> list[int]:
        """
        Given a compromised step, return ALL downstream steps that
        consumed its data (directly or transitively).

        Uses BFS through the forward edge graph.
        """
        with self._lock:
            visited: set[int] = set()
            queue = [source_step]

            while queue:
                current = queue.pop(0)
                for child in self._forward_edges.get(current, set()):
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)

            return sorted(visited)

    def trace_backward(self, step: int) -> list[int]:
        """
        Given a step, return all upstream steps whose data influenced it
        (directly or transitively).
        """
        with self._lock:
            visited: set[int] = set()
            queue = [step]

            while queue:
                current = queue.pop(0)
                for parent in self._backward_edges.get(current, set()):
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)

            return sorted(visited)

    def clean_prefix(self, source_step: int) -> list[int]:
        """
        Return the list of step numbers NOT tainted by the source step.
        This defines the safe rollback boundary.
        """
        tainted = set(self.trace_forward(source_step))
        tainted.add(source_step)

        with self._lock:
            return sorted(s for s in self._steps if s not in tainted)

    def is_tainted(self, step: int) -> bool:
        """Check if a specific step has any taint sources."""
        with self._lock:
            return bool(self._backward_edges.get(step))

    def taint_sources(self, step: int) -> list[int]:
        """Return the direct taint sources for a step."""
        with self._lock:
            return sorted(self._backward_edges.get(step, set()))

    def summary(self) -> dict:
        """Return a diagnostic snapshot of the taint graph."""
        with self._lock:
            return {
                "total_steps": len(self._steps),
                "tainted_steps": len([s for s in self._steps if self._backward_edges.get(s)]),
                "edges": {
                    str(src): sorted(dsts)
                    for src, dsts in self._forward_edges.items()
                },
                "steps": {
                    str(s): {
                        "tool": self._step_tools.get(s, "unknown"),
                        "fingerprints": len(self._output_fingerprints.get(s, set())),
                        "tainted_by": sorted(self._backward_edges.get(s, set())),
                        "taints": sorted(self._forward_edges.get(s, set())),
                    }
                    for s in self._steps
                },
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fingerprints(self, text: str) -> set[str]:
        """
        Compute SHA-256 digests from word 3-grams and overlapping byte windows.
        Ensures robust matching regardless of byte boundary alignment.
        """
        fingerprints: set[str] = set()
        if not text:
            return fingerprints

        # 1. Word n-grams (3 consecutive words)
        words = [w.strip().lower() for w in text.split() if w.strip()]
        if len(words) >= 3:
            for i in range(len(words) - 2):
                ngram = " ".join(words[i : i + 3])
                fp = hashlib.sha256(ngram.encode("utf-8")).hexdigest()
                fingerprints.add(fp)
        elif words:
            fp = hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()
            fingerprints.add(fp)

        # 2. Overlapping sliding byte chunks
        text_bytes = text.encode("utf-8", errors="replace")
        stride = max(8, self._chunk_size // 2)
        for i in range(0, len(text_bytes), stride):
            chunk = text_bytes[i : i + self._chunk_size]
            if len(chunk) >= 12:
                fp = hashlib.sha256(chunk).hexdigest()
                fingerprints.add(fp)

        return fingerprints

    def _trace_backward_chain(self, step: int) -> list[int]:
        """
        Build the full backward propagation chain for a step.
        Returns a list ordered from earliest source to the given step.
        Must be called while holding _lock.
        """
        chain: list[int] = []
        visited: set[int] = set()
        queue = [step]

        while queue:
            current = queue.pop(0)
            for parent in self._backward_edges.get(current, set()):
                if parent not in visited:
                    visited.add(parent)
                    chain.append(parent)
                    queue.append(parent)

        return sorted(chain) + [step]
