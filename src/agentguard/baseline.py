"""
agentguard/baseline.py
----------------------
Adaptive behavioral baseline per agent type.

Instead of hardcoded thresholds ("output shrunk 70%"), we learn what
"normal" looks like for each agent type across many runs and flag
statistical anomalies (> N standard deviations from the learned mean).

Algorithm: Welford's online algorithm for numerically stable running
mean and variance.  O(1) memory — no raw history is stored.

Persistence: stats are saved to a JSON file per agent type so baselines
survive process restarts and accumulate across many agent runs.

Thread safety: a threading.Lock protects all mutable state.

Usage:
    baseline = BaselineStore(agent_type="web_search_agent")
    anomaly = baseline.check_anomaly(step_event)   # call BEFORE record()
    baseline.record(step_event)
    if anomaly:
        # anomaly = {"output_size": {"value": 10, "mean": 850.5, "sigma": 3.2}, ...}
        ...
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Welford running stats ─────────────────────────────────────────────────────

@dataclass
class RunningStats:
    """
    Welford's online algorithm: O(1) memory, numerically stable mean + variance.

    Minimum 10 observations before anomaly detection is active — avoids false
    positives during warm-up when the distribution is still poorly defined.
    """
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0   # sum of squared deviations from mean

    MIN_SAMPLES: int = field(default=10, init=False, repr=False)

    def update(self, value: float) -> None:
        """Incrementally update mean and M2 with a new observation."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        """Unbiased sample variance (Bessel-corrected). 0.0 if n < 2."""
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        """
        How many standard deviations is `value` from the running mean?
        Returns 0.0 if std is zero or insufficient samples have been seen,
        preventing spurious alerts during warm-up.
        """
        if self.n < self.MIN_SAMPLES or self.std == 0:
            return 0.0
        return abs(value - self.mean) / self.std

    # Serialization for JSON persistence
    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_dict(cls, d: dict) -> RunningStats:
        s = cls()
        s.n = int(d.get("n", 0))
        s.mean = float(d.get("mean", 0.0))
        s.M2 = float(d.get("M2", 0.0))
        return s


# ── Baseline store ────────────────────────────────────────────────────────────

class BaselineStore:
    """
    Persists per-agent-type behavioral stats to a JSON file.

    One BaselineStore instance per agent_type.  Multiple Guard.watch() calls
    for the same agent_type share the same underlying stats file, allowing the
    baseline to accumulate across many runs.

    Metrics tracked:
        output_size  — character count of LLM completions
        tokens_in    — input token count per step
        tokens_out   — output token count per step
        latency_ms   — span duration in milliseconds
    """

    _METRIC_KEYS = ("output_size", "tokens_in", "tokens_out", "latency_ms")

    def __init__(
        self,
        agent_type: str,
        base_dir: str = "./agentguard_baselines",
        anomaly_threshold_sigma: float = 2.5,
    ):
        """
        Args:
            agent_type: Logical name for the agent (e.g. "web_search_agent").
                        Used as the filename key — must be consistent across runs.
            base_dir:   Directory to store per-agent-type JSON baseline files.
            anomaly_threshold_sigma: Z-score above which a metric is flagged.
                        2.5 ≈ top 1.2% of a normal distribution.
        """
        self._agent_type = agent_type
        self._threshold = anomaly_threshold_sigma
        self._lock = threading.Lock()

        os.makedirs(base_dir, exist_ok=True)

        # Build a safe filename: keep only alphanumeric + underscore, cap at 64 chars
        safe_name = "".join(c if c.isalnum() else "_" for c in agent_type)[:64]
        self._path = os.path.join(base_dir, f"{safe_name}.json")

        self._stats: dict[str, RunningStats] = {k: RunningStats() for k in self._METRIC_KEYS}
        self._load()

        logger.debug(
            "BaselineStore loaded for agent_type=%r (path=%s, samples=%d)",
            agent_type,
            self._path,
            self._stats["output_size"].n,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event) -> None:
        """
        Update running stats with metrics from a StepEvent.
        Call AFTER check_anomaly() so the current event doesn't bias its own check.
        Thread-safe. Writes to disk every 5 observations or on explicit flush.
        """
        with self._lock:
            self._stats["output_size"].update(float(event.output_size))
            self._stats["tokens_in"].update(float(event.tokens_in))
            self._stats["tokens_out"].update(float(event.tokens_out))
            self._stats["latency_ms"].update(float(event.latency_ms))
            # Batch writes to disk every 5 observations
            if self._stats["output_size"].n % 5 == 0:
                self._save()

    def check_anomaly(self, event) -> dict | None:
        """
        Check whether any metric in `event` is statistically anomalous
        compared to this agent type's historical baseline.

        Returns:
            A dict of anomalous metrics with their values, means, and sigma,
            or None if everything is within threshold.

        Call BEFORE record() to test against prior observations only.
        Thread-safe.
        """
        with self._lock:
            checks = {
                "output_size": float(event.output_size),
                "tokens_in":   float(event.tokens_in),
                "tokens_out":  float(event.tokens_out),
                "latency_ms":  float(event.latency_ms),
            }
            anomalies: dict = {}
            for metric, value in checks.items():
                z = self._stats[metric].z_score(value)
                if z > self._threshold:
                    anomalies[metric] = {
                        "value":    value,
                        "mean":     round(self._stats[metric].mean, 2),
                        "std":      round(self._stats[metric].std, 2),
                        "sigma":    round(z, 2),
                        "samples":  self._stats[metric].n,
                    }
            return anomalies if anomalies else None

    def flush(self) -> None:
        """Force save stats to disk."""
        with self._lock:
            self._save()

    @property
    def sample_count(self) -> int:
        """Total observations recorded for the primary metric (output_size)."""
        with self._lock:
            return self._stats["output_size"].n

    def summary(self) -> dict:
        """Return a snapshot of all running stats for diagnostics."""
        with self._lock:
            return {
                "agent_type": self._agent_type,
                "threshold_sigma": self._threshold,
                "metrics": {
                    k: {
                        "n":    s.n,
                        "mean": round(s.mean, 2),
                        "std":  round(s.std, 2),
                    }
                    for k, s in self._stats.items()
                },
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write current stats to disk. Called inside _lock."""
        data = {k: v.to_dict() for k, v in self._stats.items()}
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("BaselineStore: could not save stats to %s: %s", self._path, exc)

    def _load(self) -> None:
        """Load stats from disk if the file exists. Called at construction."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k in self._stats:
                    self._stats[k] = RunningStats.from_dict(v)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "BaselineStore: could not load stats from %s (%s) — starting fresh",
                self._path,
                exc,
            )
