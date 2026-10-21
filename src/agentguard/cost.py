"""
agentguard/cost.py
------------------
Token-budget enforcement and cost projection.

Security / correctness fixes vs original guide:
  1. Pricing is loaded from AGENTGUARD_PRICING_FILE env var (JSON) if set,
     so users aren't silently stuck with stale hardcoded rates after a price
     change.  Falls back to DEFAULT_MODEL_COSTS.
  2. CostGuardian is constructed with an explicit model name so cost_usd()
     doesn't need to guess.
  3. projected_total() guards against division-by-zero when history is empty.
  4. Budget thresholds (warn_pct, hard_pct) are configurable — different
     use-cases have different risk tolerances.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from .models import FailureSignal, StepEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pricing (per 1 M tokens, USD)
# Update by setting AGENTGUARD_PRICING_FILE to a JSON file with the same shape
# ---------------------------------------------------------------------------

DEFAULT_MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-6":           {"in":  5.00, "out": 25.00},
    "claude-sonnet-4-6":         {"in":  3.00, "out": 15.00},
    "claude-haiku-4-5-20251001": {"in":  1.00, "out":  5.00},
    "gpt-4o":                    {"in":  5.00, "out": 15.00},
    "gpt-4o-mini":               {"in":  0.15, "out":  0.60},
}


def _load_model_costs() -> dict[str, dict[str, float]]:
    """
    Load pricing data. If AGENTGUARD_PRICING_FILE is set, validate that
    the JSON matches the expected schema before using it.
    """
    path = os.environ.get("AGENTGUARD_PRICING_FILE", "")
    if not path:
        return DEFAULT_MODEL_COSTS
    try:
        # Security: resolve to absolute path and validate it exists
        resolved = os.path.abspath(path)
        with open(resolved) as f:
            data = json.load(f)

        # Schema validation: must be {str: {"in": number, "out": number}}
        if not isinstance(data, dict):
            raise ValueError("Pricing file must be a JSON object")
        for model_name, prices in data.items():
            if not isinstance(model_name, str):
                raise ValueError(f"Model key must be a string, got {type(model_name)}")
            if not isinstance(prices, dict):
                raise ValueError(f"Prices for {model_name!r} must be an object")
            if "in" not in prices or "out" not in prices:
                raise ValueError(f"Prices for {model_name!r} must have 'in' and 'out' keys")
            if not isinstance(prices["in"], (int, float)) or not isinstance(prices["out"], (int, float)):
                raise ValueError(f"Prices for {model_name!r} must be numeric")
            if prices["in"] < 0 or prices["out"] < 0:
                raise ValueError(f"Prices for {model_name!r} must be non-negative")

        logger.info("Loaded custom pricing from %s (%d models)", resolved, len(data))
        return data
    except Exception as exc:
        logger.warning(
            "Could not load AGENTGUARD_PRICING_FILE=%r (%s) — using defaults",
            path, exc,
        )
        return DEFAULT_MODEL_COSTS


MODEL_COSTS = _load_model_costs()


# ---------------------------------------------------------------------------
# CostGuardian
# ---------------------------------------------------------------------------

class CostGuardian:
    """
    Tracks token spend per run and projects whether remaining steps will
    exceed the budget before they execute.

    One instance per Guard.watch() call — never shared across runs.
    All mutable state is protected by threading.Lock for thread safety.
    """

    def __init__(
        self,
        budget_tokens: int = 100_000,
        budget_usd: Optional[float] = None,
        model: str = "claude-sonnet-4-6",
        warn_pct: float = 0.75,   # fire budget_pressure signal at 75% projected
        hard_pct: float = 1.00,   # hard stop at 100% budget
        assumed_remaining_steps: int = 10,
    ):
        self.model = model
        self.budget_usd = budget_usd
        if budget_usd is not None and budget_usd > 0:
            prices = MODEL_COSTS.get(model, MODEL_COSTS.get("claude-sonnet-4-6", {"in": 3.0, "out": 15.0}))
            avg_price_per_m = (prices["in"] + prices["out"]) / 2.0
            if avg_price_per_m > 0:
                budget_tokens = int((budget_usd / avg_price_per_m) * 1_000_000)

        self.budget_tokens = budget_tokens
        self._warn_pct = warn_pct
        self._hard_pct = hard_pct
        self._assumed_remaining = assumed_remaining_steps

        self._lock = threading.Lock()
        self._tokens_used: int = 0
        self._usd_used: float = 0.0
        self._history: list[tuple[int, int, int]] = []  # (step_num, in, out)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: StepEvent) -> None:
        """Called from interceptor after each span ends. Thread-safe."""
        with self._lock:
            self._tokens_used += event.tokens_in + event.tokens_out
            self._history.append((event.step_number, event.tokens_in, event.tokens_out))

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def is_over_budget(self) -> bool:
        return self._tokens_used >= int(self.budget_tokens * self._hard_pct)

    def remaining(self) -> int:
        return max(0, self.budget_tokens - self._tokens_used)

    def projected_total(self) -> int:
        """
        Rolling average of last 5 steps × assumed remaining steps + current spend.
        Returns current spend if fewer than 3 steps recorded (can't extrapolate).
        """
        if len(self._history) < 3:
            return self._tokens_used

        recent = self._history[-5:]
        total_tokens = sum(i + o for _, i, o in recent)
        avg_per_step = total_tokens / len(recent)
        return int(self._tokens_used + avg_per_step * self._assumed_remaining)

    def should_warn(self) -> bool:
        return self.projected_total() > self.budget_tokens * self._warn_pct

    def should_warn_signal(self, run_id: str, step_number: int) -> Optional[FailureSignal]:
        """Convenience: return a budget_pressure FailureSignal if warranted, else None."""
        if not self.should_warn():
            return None
        projected = self.projected_total()
        pct = projected / self.budget_tokens * 100
        return FailureSignal(
            run_id=run_id,
            step_number=step_number,
            signal_type="budget_pressure",
            confidence=min(0.95, pct / 100),
            detail=(
                f"Projected total {projected:,} tokens ({pct:.0f}% of budget "
                f"{self.budget_tokens:,}). Current spend: {self._tokens_used:,}."
            ),
            recommended_action="warn" if pct < 90 else "redirect",
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def cost_usd(self, model: Optional[str] = None) -> float:
        prices = MODEL_COSTS.get(model or self.model, MODEL_COSTS["claude-sonnet-4-6"])
        total_in  = sum(i for _, i, _ in self._history) / 1_000_000
        total_out = sum(o for _, _, o in self._history) / 1_000_000
        return (total_in * prices["in"]) + (total_out * prices["out"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "tokens_used":      self._tokens_used,
            "tokens_budget":    self.budget_tokens,
            "tokens_remaining": self.remaining(),
            "projected_total":  self.projected_total(),
            "over_budget":      self.is_over_budget(),
            "should_warn":      self.should_warn(),
            "estimated_cost_usd": self.cost_usd(),
            "model":            self.model,
        }
