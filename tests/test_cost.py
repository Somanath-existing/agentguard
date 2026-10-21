"""
tests/test_cost.py
"""
from __future__ import annotations

import pytest

from agentguard.cost import CostGuardian
from agentguard.models import StepEvent


def make_event(step: int, tokens_in: int = 100, tokens_out: int = 200) -> StepEvent:
    return StepEvent(
        run_id="cost-run",
        step_number=step,
        span_id=f"s{step}",
        span_name="tool.test",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


class TestCostGuardian:
    def test_records_tokens(self):
        g = CostGuardian(budget_tokens=10_000)
        g.record(make_event(1, tokens_in=500, tokens_out=300))
        s = g.summary()
        assert s["tokens_used"] == 800

    def test_is_over_budget(self):
        g = CostGuardian(budget_tokens=1000)
        g.record(make_event(1, tokens_in=600, tokens_out=500))
        assert g.is_over_budget()

    def test_remaining_never_negative(self):
        g = CostGuardian(budget_tokens=100)
        g.record(make_event(1, tokens_in=200, tokens_out=200))
        assert g.remaining() == 0

    def test_projected_total_needs_3_steps(self):
        g = CostGuardian(budget_tokens=50_000)
        g.record(make_event(1))
        g.record(make_event(2))
        # Fewer than 3 steps — projection == current spend
        assert g.projected_total() == g._tokens_used

    def test_projected_total_extrapolates(self):
        g = CostGuardian(budget_tokens=100_000, assumed_remaining_steps=10)
        for i in range(5):
            g.record(make_event(i + 1, tokens_in=1000, tokens_out=1000))
        # 5 steps × 2000 tokens = 10000 used. Avg 2000/step × 10 = 20000 projected extra
        projected = g.projected_total()
        assert projected > g._tokens_used

    def test_should_warn_at_threshold(self):
        # warn at 75% default; force a high projected total
        g = CostGuardian(budget_tokens=1000, warn_pct=0.75, assumed_remaining_steps=100)
        # Record steps with high burn rate
        for i in range(5):
            g.record(make_event(i + 1, tokens_in=50, tokens_out=50))
        # projected = 500 + (100 × 100) = way over threshold
        assert g.should_warn()

    def test_cost_usd_calculation(self):
        g = CostGuardian(budget_tokens=10_000, model="claude-sonnet-4-6")
        # 1M input tokens @ $3.00 = $3.00
        g.record(make_event(1, tokens_in=1_000_000, tokens_out=0))
        cost = g.cost_usd()
        assert abs(cost - 3.00) < 0.001

    def test_no_divzero_empty_history(self):
        g = CostGuardian(budget_tokens=10_000)
        assert g.projected_total() == 0
        assert g.cost_usd() == 0.0

    def test_warn_signal_returned_when_warranted(self):
        g = CostGuardian(budget_tokens=1000, warn_pct=0.10, assumed_remaining_steps=100)
        for i in range(5):
            g.record(make_event(i + 1, tokens_in=10, tokens_out=10))
        signal = g.should_warn_signal("run1", 5)
        assert signal is not None
        assert signal.signal_type == "budget_pressure"

    def test_warn_signal_none_when_not_warranted(self):
        g = CostGuardian(budget_tokens=1_000_000, warn_pct=0.75)
        g.record(make_event(1, tokens_in=100, tokens_out=100))
        g.record(make_event(2, tokens_in=100, tokens_out=100))
        g.record(make_event(3, tokens_in=100, tokens_out=100))
        assert g.should_warn_signal("run1", 3) is None
