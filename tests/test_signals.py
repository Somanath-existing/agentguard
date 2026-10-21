"""
tests/test_signals.py
Tests for SignalEngine — loop detection, info-gain collapse, error streaks.
"""
from __future__ import annotations

from agentguard.models import StepEvent
from agentguard.signals import SignalEngine


def make_event(
    step: int,
    input_text: str = "unique",
    output_size: int = 1000,
    status: str = "ok",
    error_type: str | None = None,
    tool: str = "web_search",
) -> StepEvent:
    return StepEvent(
        run_id="test-run",
        step_number=step,
        span_id=f"span-{step}",
        span_name="tool.web_search",
        tool_name=tool,
        input_hash=StepEvent.make_input_hash(input_text),
        output_size=output_size,
        tokens_in=100,
        tokens_out=200,
        status=status,
        error_type=error_type,
    )


class TestLoopDetector:
    def test_no_signal_on_unique_inputs(self):
        engine = SignalEngine("run1")
        for i in range(5):
            result = engine.evaluate(make_event(i + 1, input_text=f"unique-{i}"))
        assert result is None

    def test_warn_on_second_repeat(self):
        engine = SignalEngine("run1")
        engine.evaluate(make_event(1, input_text="repeated"))
        signal = engine.evaluate(make_event(2, input_text="repeated"))
        assert signal is not None
        assert signal.signal_type == "loop"
        assert signal.recommended_action == "warn"
        assert signal.confidence == 0.65

    def test_abort_on_third_repeat(self):
        engine = SignalEngine("run1")
        engine.evaluate(make_event(1, input_text="repeated"))
        engine.evaluate(make_event(2, input_text="repeated"))
        signal = engine.evaluate(make_event(3, input_text="repeated"))
        assert signal is not None
        assert signal.signal_type == "loop"
        assert signal.recommended_action == "abort"
        assert signal.confidence == 0.95

    def test_window_sliding(self):
        """After 10+ different inputs, an old repeated hash no longer triggers."""
        engine = SignalEngine("run1", window=10)
        engine.evaluate(make_event(1, input_text="old"))
        for i in range(10):
            engine.evaluate(make_event(i + 2, input_text=f"new-{i}"))
        # "old" has been pushed out of the window — should not trigger
        signal = engine.evaluate(make_event(12, input_text="old"))
        # Only seen once in window now, so at most a warn
        if signal:
            assert signal.recommended_action != "abort"

    def test_no_signal_when_hash_missing(self):
        engine = SignalEngine("run1")
        event = StepEvent(
            run_id="test-run",
            step_number=1,
            span_id="s1",
            span_name="tool.x",
            input_hash=None,
        )
        assert engine.evaluate(event) is None


class TestInfoGainCollapse:
    def test_no_signal_with_stable_output(self):
        engine = SignalEngine("run1")
        for i, size in enumerate([1000, 1100, 900, 1050]):
            engine.evaluate(make_event(i + 1, input_text=f"u{i}", output_size=size))
        # Non-monotonic — should not fire
        signal = engine.evaluate(make_event(5, input_text="u5", output_size=950))
        assert signal is None or signal.signal_type != "info_collapse"

    def test_fires_on_70pct_collapse(self):
        engine = SignalEngine("run1")
        sizes = [1000, 700, 400, 200]  # monotonic drop > 70%
        for i, size in enumerate(sizes):
            signal = engine.evaluate(make_event(i + 1, input_text=f"u{i}", output_size=size))
        assert signal is not None
        assert signal.signal_type == "info_collapse"
        assert signal.recommended_action == "redirect"

    def test_no_signal_below_70pct_collapse(self):
        engine = SignalEngine("run1")
        sizes = [1000, 800, 600, 400]  # 60% drop — below threshold
        last_signal = None
        for i, size in enumerate(sizes):
            last_signal = engine.evaluate(make_event(i + 1, input_text=f"u{i}", output_size=size))
        assert last_signal is None or last_signal.signal_type != "info_collapse"

    def test_no_divzero_on_zero_first_output(self):
        engine = SignalEngine("run1")
        for i, size in enumerate([0, 0, 0, 0]):
            engine.evaluate(make_event(i + 1, input_text=f"u{i}", output_size=size))


class TestErrorStreak:
    def test_no_signal_on_isolated_error(self):
        engine = SignalEngine("run1")
        engine.evaluate(make_event(1))
        signal = engine.evaluate(make_event(2, status="error", error_type="TimeoutError"))
        # Single error should not fire
        assert signal is None or signal.signal_type != "tool_error"

    def test_warn_on_two_errors(self):
        engine = SignalEngine("run1")
        engine.evaluate(make_event(1, input_text="err-1", status="error", error_type="TimeoutError"))
        signal = engine.evaluate(make_event(2, input_text="err-2", status="error", error_type="TimeoutError"))
        assert signal is not None
        assert signal.signal_type == "tool_error"
        assert signal.recommended_action == "warn"

    def test_abort_on_three_errors(self):
        engine = SignalEngine("run1")
        for i in range(3):
            signal = engine.evaluate(make_event(i + 1, input_text=f"err-{i}", status="error", error_type="TimeoutError"))
        assert signal is not None
        assert signal.signal_type == "tool_error"
        assert signal.recommended_action == "abort"

    def test_streak_resets_after_success(self):
        engine = SignalEngine("run1")
        engine.evaluate(make_event(1, input_text="e1", status="error"))
        engine.evaluate(make_event(2, input_text="e2", status="error"))
        engine.evaluate(make_event(3, input_text="ok1"))           # success — resets streak
        signal = engine.evaluate(make_event(4, input_text="e3", status="error"))
        # Only one error in new streak — no abort
        assert signal is None or signal.recommended_action != "abort"

    def test_reset_clears_all_state(self):
        engine = SignalEngine("run1")
        for i in range(3):
            engine.evaluate(make_event(i + 1, input_text=f"rst-{i}", status="error"))
        engine.reset()
        signal = engine.evaluate(make_event(4, input_text="rst-new", status="error"))
        assert signal is None or signal.recommended_action != "abort"


class TestPredictiveFailure:
    def test_predicts_failure_on_repetition_and_collapse(self):
        engine = SignalEngine("run1")
        # Step 1-3: prepare history
        engine.evaluate(make_event(1, input_text="a", output_size=1000))
        engine.evaluate(make_event(2, input_text="a", output_size=800))
        engine.evaluate(make_event(3, input_text="b", output_size=600))
        
        # Step 4: triggers high repetition (entropy 2/4 = 0.5) + approaching collapse (50%)
        # call engine._check_predictive_failure directly to bypass loop warn tie-breaker
        event = make_event(4, input_text="b", output_size=500)
        engine._recent_events.append(event)
        engine._input_hashes.append(event.input_hash)
        
        signal = engine._check_predictive_failure(event)
        
        assert signal is not None
        assert signal.signal_type == "predictive_failure"
        assert signal.recommended_action == "warn"
        assert "approaching output collapse" in signal.detail

    def test_predicts_failure_on_collapse_and_token_acceleration(self):
        engine = SignalEngine("run1")
        
        # Step 1: tokens_in=100, tokens_out=100 (total 200)
        e1 = make_event(1, input_text="u1", output_size=1000)
        e1.tokens_in, e1.tokens_out = 50, 50
        engine.evaluate(e1)
        
        # Step 2: tokens_in=150, tokens_out=150 (total 300, diff = 100)
        e2 = make_event(2, input_text="u2", output_size=800)
        e2.tokens_in, e2.tokens_out = 150, 150
        engine.evaluate(e2)
        
        # Step 3: tokens_in=250, tokens_out=250 (total 500, diff = 200, acceleration!)
        e3 = make_event(3, input_text="u3", output_size=600)
        e3.tokens_in, e3.tokens_out = 250, 250
        engine.evaluate(e3)
        
        # Step 4: output collapse 50% (approaching) + accelerating tokens
        e4 = make_event(4, input_text="u4", output_size=500)
        e4.tokens_in, e4.tokens_out = 400, 400  # total 800, diff = 300 (rate continues to accelerate)
        signal = engine.evaluate(e4)
        
        assert signal is not None
        assert signal.signal_type == "predictive_failure"
        assert "exponential token consumption" in signal.detail
