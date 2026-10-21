"""
agentguard/interceptor.py
-------------------------
Custom OpenTelemetry SpanProcessor — the hook layer.

Security fixes vs original guide:
  1. Raw prompt / completion text is NEVER stored. Only the SHA-256 hash of
     the input is computed (for loop detection) and the character count of the
     output (for info-gain collapse).  Storing raw LLM content creates a data
     exfiltration surface and a PII/secret storage risk.
  2. error_type is sanitized via Untrusted.sanitize() before being stored in
     StepEvent, preventing injected error strings from propagating into prompts
     or log-driven automation.
  3. The processor does NOT replace the global TracerProvider.  The Guard class
     (in __init__.py) attaches to the existing provider via add_span_processor()
     so AgentGuard doesn't silently destroy the host app's OTel pipeline.
  4. on_end() is wrapped in a try/except — a bug in the interceptor must never
     crash the agent it's monitoring.
  5. Tool authorization: if a ToolGuard is configured, every tool call is
     checked against the allowlist/denylist before processing.
  6. Taint tracking: if a TaintTracker is configured, every span's input/output
     is automatically fingerprinted and added to the provenance DAG.  When
     injection is detected at step N, any future step that consumes step N's
     output triggers a taint_propagation abort signal with clean rollback prefix.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from .cost import CostGuardian
from .injection import scan_for_injection
from .models import AgentGuardAbort, AuditEntry, FailureSignal, StepEvent, Untrusted
from .semantic_loop import extract_active_prompt
from .signals import SignalEngine
from .state import CheckpointStore
from .toolguard import ToolGuard

if TYPE_CHECKING:
    from .taint import TaintTracker

logger = logging.getLogger(__name__)

# Span name keywords we care about — anything else is infrastructure noise
_RELEVANT_KEYWORDS = ("tool", "llm", "chat", "completion", "retrieve", "agent")


class AgentGuardProcessor(SpanProcessor):
    """
    Registered with the OTel TracerProvider.
    on_end() fires after every span in the agent process.
    We filter to tool/LLM spans and build a StepEvent from each.
    """

    def __init__(
        self,
        session,             # GuardSession — circular import avoided by type hint
        signal_engine: SignalEngine,
        cost_guardian: CostGuardian,
        store: CheckpointStore,
        tool_guard: ToolGuard | None = None,
        taint_tracker: Optional["TaintTracker"] = None,
    ):
        self._session = session
        self._signals = signal_engine
        self._cost = cost_guardian
        self._store = store
        self._tool_guard = tool_guard
        self._taint = taint_tracker
        self._lock = threading.Lock()
        self._step_counts: dict[str, int] = {}
        # Steps where injection was detected — used to propagate taint labels
        self._tainted_sources: set[int] = set()

    # ------------------------------------------------------------------
    # SpanProcessor interface
    # ------------------------------------------------------------------

    def on_start(self, span, parent_context=None) -> None:
        """Stamp every span with the run_id so we can correlate in on_end."""
        run_id = self._session.run_id
        if run_id:
            span.set_attribute("agentguard.run_id", run_id)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Main interception point. Wrapped in try/except — a crash here must
        never propagate up and kill the monitored agent.
        """
        try:
            self._process_span(span)
        except AgentGuardAbort:
            raise
        except Exception:
            logger.exception(
                "AgentGuardProcessor.on_end() raised unexpectedly for span %r",
                span.name,
            )

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal span processing
    # ------------------------------------------------------------------

    def _process_span(self, span: ReadableSpan) -> None:
        # Filter: only process spans we care about
        name = span.name.lower()
        if not any(k in name for k in _RELEVANT_KEYWORDS):
            return

        run_id = str(span.attributes.get("agentguard.run_id", "unknown"))

        # Maintain per-run step counter (thread-safe)
        with self._lock:
            step = self._step_counts.get(run_id, 0) + 1
            self._step_counts[run_id] = step

        # --- Security: hash input, NEVER store raw text ---
        raw_input = (
            span.attributes.get("gen_ai.input.messages")
            or span.attributes.get("gen_ai.prompt")
            or span.attributes.get("gen_ai.prompt.0.content")
            or span.attributes.get("llm.prompts.0.content")
            or ""
        )
        if not raw_input:
            for k, v in span.attributes.items():
                kl = k.lower()
                if any(p in kl for p in ("prompt", "input")):
                    if not any(p in kl for p in ("usage", "tokens", "model", "id", "provider", "tier")):
                        if isinstance(v, str) and v:
                            raw_input = v
                            break

        # Extract active prompt to eliminate cumulative multi-turn history dilution
        active_prompt = extract_active_prompt(str(raw_input)) if raw_input else ""
        input_hash = StepEvent.make_input_hash(active_prompt) if active_prompt else None

        # Output: scan for injection BEFORE discarding raw text, then measure size only.
        # raw_output is eligible for GC immediately after this block — not stored.
        raw_output = (
            span.attributes.get("gen_ai.output.messages")
            or span.attributes.get("gen_ai.completion")
            or span.attributes.get("gen_ai.completion.0.content")
            or span.attributes.get("llm.completions.0.content")
            or ""
        )
        if not raw_output:
            for k, v in span.attributes.items():
                kl = k.lower()
                if any(p in kl for p in ("completion", "output", "choices")):
                    if not any(p in kl for p in ("prompt", "input", "usage", "tokens", "model", "id", "provider", "tier", "fingerprint")):
                        if isinstance(v, str) and v:
                            raw_output = v
                            break

        # ────────────────────────────────────────────────────────────────
        # BUILD & PERSIST StepEvent BEFORE any safety check that can abort
        # ────────────────────────────────────────────────────────────────
        output_size = len(str(raw_output))

        # --- Security: sanitize error strings ---
        from opentelemetry.trace import StatusCode
        status_code = getattr(span.status, "status_code", None)
        is_error = (status_code == StatusCode.ERROR) if status_code is not None else getattr(span.status, "is_error", False)

        raw_error = getattr(span.status, "description", None) if is_error else None
        safe_error = Untrusted(raw_error).sanitize(max_len=200) if raw_error else None

        tokens_in = int(
            span.attributes.get("gen_ai.usage.input_tokens")
            or span.attributes.get("gen_ai.usage.prompt_tokens")
            or span.attributes.get("llm.usage.prompt_tokens")
            or 0
        )
        tokens_out = int(
            span.attributes.get("gen_ai.usage.output_tokens")
            or span.attributes.get("gen_ai.usage.completion_tokens")
            or span.attributes.get("llm.usage.completion_tokens")
            or 0
        )

        event = StepEvent(
            run_id=run_id,
            step_number=step,
            span_id=str(span.context.span_id),
            span_name=span.name,
            tool_name=str(span.attributes.get("gen_ai.tool.name", "")) or None,
            input_hash=input_hash,
            output_size=output_size,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(span.end_time - span.start_time) / 1e6,
            status="error" if is_error else "ok",
            error_type=safe_error,
        )

        # 1. Checkpoint step BEFORE any safety check that might abort
        self._store.save_step(event)

        # 2. Record token budget BEFORE abort checks
        self._cost.record(event)

        # Record tokens in shared topology tracker if present
        topo = getattr(self._session._guard, "_topology", None)
        if topo:
            topo.record_tokens(run_id, event.tokens_in + event.tokens_out)

        # ────────────────────────────────────────────────────────────────
        # AUTO TAINT TRACKING — build provenance DAG from every span
        # ────────────────────────────────────────────────────────────────
        taint_labels = []
        if self._taint is not None:
            if raw_output:
                tool_name_for_taint = (
                    str(span.attributes.get("gen_ai.tool.name", "")) or span.name
                )
                self._taint.on_step_output(step, str(raw_output), tool_name_for_taint)

            if raw_input:
                taint_labels = self._taint.on_step_input(step, str(raw_input))

        # Scan both input and output for indirect injections
        injection_result = None
        if raw_input:
            injection_result = scan_for_injection(str(raw_input))
        if not injection_result and raw_output:
            injection_result = scan_for_injection(str(raw_output))

        # Taint source marking
        if injection_result and self._taint is not None:
            with self._lock:
                self._tainted_sources.add(step)
            logger.debug(
                "TaintTracker: step %d marked as tainted source (injection: %s)",
                step, injection_result.label,
            )

        # Taint propagation check — fire abort if tainted data consumed
        # Exclude the CURRENT step from self-taint: a step that is itself
        # the injection source should not also trigger propagation to itself.
        if taint_labels and self._taint is not None:
            with self._lock:
                tainted_srcs = set(self._tainted_sources) - {step}
            if tainted_srcs:
                for label in taint_labels:
                    chain_steps = set(label.propagation_chain)
                    chain_steps.add(label.source_step)
                    overlap = chain_steps & tainted_srcs
                    if overlap:
                        clean_prefix = self._taint.clean_prefix(label.source_step)
                        taint_origin = min(overlap)
                        self._session.on_signal(
                            FailureSignal(
                                run_id=run_id,
                                step_number=step,
                                signal_type="taint_propagation",
                                confidence=label.confidence,
                                detail=(
                                    f"Tainted output from step {taint_origin} "
                                    f"(tool: {label.source_tool}) propagated to step {step} "
                                    f"via chain {list(label.propagation_chain)}. "
                                    f"Safe rollback boundary: steps {clean_prefix}"
                                ),
                                recommended_action="abort",
                            )
                        )
                        break  # one taint signal per step is sufficient

        # 3. Tool authorization check
        if self._tool_guard and event.tool_name:
            tool_args = str(span.attributes.get("gen_ai.tool.args", ""))
            auth = self._tool_guard.authorize(event.tool_name, tool_args)
            if not auth.allowed:
                self._store.save_audit_event(
                    AuditEntry(
                        run_id=run_id,
                        entry_type="tool_blocked",
                        action="deny",
                        triggered_by="tool_guard",
                        step_number=step,
                        detail=f"{auth.reason} (rule: {auth.matched_rule})",
                    )
                )
                self._session.on_signal(
                    FailureSignal(
                        run_id=run_id,
                        step_number=step,
                        signal_type="unauthorized_tool",
                        confidence=1.0,
                        detail=auth.reason,
                        recommended_action="abort",
                    )
                )

        # 4. Indirect prompt injection check
        if injection_result:
            inj_signal = FailureSignal(
                run_id=run_id,
                step_number=step,
                signal_type="drift",
                confidence=injection_result.confidence,
                detail=(
                    f"Indirect injection detected [{injection_result.label}]: "
                    f"...{injection_result.snippet}..."
                ),
                recommended_action="abort",
            )
            self._session.on_signal(inj_signal)

        # 5. Run signal engine (loop / info-collapse / error-streak + baseline + semantic loop)
        signal = self._signals.evaluate(
            event,
            raw_input=active_prompt if active_prompt else (str(raw_input) if raw_input else None),
        )
        if signal and signal.recommended_action in ("warn", "redirect", "abort"):
            self._session.on_signal(signal)

        # Budget pressure signal (local or shared topology)
        budget_signal = self._cost.should_warn_signal(run_id, step)
        if not budget_signal and topo:
            used = topo.total_tokens_used()
            total = topo._total_budget
            if total > 0 and (used / total) >= self._cost._warn_pct:
                budget_signal = FailureSignal(
                    run_id=run_id,
                    step_number=step,
                    signal_type="budget_pressure",
                    confidence=1.0,
                    detail=f"Shared topology budget usage high: {used}/{total} tokens used",
                    recommended_action="warn",
                )

        if budget_signal:
            self._session.on_signal(budget_signal)

        # Hard budget exceeded (local or shared topology)
        local_over = self._cost.is_over_budget()
        topo_over = topo.is_tree_over_budget() if topo else False

        if local_over or topo_over:
            remaining = topo.remaining_budget() if topo_over else self._cost.remaining()
            self._session.on_budget_exceeded(remaining)
