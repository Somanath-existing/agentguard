"""
agentguard/__init__.py
----------------------
Public API surface:  from agentguard import Guard, AgentGuardAbort

Security fixes vs original guide:
  1. Guard.watch() uses add_span_processor() on the EXISTING TracerProvider
     rather than replacing the global one.  Replacing it silently destroys the
     host application's OTel pipeline and causes two concurrent Guard.watch()
     calls to overwrite each other.
  2. Auto-instrumentation is idempotent — calling instrument() on an already-
     instrumented framework raises; we catch and skip.
  3. Config is validated at construction time (budget > 0, on_failure is a
     known value, webhook starts with https:// if provided).
  4. AgentGuardAbort exception is raised on abort signals when enforce_abort=True,
     converting detection into actual enforcement.
  5. webhook_secret defaults to the AGENTGUARD_WEBHOOK_SECRET env var and is
     never exposed in __repr__.
  6. Span processor is properly cleaned up in the finally block to prevent
     memory leaks across repeated watch() calls.
  7. awatch() async context manager for LangGraph, CrewAI async, and other
     async-first agent frameworks.  Same lifecycle as watch() but yields
     from an async generator.
  8. TaintTracker is automatically created and wired into the interceptor
     pipeline — no manual instrumentation required.  Every span's input/output
     is fingerprinted into a provenance DAG.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider  # only for static analysis

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[assignment]
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    class TracerProvider:  # type: ignore[no-redef]
        """Dummy fallback — opentelemetry not installed."""
        pass

from .baseline import BaselineStore
from .cost import CostGuardian
from .interceptor import AgentGuardProcessor
from .memory import RecoveryMemory
from .models import AgentGuardAbort
from .recovery import RecoveryRouter
from .self_test import GuardrailSelfTest
from .semantic_loop import LoopCheckResult, SemanticLoopDetector, extract_active_prompt
from .session import GuardSession
from .state import CheckpointStore, sanitize_run_id
from .taint import TaintTracker
from .toolguard import ToolGuard
from .topology import InterAgentScanResult, TopologyTracker  # noqa: F401 — re-exported for users

logger = logging.getLogger(__name__)

__version__ = "0.3.0"
__all__ = [
    "Guard",
    "AgentGuardAbort",
    "TopologyTracker",
    "InterAgentScanResult",
    "TaintTracker",
    "GuardrailSelfTest",
    "SemanticLoopDetector",
    "LoopCheckResult",
    "extract_active_prompt",
]


class Guard:
    """
    Entry point for AgentGuard.

    Typical usage (sync):

        guard = Guard(
            budget_tokens=50_000,
            on_failure="replan",
            alert_webhook="https://hooks.example.com/agentguard",
            webhook_secret="super-secret",
        )

        with guard.watch(run_id="task-123") as session:
            result = my_agent.run(task)

        print(session.report())

    Typical usage (async — LangGraph, CrewAI, etc.):

        async with guard.awatch(run_id="task-123") as session:
            result = await graph.ainvoke({"input": "Analyze Q3 earnings"})

        print(session.report())
    """

    _VALID_ON_FAILURE = ("retry", "replan", "escalate", "abort")

    def __init__(
        self,
        budget_tokens: int = 100_000,
        on_failure: str = "replan",
        alert_webhook: str | None = None,
        webhook_secret: str = "",
        max_retries: int = 3,
        allow_model_downgrade: bool = False,
        cheap_model: str = "claude-haiku-4-5-20251001",
        store_backend: str = "sqlite",
        store_base_dir: str = "./agentguard_runs",
        redis_client=None,
        warn_pct: float = 0.75,
        model: str = "claude-sonnet-4-6",
        enforce_abort: bool = True,
        tool_allowlist: list[str] | None = None,
        tool_denylist: list[str] | None = None,
        tool_arg_inspection: bool = True,
        tool_custom_validators: list[Callable[[str, str], str | None]] | None = None,
        # ── Adaptive Behavioral Baseline ─────────────────────────────
        agent_type: str = "default",
        anomaly_threshold: float = 2.5,
        baseline_dir: str = "./agentguard_baselines",
        enable_baseline: bool = True,
        # ── Cross-Run Recovery Memory ────────────────────────────────────
        memory_dir: str = "./agentguard_memory",
        enable_memory: bool = True,
        # ── Multi-Agent Topology ─────────────────────────────────────────
        topology: TopologyTracker | None = None,
        parent_run_id: str | None = None,
        # ── Taint Tracking ──────────────────────────────────────────────
        enable_taint: bool = True,
        taint_chunk_size: int = 64,
    ):
        # --- Validate config at construction time ---
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be > 0")
        if on_failure not in self._VALID_ON_FAILURE:
            raise ValueError(
                f"on_failure must be one of {self._VALID_ON_FAILURE}, got {on_failure!r}"
            )
        if alert_webhook and not alert_webhook.startswith("https://"):
            raise ValueError(
                "alert_webhook must start with https:// to prevent plaintext transmission"
            )
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.budget_tokens = budget_tokens
        self.on_failure = on_failure
        self._enforce_abort = enforce_abort
        self._agent_type = agent_type
        self._topology = topology
        self._parent_run_id = parent_run_id
        self._enable_taint = enable_taint
        self._taint_chunk_size = taint_chunk_size

        # One BaselineStore per agent_type — persists across watch() calls
        self._baseline: BaselineStore | None = (
            BaselineStore(
                agent_type=agent_type,
                base_dir=baseline_dir,
                anomaly_threshold_sigma=anomaly_threshold,
            )
            if enable_baseline
            else None
        )

        # One RecoveryMemory per Guard — persists across watch() calls
        self._memory: RecoveryMemory | None = (
            RecoveryMemory(base_dir=memory_dir)
            if enable_memory
            else None
        )

        # Security: prefer env var for webhook secret, never expose in repr
        resolved_secret = webhook_secret or os.environ.get("AGENTGUARD_WEBHOOK_SECRET", "")

        self._config: dict[str, Any] = {
            "alert_webhook":       alert_webhook or "",
            "webhook_secret":      resolved_secret,
            "max_retries":         max_retries,
            "allow_model_downgrade": allow_model_downgrade,
            "cheap_model":         cheap_model,
            "on_failure":          on_failure,
        }
        self._store_backend = store_backend
        self._store_base_dir = store_base_dir
        self._redis_client = redis_client
        self._warn_pct = warn_pct
        self._model = model

        # Tool authorization configuration
        self._tool_guard = ToolGuard(
            allowlist=tool_allowlist,
            denylist=tool_denylist,
            arg_inspection=tool_arg_inspection,
            custom_validators=tool_custom_validators,
        )

        logger.info(
            "Guard initialized: budget_tokens=%d on_failure=%s enforce_abort=%s "
            "agent_type=%s baseline=%s memory=%s topology=%s taint=%s",
            budget_tokens, on_failure, enforce_abort,
            agent_type,
            "enabled" if enable_baseline else "disabled",
            "enabled" if enable_memory else "disabled",
            "linked" if topology else "standalone",
            "enabled" if enable_taint else "disabled",
        )

    def __repr__(self) -> str:
        """Safe repr that never leaks secrets."""
        return (
            f"Guard(budget_tokens={self.budget_tokens}, on_failure={self.on_failure!r}, "
            f"enforce_abort={self._enforce_abort}, model={self._model!r}, "
            f"agent_type={self._agent_type!r})"
        )

    # ------------------------------------------------------------------
    # Shared setup — used by both watch() and awatch()
    # ------------------------------------------------------------------

    def _build_session(self, run_id: str) -> tuple:
        """
        Build all per-run objects: store, cost, recovery, session, signal
        engine, taint tracker, and processor.

        Returns (session, processor, safe_run_id, taint_tracker) if OTel is
        available, or (session, None, safe_run_id, None) if not.
        """
        safe_run_id = sanitize_run_id(run_id)

        # Register with shared topology before doing anything else
        if self._topology:
            self._topology.register(
                run_id=safe_run_id,
                agent_type=self._agent_type,
                parent_run_id=self._parent_run_id,
            )
            if self._parent_run_id:
                loop_msg = self._topology.check_orchestrator_loop(self._parent_run_id)
                if loop_msg:
                    logger.warning("TopologyTracker orchestrator loop: %s", loop_msg)

        store = CheckpointStore(
            run_id=safe_run_id,
            backend=self._store_backend,
            base_dir=self._store_base_dir,
            redis_client=self._redis_client,
        )
        cost = CostGuardian(
            budget_tokens=self.budget_tokens,
            model=self._model,
            warn_pct=self._warn_pct,
        )
        recovery = RecoveryRouter(
            store=store,
            config=self._config,
            memory=self._memory,
            agent_type=self._agent_type,
        )

        # Create per-session taint tracker
        taint_tracker: TaintTracker | None = None
        if self._enable_taint:
            taint_tracker = TaintTracker(chunk_size=self._taint_chunk_size)

        session = GuardSession(
            run_id=safe_run_id,
            guard=self,
            store=store,
            cost=cost,
            recovery=recovery,
            enforce_abort=self._enforce_abort,
            taint_tracker=taint_tracker,
        )

        if not OTEL_AVAILABLE:
            return session, None, safe_run_id, taint_tracker

        from .signals import SignalEngine
        signal_engine = SignalEngine(
            run_id=safe_run_id,
            baseline=self._baseline,
        )

        processor = AgentGuardProcessor(
            session=session,
            signal_engine=signal_engine,
            cost_guardian=cost,
            store=store,
            tool_guard=self._tool_guard,
            taint_tracker=taint_tracker,
        )

        return session, processor, safe_run_id, taint_tracker

    def _cleanup_session(self, safe_run_id: str, processor) -> None:
        """Shared cleanup for both watch() and awatch()."""
        if self._topology:
            self._topology.mark_completed(safe_run_id)
        if processor is not None:
            processor.shutdown()
            self._detach_processor(processor)
            logger.debug("AgentGuardProcessor shut down and detached for run_id=%s", safe_run_id)

    # ------------------------------------------------------------------
    # Sync context manager — the primary API
    # ------------------------------------------------------------------

    @contextmanager
    def watch(self, run_id: str):
        """
        Wraps an agent run (synchronous).  Yields a GuardSession.

        with guard.watch(run_id="task-123") as session:
            result = my_agent.run(task)

        print(session.report())
        """
        session, processor, safe_run_id, _ = self._build_session(run_id)

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry is not available. AgentGuard monitoring and guardrails are INACTIVE."
            )
            try:
                yield session
            finally:
                self._cleanup_session(safe_run_id, None)
            return

        self._attach_processor(processor)
        _auto_instrument()

        try:
            yield session
        finally:
            self._cleanup_session(safe_run_id, processor)

    # ------------------------------------------------------------------
    # Async context manager — for LangGraph, CrewAI async, etc.
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def awatch(self, run_id: str):
        """
        Wraps an agent run (async).  Yields a GuardSession.

        Identical lifecycle to watch(), but compatible with async/await syntax
        used by LangGraph, CrewAI, AutoGen, and other async-first frameworks.

        The OTel SpanProcessor (on_end) is always synchronous by design — this
        is correct because OTel itself calls on_end synchronously regardless of
        whether the host app is async.  The critical sections inside on_end are
        lightweight (hash computations, deque appends, SQLite WAL writes)
        and will not meaningfully block the event loop.

        Usage:
            async with guard.awatch(run_id="task-456") as session:
                result = await graph.ainvoke({"input": "..."})

            print(session.report())
        """
        session, processor, safe_run_id, _ = self._build_session(run_id)

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry is not available. AgentGuard monitoring and guardrails are INACTIVE."
            )
            try:
                yield session
            finally:
                self._cleanup_session(safe_run_id, None)
            return

        self._attach_processor(processor)
        _auto_instrument()

        try:
            yield session
        finally:
            self._cleanup_session(safe_run_id, processor)

    # ------------------------------------------------------------------
    # OTel integration (secure)
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_processor(processor: AgentGuardProcessor) -> None:
        """
        Attach the span processor to the existing TracerProvider.

        If no SDK provider is set yet, we create one.  We never call
        trace.set_tracer_provider() on an existing SDK provider — that would
        silently destroy the host app's spans.
        """
        if not OTEL_AVAILABLE:
            return

        existing = trace.get_tracer_provider()

        if isinstance(existing, TracerProvider):
            # Host app already set up OTel — just add our processor
            existing.add_span_processor(processor)
            logger.debug("Attached AgentGuardProcessor to existing TracerProvider")
        else:
            # No SDK provider yet — safe to create and register one
            provider = TracerProvider()
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            logger.debug("Created new TracerProvider for AgentGuard")

    @staticmethod
    def _detach_processor(processor: AgentGuardProcessor) -> None:
        """
        Remove the span processor from the TracerProvider's active processor list
        to prevent memory leaks and performance overhead on subsequent runs.
        """
        if not OTEL_AVAILABLE:
            return

        try:
            existing = trace.get_tracer_provider()
            if hasattr(existing, "_active_span_processor"):
                asp = existing._active_span_processor
                if hasattr(asp, "_lock") and hasattr(asp, "_span_processors"):
                    with asp._lock:
                        asp._span_processors = tuple(
                            p for p in asp._span_processors if p is not processor
                        )
                    logger.debug("Detached AgentGuardProcessor from TracerProvider")
        except Exception as e:
            logger.warning(
                "Could not detach AgentGuardProcessor from TracerProvider due to OTel SDK internal structure changes: %s",
                e
            )


def _auto_instrument() -> None:
    """
    Attempt to auto-instrument whichever LLM frameworks are installed.
    Each is wrapped in try/except — a missing package or already-instrumented
    framework must not crash the guard.
    """
    if not OTEL_AVAILABLE:
        return
    for name, module_path, cls_name in [
        ("LangChain", "opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
        ("OpenAI",    "opentelemetry.instrumentation.openai",    "OpenAIInstrumentor"),
        ("Anthropic", "opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            instrumentor = getattr(mod, cls_name)()
            instrumentor.instrument()
            logger.debug("Auto-instrumented %s", name)
        except ImportError:
            pass  # framework not installed — skip silently
        except Exception as exc:
            # Already instrumented raises RuntimeError; log and continue
            logger.debug("Could not instrument %s: %s", name, exc)
