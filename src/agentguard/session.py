"""
agentguard/session.py
---------------------
GuardSession: the run context that holds together all sub-systems
and produces the terminal report at the end of a run.

Security notes:
  - Every signal and recovery action is written to the append-only audit log
    in CheckpointStore — immutable forensic trail.
  - on_signal() delegates to RecoveryRouter which owns the inject_prompt logic;
    GuardSession never constructs prompts itself.
  - The report() method never prints raw LLM content — only metadata.
  - When enforce_abort=True (the default), on_signal() with action='abort'
    raises AgentGuardAbort to actually kill the agent — not just log a warning.
  - All mutable state (_signals_fired, _recovery_actions, _aborted) is protected
    by threading.Lock for safe concurrent access.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import io
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import AgentGuardAbort, AuditEntry, FailureSignal

if TYPE_CHECKING:
    from .cost import CostGuardian
    from .recovery import RecoveryRouter
    from .state import CheckpointStore
    from .taint import TaintTracker

logger = logging.getLogger(__name__)


class SafeTextIOWrapper(io.TextIOWrapper):
    """A TextIOWrapper that does not close the underlying stream when closed or deleted."""
    def close(self) -> None:
        pass


class GuardSession:
    """
    Lifecycle object for a single agent run.
    Created by Guard.watch() and returned as the context-manager value.

    Usage:
        with guard.watch(run_id="task-123") as session:
            result = my_agent.run(task)
        print(session.report())

    If enforce_abort=True (default), an AgentGuardAbort exception is raised
    when an abort signal fires or the budget is exceeded. Callers can catch
    this for graceful shutdown:

        try:
            with guard.watch(run_id="task") as session:
                agent.run(task)
        except AgentGuardAbort as e:
            print(f"Guard killed the run: {e}")
    """

    def __init__(
        self,
        run_id: str,
        guard,                           # Guard instance (avoids circular import)
        store: "CheckpointStore",
        cost: "CostGuardian",
        recovery: "RecoveryRouter",
        enforce_abort: bool = True,      # NEW: actually kill the agent on abort
        taint_tracker: Optional["TaintTracker"] = None,
    ):
        self.run_id = run_id
        self._guard = guard
        self._store = store
        self._cost = cost
        self._recovery = recovery
        self._enforce_abort = enforce_abort
        self._taint = taint_tracker

        self._lock = threading.Lock()
        self._signals_fired: list[FailureSignal] = []
        self._recovery_actions: list[dict] = []
        self._aborted: bool = False
        self._abort_reason: str = ""
        self._last_signal_type: Optional[str] = None  # for recovery outcome tracking (Idea 3)

        # Force UTF-8 output on Windows to avoid CP1252 emoji encoding errors
        self._console = Console(file=SafeTextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, 'buffer') else sys.stdout, highlight=True)
        self._start_time = datetime.now(tz=timezone.utc)

        # Record session start in audit log
        self._store.save_audit_event(
            AuditEntry(
                run_id=self.run_id,
                entry_type="session_start",
                detail=f"Guard session started. budget_tokens={guard.budget_tokens} enforce_abort={enforce_abort}",
            )
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def aborted(self) -> bool:
        """Whether this session has been aborted by the guard."""
        return self._aborted

    @property
    def abort_reason(self) -> str:
        """Human-readable reason for the abort, if any."""
        return self._abort_reason

    # ------------------------------------------------------------------
    # Callbacks from AgentGuardProcessor
    # ------------------------------------------------------------------

    def on_signal(self, signal: FailureSignal) -> None:
        """
        Called by the interceptor when a high-severity signal fires.

        If enforce_abort=True and the signal recommends 'abort', this method
        raises AgentGuardAbort to actually stop the agent.
        """
        with self._lock:
            self._signals_fired.append(signal)

        # Log to audit trail first — before any recovery action
        self._store.save_audit_event(
            AuditEntry(
                run_id=self.run_id,
                entry_type="signal_fired",
                triggered_by=signal.signal_type,
                step_number=signal.step_number,
                detail=signal.detail,
            )
        )

        # Pretty-print to terminal
        color = "bold red" if signal.recommended_action == "abort" else "yellow"
        icon = "[!!]" if signal.recommended_action == "abort" else "[!]"
        self._console.print(
            f"[{color}]{icon} [{signal.signal_type.upper()}] step {signal.step_number}: "
            f"{signal.detail[:120]}[/]"
        )

        # Route to recovery
        if signal.recommended_action in ("redirect", "abort"):
            action = self._recovery.handle(signal)
            with self._lock:
                self._recovery_actions.append(action)
                self._last_signal_type = signal.signal_type  # Idea 3: track for outcome

            self._console.print(
                f"[dim]   → Recovery: {action['action']} "
                f"(reason: {action.get('reason', '')[:80]})[/]"
            )

            # ENFORCEMENT: actually kill the agent on abort
            if action["action"] == "abort" and self._enforce_abort:
                self._aborted = True
                self._abort_reason = action.get("reason", signal.detail)
                self._store.save_audit_event(
                    AuditEntry(
                        run_id=self.run_id,
                        entry_type="budget_event",
                        action="enforce_abort",
                        triggered_by=signal.signal_type,
                        step_number=signal.step_number,
                        detail=f"Enforcement: raising AgentGuardAbort. Reason: {self._abort_reason[:200]}",
                    )
                )
                raise AgentGuardAbort(
                    reason=self._abort_reason,
                    signal_type=signal.signal_type,
                    step_number=signal.step_number,
                )

    def on_budget_exceeded(self, remaining: int) -> None:
        """
        Called by the interceptor when the hard budget limit is hit.

        If enforce_abort=True, raises AgentGuardAbort to stop the agent.
        """
        self._console.print(
            f"[bold red][$] Budget exceeded -- {remaining:,} tokens remaining in budget.[/]"
        )
        self._store.save_audit_event(
            AuditEntry(
                run_id=self.run_id,
                entry_type="budget_event",
                action="budget_exceeded",
                detail=f"Hard budget hit. {remaining} tokens remaining.",
            )
        )

        if self._enforce_abort:
            self._aborted = True
            self._abort_reason = f"Token budget exceeded. {remaining} tokens remaining."
            raise AgentGuardAbort(
                reason=self._abort_reason,
                signal_type="budget_exceeded",
                step_number=0,
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GuardSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Determine if the exit was due to our own abort
        was_guard_abort = exc_type is AgentGuardAbort
        clean_exit = exc_type is None

        # Idea 3: record recovery outcome before closing store
        self._record_recovery_outcome(succeeded=clean_exit)

        self._store.save_audit_event(
            AuditEntry(
                run_id=self.run_id,
                entry_type="session_end",
                detail=(
                    f"Session ended "
                    f"{'(AgentGuard abort)' if was_guard_abort else 'with exception' if exc_type else 'cleanly'}. "
                    f"signals={len(self._signals_fired)} "
                    f"recovery_actions={len(self._recovery_actions)}"
                ),
            )
        )
        self._store.close()
        self._console.print("[dim]AgentGuard session closed.[/]")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> str:
        """
        Print a rich terminal report and return a plain-text summary string.
        Never includes raw LLM content — metadata only.
        """
        elapsed = (datetime.now(tz=timezone.utc) - self._start_time).total_seconds()

        # ── Main metrics table ──────────────────────────────────────────
        table = Table(
            title=f"AgentGuard Report — {self.run_id}",
            show_lines=True,
            highlight=True,
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Run ID", self.run_id)
        table.add_row("Elapsed", f"{elapsed:.1f}s")
        table.add_row("Signals fired", str(len(self._signals_fired)))
        table.add_row("Recovery actions", str(len(self._recovery_actions)))
        table.add_row("Aborted?", "[red]YES[/red]" if self._aborted else "[green]No[/green]")

        cost_summary = self._cost.summary()
        table.add_row("Tokens used", f"{cost_summary['tokens_used']:,}")
        table.add_row("Token budget", f"{cost_summary['tokens_budget']:,}")
        table.add_row("Tokens remaining", f"{cost_summary['tokens_remaining']:,}")
        table.add_row("Projected total", f"{cost_summary['projected_total']:,}")
        table.add_row("Est. cost (USD)", f"${cost_summary['estimated_cost_usd']:.4f}")
        table.add_row("Over budget?", "[red]YES[/red]" if cost_summary["over_budget"] else "[green]No[/green]")

        steps = self._store.all_steps() if hasattr(self._store, "all_steps") else []
        table.add_row("Steps recorded", str(len(steps)))

        self._console.print(table)

        # ── Signals detail ──────────────────────────────────────────────
        if self._signals_fired:
            sig_table = Table(title="Signals", show_lines=True)
            sig_table.add_column("Step", style="dim")
            sig_table.add_column("Type", style="yellow")
            sig_table.add_column("Action", style="red")
            sig_table.add_column("Confidence", style="magenta")
            sig_table.add_column("Detail")
            for s in self._signals_fired:
                sig_table.add_row(
                    str(s.step_number),
                    s.signal_type,
                    s.recommended_action,
                    f"{s.confidence:.0%}",
                    s.detail[:80],
                )
            self._console.print(sig_table)

        # ── Recovery actions ────────────────────────────────────────────
        if self._recovery_actions:
            rec_table = Table(title="Recovery Actions", show_lines=True)
            rec_table.add_column("Action", style="blue")
            rec_table.add_column("Reason")
            for a in self._recovery_actions:
                rec_table.add_row(a.get("action", "?"), str(a.get("reason", ""))[:80])
            self._console.print(rec_table)

        # ── Taint provenance DAG ────────────────────────────────────────
        if self._taint:
            taint_info = self._taint.summary()
            if taint_info["total_steps"] > 0:
                taint_table = Table(title="Taint Provenance DAG", show_lines=True)
                taint_table.add_column("Metric", style="cyan")
                taint_table.add_column("Value", style="white")
                taint_table.add_row("Steps tracked", str(taint_info["total_steps"]))
                taint_table.add_row(
                    "Tainted steps",
                    f"[red]{taint_info['tainted_steps']}[/red]"
                    if taint_info["tainted_steps"] > 0
                    else "[green]0[/green]",
                )
                if taint_info["edges"]:
                    edge_strs = [
                        f"{src} → {dsts}" for src, dsts in taint_info["edges"].items()
                    ]
                    taint_table.add_row("Data flow edges", "; ".join(edge_strs))
                self._console.print(taint_table)

        # ── Audit log snippet ───────────────────────────────────────────
        audit = self._store.audit_log()
        if audit:
            self._console.print(
                Panel(
                    f"[dim]{len(audit)} audit entries recorded. "
                    f"Review agentguard_{self.run_id}.db for full history.[/]",
                    title="Audit Trail",
                    border_style="dim",
                )
            )

        taint_str = ""
        if self._taint:
            ts = self._taint.summary()
            taint_str = f" | taint_edges={len(ts['edges'])} tainted_steps={ts['tainted_steps']}"

        return (
            f"run_id={self.run_id} | "
            f"steps={len(steps)} | "
            f"signals={len(self._signals_fired)} | "
            f"tokens={cost_summary['tokens_used']:,} | "
            f"cost=${cost_summary['estimated_cost_usd']:.4f}"
            f"{taint_str}"
            f"{' | ABORTED' if self._aborted else ''}"
        )

    # ------------------------------------------------------------------
    # Causal Replay — Idea 2
    # ------------------------------------------------------------------

    def explain_failure(self) -> dict:
        """
        Post-mortem causal analysis: binary-search through the stored
        checkpointed steps to find the *minimal* prefix that reproduces
        the failure signal seen in the full run.

        Call this after catching AgentGuardAbort:

            try:
                with guard.watch("task-123") as session:
                    agent.run(task)
            except AgentGuardAbort:
                analysis = session.explain_failure()
                print(analysis)
                # {
                #   'root_cause_step': 4,
                #   'tool_name': 'web_search',
                #   'signal_type': 'loop',
                #   'replay_confidence': 0.95,
                #   'explanation': "Removing steps after step 4 eliminates..."
                # }

        Returns {'status': 'not_aborted'} if the session ended cleanly.
        """
        if not self._aborted:
            return {"status": "not_aborted", "run_id": self.run_id}
        from .replay import CausalReplayer
        return CausalReplayer(self._store).find_root_cause()

    def taint_summary(self) -> dict:
        """
        Return the taint provenance DAG summary.

        Provides:
          - total_steps: how many steps were fingerprinted
          - tainted_steps: how many steps consumed data from a prior step
          - edges: {source_step: [consumer_steps]}
          - steps: per-step detail (tool, fingerprint count, taint sources)

        Returns an empty dict if taint tracking is not active.
        """
        if not self._taint:
            return {"status": "taint_tracking_inactive"}
        return self._taint.summary()

    # ------------------------------------------------------------------
    # Recovery outcome recording — Idea 3
    # ------------------------------------------------------------------

    def _record_recovery_outcome(self, succeeded: bool) -> None:
        """
        Record whether the last recovery action in this session ultimately
        led to a successful run completion.  Feeds RecoveryMemory so future
        runs for the same (signal_type, agent_type) pair can pick a better
        action automatically.

        Called automatically from __exit__ — no user action required.
        """
        memory = getattr(self._recovery, "_memory", None)
        agent_type = getattr(self._recovery, "_agent_type", "default")

        if not memory or not self._recovery_actions or not self._last_signal_type:
            return

        # Record the last recovery action taken in this session
        last_action = self._recovery_actions[-1].get("action", "unknown")
        try:
            memory.record_outcome(
                signal_type=self._last_signal_type,
                agent_type=agent_type,
                action=last_action,
                succeeded=succeeded,
            )
            logger.debug(
                "RecoveryMemory: signal=%s action=%s succeeded=%s agent=%s",
                self._last_signal_type,
                last_action,
                succeeded,
                agent_type,
            )
        except Exception:
            logger.exception("Failed to record recovery outcome — non-fatal, continuing")

