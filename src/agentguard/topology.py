"""
agentguard/topology.py
----------------------
Multi-agent topology tracker.

Builds a parent → children DAG using run_ids and optional parent_run_ids.
Enforces a shared token budget across the entire agent tree so sub-agents
cannot silently exhaust the parent's budget.
Detects orchestrator-level loops (same sub-agent type spawned 2+ times in
the last 10 sub-tasks).
Scans inter-agent message boundaries for injection propagation.

Thread safety: single threading.Lock for all state mutations.

Usage:
    from agentguard import Guard
    from agentguard.topology import TopologyTracker

    topology = TopologyTracker(total_budget=500_000)

    orch_guard = Guard(topology=topology, agent_type="orchestrator")
    with orch_guard.watch("orch-run-1") as orch_session:
        for subtask in tasks:
            sub_guard = Guard(
                topology=topology,
                agent_type="researcher",
                parent_run_id="orch-run-1",
            )
            with sub_guard.watch(f"sub-{subtask.id}") as sub_session:
                researcher_agent.run(subtask)

    print(topology.summary())
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .injection import InjectionResult, scan_for_injection
from .models import Untrusted

logger = logging.getLogger(__name__)


@dataclass
class AgentNode:
    """Represents a single agent run in the topology tree."""
    run_id: str
    parent_run_id: str | None
    agent_type: str
    tokens_used: int = 0
    completed: bool = False
    children: list[AgentNode] = field(default_factory=list)


@dataclass(frozen=True)
class InterAgentScanResult:
    """Result of scanning content at an inter-agent boundary."""
    blocked: bool                          # True if injection was found
    sanitized_content: str                 # safe version of the content
    violation: InjectionResult | None   # details of the injection, or None
    from_run_id: str = ""
    to_run_id: str = ""


class TopologyTracker:
    """
    Shared coordinator for a multi-agent system.

    Pass the SAME TopologyTracker instance to every Guard in the system.
    The tracker maintains the agent tree, enforces a shared token budget,
    and detects orchestrator-level spawn loops.
    """

    def __init__(self, total_budget: int = 500_000):
        """
        Args:
            total_budget: Maximum total tokens across ALL agents in the tree.
                          Set to a value that covers the full orchestration task.
        """
        if total_budget <= 0:
            raise ValueError("total_budget must be > 0")
        self._total_budget = total_budget
        self._lock = threading.Lock()
        self._nodes: dict[str, AgentNode] = {}
        # {parent_run_id: deque of agent_type strings, maxlen=10}
        self._spawn_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        # Records of injection blocked at inter-agent boundaries
        self._boundary_violations: list[dict] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def register(
        self,
        run_id: str,
        agent_type: str,
        parent_run_id: str | None = None,
    ) -> None:
        """
        Register a new agent run in the topology.

        Call this at the start of Guard.watch() before yielding the session.
        """
        with self._lock:
            node = AgentNode(
                run_id=run_id,
                parent_run_id=parent_run_id,
                agent_type=agent_type,
            )
            self._nodes[run_id] = node
            if parent_run_id and parent_run_id in self._nodes:
                self._nodes[parent_run_id].children.append(node)
                # Record spawn for orchestrator loop detection
                self._spawn_history[parent_run_id].append(agent_type)
                logger.debug(
                    "TopologyTracker: registered sub-agent run_id=%s type=%s under parent=%s",
                    run_id, agent_type, parent_run_id,
                )
            else:
                logger.debug(
                    "TopologyTracker: registered root agent run_id=%s type=%s",
                    run_id, agent_type,
                )

    def mark_completed(self, run_id: str) -> None:
        """Mark an agent run as completed (called at session close)."""
        with self._lock:
            if run_id in self._nodes:
                self._nodes[run_id].completed = True

    def record_tokens(self, run_id: str, tokens: int) -> None:
        """Accumulate token usage for a run. Thread-safe."""
        with self._lock:
            if run_id in self._nodes:
                self._nodes[run_id].tokens_used += tokens

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def total_tokens_used(self) -> int:
        """Sum of tokens used across ALL registered agents."""
        with self._lock:
            return sum(n.tokens_used for n in self._nodes.values())

    def is_tree_over_budget(self) -> bool:
        """True if the aggregate token spend has hit or exceeded the total budget."""
        return self.total_tokens_used() >= self._total_budget

    def remaining_budget(self) -> int:
        """Tokens remaining in the shared tree budget (floored at 0)."""
        return max(0, self._total_budget - self.total_tokens_used())

    # ------------------------------------------------------------------
    # Orchestrator loop detection
    # ------------------------------------------------------------------

    def check_orchestrator_loop(self, parent_run_id: str) -> str | None:
        """
        Return a warning string if the orchestrator is spawning the same
        sub-agent type repeatedly within its last 10 sub-tasks.

        Returns None if no loop is detected.
        """
        with self._lock:
            history = list(self._spawn_history.get(parent_run_id, []))

        if len(history) < 2:
            return None

        last = history[-1]
        count = history.count(last)
        if count >= 2:
            msg = (
                f"Orchestrator '{parent_run_id}' spawned agent_type='{last}' "
                f"{count}x in the last {len(history)} sub-tasks — possible orchestrator loop"
            )
            logger.warning("TopologyTracker: %s", msg)
            return msg

        return None

    # ------------------------------------------------------------------
    # Cross-agent injection propagation scanning
    # ------------------------------------------------------------------

    def scan_inter_agent_message(
        self,
        from_run_id: str,
        to_run_id: str,
        content: str,
    ) -> InterAgentScanResult:
        """
        Scan content passing between agents for injection markers.

        When Agent A's output becomes Agent B's input, this method should
        be called at the boundary.  If an injection is detected, the
        content is sanitized before delivery.

        Security: raw content is scanned then discarded.  Only a short
        snippet (≤80 chars) is retained in the InjectionResult for audit.
        """
        injection = scan_for_injection(content)
        if injection:
            raw_sanitized = Untrusted(content).sanitize()
            # Active redaction: remove matched injection snippet from boundary message
            matched_text = injection.snippet.strip()
            if matched_text and matched_text in raw_sanitized:
                sanitized = raw_sanitized.replace(matched_text, "[REDACTED_INJECTION]")
            else:
                sanitized = raw_sanitized
            with self._lock:
                self._boundary_violations.append({
                    "from": from_run_id,
                    "to": to_run_id,
                    "label": injection.label,
                    "confidence": injection.confidence,
                    "snippet": injection.snippet,
                })
            logger.warning(
                "TopologyTracker: injection blocked at boundary %s → %s [%s] (conf=%.2f)",
                from_run_id, to_run_id, injection.label, injection.confidence,
            )
            return InterAgentScanResult(
                blocked=True,
                sanitized_content=sanitized,
                violation=injection,
                from_run_id=from_run_id,
                to_run_id=to_run_id,
            )
        return InterAgentScanResult(
            blocked=False,
            sanitized_content=content,
            violation=None,
            from_run_id=from_run_id,
            to_run_id=to_run_id,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a full snapshot of the topology and budget state."""
        with self._lock:
            return {
                "total_budget":    self._total_budget,
                "total_used":      sum(n.tokens_used for n in self._nodes.values()),
                "remaining":       max(0, self._total_budget - sum(n.tokens_used for n in self._nodes.values())),
                "over_budget":     sum(n.tokens_used for n in self._nodes.values()) >= self._total_budget,
                "agent_count":     len(self._nodes),
                "boundary_violations": len(self._boundary_violations),
                "agents": {
                    run_id: {
                        "type":      node.agent_type,
                        "tokens":    node.tokens_used,
                        "parent":    node.parent_run_id,
                        "children":  len(node.children),
                        "completed": node.completed,
                    }
                    for run_id, node in self._nodes.items()
                },
            }
