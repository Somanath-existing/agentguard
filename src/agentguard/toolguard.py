"""
agentguard/toolguard.py
-----------------------
Tool authorization layer: allowlist/denylist enforcement for agent tool calls.

This is the "what is the agent doing?" counterpart to the signal engine's
"how is the agent behaving?" — it inspects the *identity* of each tool call
rather than just its behavioral patterns.

Security model:
  - If an allowlist is configured, ONLY those tools may be called (whitelist mode).
  - If a denylist is configured, those tools are blocked (blacklist mode).
  - If both are configured, the allowlist takes precedence (allowlist-first).
  - Tool name matching is case-insensitive and supports glob patterns.
  - Argument inspection hooks allow blocking dangerous argument patterns
    (e.g., shell injection in tool args).
  - All authorization decisions are logged to the audit trail.

Usage:
    guard = Guard(
        tool_allowlist=["web_search", "calculator", "read_file"],
        # OR
        tool_denylist=["shell_exec", "run_code", "delete_*"],
    )
"""

from __future__ import annotations

import fnmatch
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authorization result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolAuthResult:
    """Result of a tool authorization check."""
    allowed: bool
    tool_name: str
    reason: str
    matched_rule: Optional[str] = None  # which pattern matched


# ---------------------------------------------------------------------------
# Dangerous argument patterns (built-in)
# ---------------------------------------------------------------------------

# Patterns that indicate shell injection / command injection in tool args
_DANGEROUS_ARG_PATTERNS = [
    re.compile(r";\s*(rm|del|rmdir|format|mkfs)\b", re.IGNORECASE),
    re.compile(r"\|\s*(bash|sh|cmd|powershell|pwsh)\b", re.IGNORECASE),
    re.compile(r"&&\s*(rm|del|curl|wget|nc|ncat)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`"),                    # backtick command substitution
    re.compile(r"\$\([^)]+\)"),                # $(command) substitution
    re.compile(r">\s*/dev/|>\s*[A-Z]:\\", re.IGNORECASE),  # file redirect to system paths
]


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------

class ToolGuard:
    """
    Enforces tool-call authorization policies.

    Instantiate one per Guard — shared across all watch() sessions.
    Thread-safe: all mutable state (counters) protected by lock.
    """

    def __init__(
        self,
        allowlist: Optional[list[str]] = None,
        denylist: Optional[list[str]] = None,
        arg_inspection: bool = True,
        custom_validators: Optional[list[Callable[[str, str], Optional[str]]]] = None,
    ):
        """
        Args:
            allowlist: If set, only tools matching these patterns are allowed.
                       Supports glob patterns (e.g., "read_*").
            denylist:  If set, tools matching these patterns are blocked.
                       Ignored if allowlist is also set.
            arg_inspection: If True, tool arguments are checked against
                           built-in dangerous patterns.
            custom_validators: List of callables (tool_name, tool_args) -> Optional[str].
                              Return None if OK, or a reason string if blocked.
        """
        # Normalize to lowercase for case-insensitive matching
        self._allowlist = [p.lower() for p in allowlist] if allowlist else None
        self._denylist = [p.lower() for p in denylist] if denylist else None
        self._arg_inspection = arg_inspection
        self._custom_validators = custom_validators or []

        self._lock = threading.Lock()
        self._stats: dict[str, int] = {"allowed": 0, "denied": 0}

        mode = "allowlist" if self._allowlist else ("denylist" if self._denylist else "permissive")
        logger.info(
            "ToolGuard initialized: mode=%s, arg_inspection=%s, custom_validators=%d",
            mode, arg_inspection, len(self._custom_validators),
        )

    # ------------------------------------------------------------------
    # Main authorization check
    # ------------------------------------------------------------------

    def authorize(self, tool_name: str, tool_args: str = "") -> ToolAuthResult:
        """
        Check if a tool call is authorized.

        Args:
            tool_name: Name of the tool being invoked.
            tool_args: Serialized arguments (for injection detection).

        Returns:
            ToolAuthResult with allowed=True/False and reason.
        """
        if not tool_name:
            return ToolAuthResult(allowed=True, tool_name="", reason="No tool name — LLM span")

        normalized = tool_name.lower().strip()

        # --- Allowlist mode (strictest) ---
        if self._allowlist is not None:
            matched = self._match_patterns(normalized, self._allowlist)
            if not matched:
                with self._lock:
                    self._stats["denied"] += 1
                return ToolAuthResult(
                    allowed=False,
                    tool_name=tool_name,
                    reason=f"Tool '{tool_name}' not in allowlist",
                    matched_rule="allowlist_miss",
                )

        # --- Denylist mode ---
        if self._allowlist is None and self._denylist is not None:
            matched = self._match_patterns(normalized, self._denylist)
            if matched:
                with self._lock:
                    self._stats["denied"] += 1
                return ToolAuthResult(
                    allowed=False,
                    tool_name=tool_name,
                    reason=f"Tool '{tool_name}' matches denylist pattern '{matched}'",
                    matched_rule=f"denylist:{matched}",
                )

        # --- Argument injection inspection ---
        if self._arg_inspection and tool_args:
            for pattern in _DANGEROUS_ARG_PATTERNS:
                match = pattern.search(tool_args)
                if match:
                    with self._lock:
                        self._stats["denied"] += 1
                    return ToolAuthResult(
                        allowed=False,
                        tool_name=tool_name,
                        reason=f"Dangerous argument pattern detected in '{tool_name}' args: '{match.group()[:50]}'",
                        matched_rule=f"arg_injection:{pattern.pattern[:30]}",
                    )

        # --- Custom validators ---
        for validator in self._custom_validators:
            try:
                deny_reason = validator(tool_name, tool_args)
                if deny_reason:
                    with self._lock:
                        self._stats["denied"] += 1
                    return ToolAuthResult(
                        allowed=False,
                        tool_name=tool_name,
                        reason=deny_reason[:200],
                        matched_rule="custom_validator",
                    )
            except Exception:
                logger.exception("Custom tool validator raised for tool=%s", tool_name)

        # --- All checks passed ---
        with self._lock:
            self._stats["allowed"] += 1

        return ToolAuthResult(allowed=True, tool_name=tool_name, reason="Authorized")

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_patterns(name: str, patterns: list[str]) -> Optional[str]:
        """Return the first matching pattern, or None."""
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return pattern
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return authorization statistics."""
        with self._lock:
            return dict(self._stats)
