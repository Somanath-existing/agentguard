"""
agentguard/models.py
--------------------
Pydantic v2 data models for all AgentGuard events.

Security notes:
  - timestamp uses Field(default_factory=...) — NOT datetime.utcnow() as a
    default value, which evaluates once at class definition time and gives every
    instance the same timestamp (forensic corruption).
  - input_hash stores a SHA-256 hex digest only — raw prompt text is NEVER
    persisted in StepEvent. Callers hash before constructing the model.
  - Untrusted[str] marks fields whose value came from external sources
    (span attributes, LLM output, error strings). Any consumer must sanitize
    before using in prompts, filenames, or SQL.
  - AgentGuardAbort is a custom exception used by the enforcement layer to
    actually stop a runaway agent (not just log warnings).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
import re

# Precompile regexes and define translation map for Untrusted sanitize to avoid overhead in hot path
_CONFUSABLE_MAP = str.maketrans({
    '\u1d00': 'a', '\u0299': 'b', '\u1d04': 'c', '\u1d05': 'd',
    '\u1d07': 'e', '\ua730': 'f', '\u0262': 'g', '\u029c': 'h',
    '\u026a': 'i', '\u1d0a': 'j', '\u1d0b': 'k', '\u029f': 'l',
    '\u1d0d': 'm', '\u0274': 'n', '\u1d0f': 'o', '\u1d18': 'p',
    '\u0280': 'r', '\ua731': 's', '\u1d1b': 't', '\u1d1c': 'u',
    '\u1d20': 'v', '\u1d21': 'w', '\u028f': 'y', '\u1d22': 'z',
})

_INJECTION_MARKERS_RE = re.compile(
    r"("
    # Generic role prefixes
    r"system\s*:|user\s*:|assistant\s*:"
    # Anthropic legacy format
    r"|human\s*:|ai\s*:"
    # OpenAI chat-ml tokens
    r"|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
    # Generic pipe-delimited tokens
    r"|<\|.*?\|>"
    # LLaMA / Mistral instruction tags
    r"|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>"
    # Claude thinking / answer tags
    r"|</?thinking>|</?answer>"
    # Markdown heading injection
    r"|###"
    r")",
    re.IGNORECASE
)

_CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')



# ---------------------------------------------------------------------------
# Enforcement exception — the kill mechanism
# ---------------------------------------------------------------------------

class AgentGuardAbort(Exception):
    """
    Raised by GuardSession when enforcement mode is active and a signal
    with recommended_action='abort' fires, or when the hard token budget
    is exceeded.

    Agents / callers should catch this if they want graceful shutdown:

        try:
            with guard.watch(run_id="task") as session:
                agent.run(task)
        except AgentGuardAbort as e:
            print(f"AgentGuard killed the run: {e}")

    If not caught, the agent process terminates — which is the intended
    behavior for a guardrail.
    """

    def __init__(self, reason: str, signal_type: str = "unknown", step_number: int = 0):
        self.reason = reason
        self.signal_type = signal_type
        self.step_number = step_number
        super().__init__(f"[AgentGuard ABORT] step={step_number} type={signal_type}: {reason}")


# ---------------------------------------------------------------------------
# Trust boundary marker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Untrusted:
    """
    Wraps a string that came from an external / attacker-influenced source
    (LLM completions, span attributes, error messages).

    Any function that consumes an Untrusted value MUST call .sanitize() or
    otherwise validate it before:
      - interpolating into a prompt
      - using as a filesystem path
      - sending to a webhook
      - writing into a log that drives further automation

    The frozen dataclass means you can't accidentally mutate it.
    """
    value: str

    def sanitize(self, max_len: int = 500) -> str:
        """
        Strip prompt-injection markers and cap length.
        Safe to use in log messages and human-readable output.
        NOT safe for shell execution or SQL without further escaping.

        Covers:
          - OpenAI chat-ml: <|im_start|>, <|im_end|>, <|endoftext|>
          - LLaMA / Mistral: [INST], [/INST], <<SYS>>, <</SYS>>
          - Anthropic legacy: Human:, AI:
          - Claude: <thinking>, </thinking>, <answer>, </answer>
          - Generic role prefixes: system:, user:, assistant:
          - Markdown heading injection: ###
          - Unicode confusable role prefixes (small-cap variants)
        """
        normalized = self.value.translate(_CONFUSABLE_MAP)

        # Phase 2: Strip all known prompt-injection markers
        cleaned = _INJECTION_MARKERS_RE.sub("", normalized)

        # Phase 3: Strip any remaining control characters (U+0000–U+001F except newline/tab)
        cleaned = _CONTROL_CHARS_RE.sub("", cleaned)

        return cleaned.strip()[:max_len]

    def __repr__(self) -> str:
        preview = self.value[:40] + "..." if len(self.value) > 40 else self.value
        return f"Untrusted({preview!r})"


# ---------------------------------------------------------------------------
# Step event — one per intercepted OTel span
# ---------------------------------------------------------------------------

class StepEvent(BaseModel):
    """
    Represents a single intercepted tool / LLM call.

    Fields that carry external data use Optional[str] but callers are
    expected to pass only sanitized or hashed values (see interceptor.py).
    """

    run_id: str
    step_number: int
    span_id: str
    span_name: str                     # e.g. "tool.web_search"
    tool_name: Optional[str] = None

    # SHA-256 hex digest of the raw input — used for loop detection.
    # Raw prompt text is NEVER stored here.
    input_hash: Optional[str] = None

    output_size: int = 0               # character count of output
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    status: Literal["ok", "error", "timeout"] = "ok"

    # Sanitized error type string — callers must call Untrusted.sanitize()
    # before assigning here.
    error_type: Optional[str] = None

    # Fix: default_factory ensures each instance gets the current time, not
    # the class-definition time.
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @field_validator("run_id")
    @classmethod
    def run_id_safe(cls, v: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_-]", "_", v)[:64]
        if not clean:
            raise ValueError("run_id must contain at least one alphanumeric character")
        return clean

    @field_validator("input_hash")
    @classmethod
    def hash_looks_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) not in (16, 32, 64):
            raise ValueError("input_hash must be a hex digest (16, 32, or 64 chars)")
        return v

    @staticmethod
    def make_input_hash(raw_input: str) -> str:
        """
        Returns a full SHA-256 hex digest (64 chars = 256 bits).

        Previous guide used [:16] (64 bits) which has a non-trivial collision
        probability for adversarial inputs. Full digest is cheaper than the
        consequence of a hash collision evading loop detection.
        """
        import hashlib
        return hashlib.sha256(raw_input.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Failure signal — emitted by SignalEngine
# ---------------------------------------------------------------------------

class FailureSignal(BaseModel):
    run_id: str
    step_number: int
    signal_type: Literal["loop", "drift", "budget_pressure", "tool_error", "info_collapse", "unauthorized_tool", "predictive_failure", "taint_propagation"]
    confidence: float          # 0.0 – 1.0
    detail: str                # human-readable, sanitized before storage
    recommended_action: Literal["continue", "warn", "redirect", "abort"]

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 4)


# ---------------------------------------------------------------------------
# Audit log entry — append-only, written by GuardSession
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """
    Immutable record of every recovery action taken.
    Written with INSERT (not INSERT OR REPLACE) so history cannot be silently
    overwritten.
    """
    run_id: str
    entry_type: Literal["recovery_action", "signal_fired", "budget_event", "session_start", "session_end", "tool_blocked"]
    action: Optional[str] = None       # e.g. "retry", "abort", "downgrade_model"
    triggered_by: Optional[str] = None # signal_type that caused this
    step_number: Optional[int] = None
    detail: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
