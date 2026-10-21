"""
agentguard/recovery.py
----------------------
Semantic recovery router: given a FailureSignal, pick the right fix.

Security fixes vs original guide:
  1. Retry cap — tool_error retries are capped at max_retries (default 3)
     with exponential backoff. Without this, a persistent error burns the
     entire token budget before the CostGuardian fires.
  2. inject_prompt is ALWAYS hardcoded — never derived from signal.detail or
     any other external string. Prevents prompt injection via crafted agent
     outputs that embed instruction text in error messages.
  3. Webhook delivery is HMAC-signed (SHA-256). The receiving server should
     verify X-AgentGuard-Signature before trusting the payload.
  4. signal.detail is passed through Untrusted.sanitize() before being
     included in any outbound payload.
  5. Model downgrade is opt-in (allow_model_downgrade=True in config) so
     adversarial budget exhaustion can't silently degrade a security-critical
     agent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from typing import Any

import httpx

from .memory import RecoveryMemory
from .models import AuditEntry, FailureSignal, Untrusted
from .state import CheckpointStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded recovery prompts — NEVER dynamically generated from signal.detail
# ---------------------------------------------------------------------------

_REPLAN_PROMPT = (
    "You appear to be repeating the same action without making progress. "
    "Stop, reconsider the goal, and choose a materially different approach."
)

_INFO_COLLAPSE_PROMPT = (
    "Your recent responses have contained progressively less useful information. "
    "Re-evaluate whether you have enough context to continue, "
    "or clarify what additional information is needed."
)


class RecoveryRouter:
    """
    Classify a FailureSignal and return a recovery action dict.

    The caller (GuardSession) is responsible for executing the action —
    this class only decides what to do, it doesn't do it.
    """

    def __init__(
        self,
        store: CheckpointStore,
        config: dict[str, Any],
        memory: RecoveryMemory | None = None,
        agent_type: str = "default",
    ):
        self._store = store
        self._config = config
        self._webhook_url: str = config.get("alert_webhook", "")
        self._webhook_secret: str = config.get("webhook_secret", "")
        self._max_retries: int = int(config.get("max_retries", 3))
        self._allow_downgrade: bool = bool(config.get("allow_model_downgrade", False))
        self._cheap_model: str = config.get("cheap_model", "claude-haiku-4-5-20251001")
        self._memory: RecoveryMemory | None = memory
        self._agent_type: str = agent_type

        self._lock = threading.Lock()
        # Per-run retry counter — keyed by run_id so concurrent runs don't share
        self._retry_counts: dict[str, int] = {}
        # Webhook rate limiting: {signal_type: last_sent_timestamp}
        self._webhook_last_sent: dict[str, float] = {}
        self._webhook_cooldown_seconds: float = 30.0  # max 1 webhook per type per 30s

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(self, signal: FailureSignal) -> dict[str, Any]:
        """
        Return a recovery action dict. Side effect: writes an AuditEntry.
        Never raises — logs errors and returns an abort action if something
        unexpected happens so the agent doesn't silently continue.
        """
        try:
            action = self._route(signal)
        except Exception:
            logger.exception("RecoveryRouter.handle() failed unexpectedly")
            action = {"action": "abort", "reason": "Recovery router internal error"}

        self._record_audit(signal, action)
        return action

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route(self, signal: FailureSignal) -> dict[str, Any]:
        last_good = self._store.last_trusted_step()
        resume_step = last_good.step_number if last_good else 0

        if signal.signal_type == "tool_error":
            return self._handle_tool_error(signal, resume_step)

        if signal.signal_type == "loop":
            # Respect configured on_failure policy or explicit abort recommendation
            configured = self._config.get("on_failure", "replan")
            if configured == "abort" or signal.recommended_action == "abort":
                best = "abort"
            else:
                best = "replan"

            memory_driven = False
            if self._memory and configured != "abort" and signal.recommended_action != "abort":
                best = self._memory.best_action("loop", self._agent_type, fallback="replan")
                memory_driven = True
            return {
                "action": best,
                "resume_from_step": resume_step,
                # inject_prompt is ALWAYS hardcoded — never derived from signal.detail
                "inject_prompt": _REPLAN_PROMPT,
                "reason": Untrusted(signal.detail).sanitize(),
                "memory_driven": memory_driven,
            }

        if signal.signal_type == "taint_propagation":
            # Taint propagation is a critical security event — always abort.
            # The signal detail contains the clean rollback prefix.
            self._push_to_human_queue(signal, resume_step)
            return {
                "action": "abort",
                "reason": Untrusted(signal.detail).sanitize(),
                "resume_from_step": resume_step,
                "taint_origin": True,
            }

        if signal.recommended_action == "abort":
            return {
                "action": "abort",
                "reason": Untrusted(signal.detail).sanitize(),
                "resume_from_step": resume_step,
            }

        if signal.signal_type in ("info_collapse", "drift"):
            self._push_to_human_queue(signal, resume_step)
            return {
                "action": "escalate",
                "resume_from_step": resume_step,
                # Hardcoded — NOT derived from signal.detail
                "inject_prompt": _INFO_COLLAPSE_PROMPT,
                "reason": Untrusted(signal.detail).sanitize(),
            }

        if signal.signal_type == "budget_pressure":
            return self._handle_budget_pressure(signal)

        # Unknown signal type — fail safe
        logger.warning("Unknown signal_type %r — aborting", signal.signal_type)
        return {"action": "abort", "reason": f"Unknown signal type: {signal.signal_type}"}

    # ------------------------------------------------------------------
    # Tool error with retry cap + exponential backoff
    # ------------------------------------------------------------------

    def _handle_tool_error(self, signal: FailureSignal, resume_step: int) -> dict[str, Any]:
        key = signal.run_id
        with self._lock:
            current = self._retry_counts.get(key, 0) + 1
            self._retry_counts[key] = current

        if current > self._max_retries:
            logger.warning(
                "run_id=%s exceeded max_retries=%d — aborting",
                signal.run_id, self._max_retries,
            )
            return {
                "action": "abort",
                "reason": (
                    f"Max retries ({self._max_retries}) exceeded after consecutive tool errors. "
                    f"Last error: {Untrusted(signal.detail).sanitize()}"
                ),
            }

        backoff_ms = 1000 * (2 ** (current - 1))  # 1s, 2s, 4s, 8s…
        return {
            "action": "retry",
            "resume_from_step": resume_step,
            "backoff_ms": backoff_ms,
            "retry_attempt": current,
            "max_retries": self._max_retries,
            "reason": Untrusted(signal.detail).sanitize(),
        }

    # ------------------------------------------------------------------
    # Budget pressure — model downgrade is opt-in
    # ------------------------------------------------------------------

    def _handle_budget_pressure(self, signal: FailureSignal) -> dict[str, Any]:
        if not self._allow_downgrade:
            logger.info(
                "Budget pressure detected but allow_model_downgrade=False — warning only"
            )
            return {
                "action": "warn",
                "reason": Untrusted(signal.detail).sanitize(),
                "note": "Set allow_model_downgrade=True in Guard config to enable auto-downgrade",
            }
        return {
            "action": "downgrade_model",
            "new_model": self._cheap_model,
            "reason": Untrusted(signal.detail).sanitize(),
        }

    # ------------------------------------------------------------------
    # HMAC-signed webhook delivery
    # ------------------------------------------------------------------

    def _push_to_human_queue(self, signal: FailureSignal, last_good_step: int) -> None:
        if not self._webhook_url:
            logger.info("No alert_webhook configured — escalation logged locally only")
            return

        # Rate limiting: don't spam the webhook endpoint
        now = time.monotonic()
        with self._lock:
            last_sent = self._webhook_last_sent.get(signal.signal_type, 0.0)
            if now - last_sent < self._webhook_cooldown_seconds:
                logger.debug(
                    "Webhook rate-limited for signal_type=%s (%.1fs cooldown)",
                    signal.signal_type, self._webhook_cooldown_seconds,
                )
                return
            self._webhook_last_sent[signal.signal_type] = now

        payload = {
            "run_id": signal.run_id,
            "step_number": signal.step_number,
            "signal_type": signal.signal_type,
            "confidence": signal.confidence,
            # sanitize before sending — detail may contain LLM-generated text
            "detail": Untrusted(signal.detail).sanitize(),
            "last_good_step": last_good_step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        body = json.dumps(payload).encode("utf-8")
        timestamp_str = str(int(time.time()))

        headers = {
            "Content-Type": "application/json",
            "X-AgentGuard-Timestamp": timestamp_str,
        }

        if self._webhook_secret:
            sig_input = f"{timestamp_str}.{body.decode('utf-8')}"
            signature = hmac.new(
                self._webhook_secret.encode("utf-8"),
                sig_input.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-AgentGuard-Signature"] = f"sha256={signature}"
        else:
            logger.warning(
                "webhook_secret not configured — webhook payload is unsigned. "
                "Set webhook_secret in Guard config for production use."
            )

        try:
            resp = httpx.post(
                self._webhook_url,
                content=body,
                headers=headers,
                verify=True,        # enforce TLS certificate verification
                timeout=5.0,
            )
            resp.raise_for_status()
            logger.info("Webhook delivered: HTTP %d", resp.status_code)
        except httpx.HTTPError as exc:
            # Log and continue — webhook failure must not crash the guard
            logger.error("Webhook delivery failed: %s", exc)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _record_audit(self, signal: FailureSignal, action: dict[str, Any]) -> None:
        try:
            self._store.save_audit_event(
                AuditEntry(
                    run_id=signal.run_id,
                    entry_type="recovery_action",
                    action=action.get("action"),
                    triggered_by=signal.signal_type,
                    step_number=signal.step_number,
                    detail=action.get("reason"),
                )
            )
        except Exception:
            logger.exception("Failed to write audit entry — continuing")
