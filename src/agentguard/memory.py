"""
agentguard/memory.py
--------------------
Cross-run recovery memory.

Tracks which recovery actions succeeded per (signal_type, agent_type).
A recovery "succeeds" if the run completed without re-triggering the same
signal type in that session.

Persistence: JSON in ./agentguard_memory/recovery_memory.json
Thread safety: threading.Lock on all mutations.

Usage:
    memory = RecoveryMemory()
    action = memory.best_action("loop", "web_search_agent")  # e.g. "replan"
    memory.record_outcome("loop", "web_search_agent", action="replan", succeeded=True)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MIN_ATTEMPTS = 5


class RecoveryMemory:
    """
    Stores per-(signal_type, agent_type) recovery outcomes.

    Data structure:
        {"<signal>:<agent>": {"<action>": [successes, attempts]}}
    """

    def __init__(self, base_dir: str = "./agentguard_memory"):
        os.makedirs(base_dir, exist_ok=True)
        self._path = os.path.join(base_dir, "recovery_memory.json")
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, list[int]]] = {}
        self._load()

    def record_outcome(
        self,
        signal_type: str,
        agent_type: str,
        action: str,
        succeeded: bool,
    ) -> None:
        """Record whether a recovery action succeeded for this (signal, agent) pair."""
        key = f"{signal_type}:{agent_type}"
        with self._lock:
            self._data.setdefault(key, {}).setdefault(action, [0, 0])
            self._data[key][action][1] += 1
            if succeeded:
                self._data[key][action][0] += 1
            self._save()
        logger.debug(
            "RecoveryMemory: %s/%s action=%s succeeded=%s", signal_type, agent_type, action, succeeded
        )

    def best_action(
        self,
        signal_type: str,
        agent_type: str,
        fallback: str = "replan",
        min_attempts: int = _DEFAULT_MIN_ATTEMPTS,
        epsilon: float = 0.0,
    ) -> str:
        """
        Return the historically highest-success-rate action using UCB1 selection.
        If epsilon > 0, performs epsilon-greedy exploration.
        """
        import random
        import math

        key = f"{signal_type}:{agent_type}"
        with self._lock:
            actions = dict(self._data.get(key, {}))

        if not actions:
            return fallback

        # Epsilon-greedy exploration check
        if epsilon > 0 and random.random() < epsilon:
            chosen = random.choice(list(actions.keys()))
            logger.debug("RecoveryMemory (epsilon-explore): %s/%s → %s", signal_type, agent_type, chosen)
            return chosen

        total_trials = sum(attempts for _, attempts in actions.values())
        best_action, best_score = fallback, -1.0

        for action, (successes, attempts) in actions.items():
            if attempts < min_attempts:
                continue
            # UCB1 score calculation: exploitation + exploration boost
            exploit = successes / attempts
            explore = math.sqrt(2 * math.log(max(1, total_trials)) / attempts) if total_trials > 0 else 0
            ucb_score = exploit + 0.1 * explore  # small UCB bonus

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action

        return best_action

    def success_rate(self, signal_type: str, agent_type: str, action: str) -> Optional[float]:
        """Return observed success rate for a specific action, or None if no data."""
        key = f"{signal_type}:{agent_type}"
        with self._lock:
            entry = self._data.get(key, {}).get(action)
        if not entry or entry[1] == 0:
            return None
        return entry[0] / entry[1]

    def summary(self) -> dict:
        with self._lock:
            return {
                k: {
                    action: {"successes": s, "attempts": a, "rate": f"{s/a:.0%}" if a else "n/a"}
                    for action, (s, a) in v.items()
                }
                for k, v in self._data.items()
            }

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError as exc:
            logger.warning("RecoveryMemory: save failed: %s", exc)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._data = loaded
                logger.debug("RecoveryMemory: loaded %d entries from %s", len(self._data), self._path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("RecoveryMemory: load failed (%s) — starting fresh", exc)
