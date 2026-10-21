"""
agentguard/injection.py
-----------------------
Indirect prompt injection detector.

Scans LLM *outputs* (what the model decided to do next) for signs
that the model was manipulated by a malicious tool response embedded
with instruction-override text ("Ignore previous instructions…").

This is distinct from ToolGuard which inspects *outgoing* tool arguments.
Here we inspect the LLM's own completion text BEFORE discarding it.

Security design:
  - Text is NEVER stored — only a max-80-char sanitized snippet is kept.
  - All patterns are compiled once at import time (no re-compilation per call).
  - scan_for_injection() is a pure function: no side effects, no state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── Injection signature patterns ──────────────────────────────────────────────
# Each entry: (label, compiled_pattern, confidence_score)
# Confidence represents how strongly the pattern indicates a real injection.
_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # "You are now a [different persona]" — persona hijack
    (
        "persona_hijack",
        re.compile(r"you\s+are\s+now\s+(a|an)\s+\w+", re.I),
        0.90,
    ),

    # "Ignore previous/prior/all/above instructions/context/rules" — direct override
    (
        "instruction_override",
        re.compile(
            r"ignore\s+(previous|prior|all|above)\s+(instructions?|context|rules?|directives?)",
            re.I,
        ),
        0.95,
    ),

    # "New task: / Actual task: / Real objective:" — goal substitution
    (
        "new_task_injection",
        re.compile(r"(new\s+task|actual\s+task|real\s+objective|true\s+goal)\s*:", re.I),
        0.80,
    ),

    # Embedded role delimiter tags (OpenAI chatML, Anthropic, LLaMA)
    (
        "role_confusion",
        re.compile(r"<\|?(system|user|assistant)\|?>", re.I),
        0.85,
    ),

    # Exfiltration command targeting credentials
    (
        "exfil_pattern",
        re.compile(
            r"(send|post|upload|exfiltrate|transmit|leak).{0,40}"
            r"(api[_\s]?key|password|secret|token|credential|auth)",
            re.I,
        ),
        0.92,
    ),

    # Known jailbreak prefixes
    (
        "jailbreak_prefix",
        re.compile(
            r"\b(DAN|developer\s+mode|jailbreak|unrestricted\s+mode|"
            r"god\s+mode|no\s+restrictions)\b",
            re.I,
        ),
        0.88,
    ),

    # "Disregard your instructions" — softer phrasing variant
    (
        "disregard_override",
        re.compile(
            r"(disregard|forget|override|bypass)\s+(your\s+)?"
            r"(instructions?|guidelines?|constraints?|safety\s+rules?)",
            re.I,
        ),
        0.87,
    ),

    # Prompt delimiter injection — attempt to close and reopen context
    (
        "delimiter_injection",
        re.compile(r"(<<<|>>>|\[\[|\]\]|---END\s+PROMPT---|---START\s+PROMPT---)", re.I),
        0.75,
    ),
]


@dataclass(frozen=True)
class InjectionResult:
    """
    Immutable result from scan_for_injection().

    Only a sanitized snippet (≤80 chars) is stored — never the full text.
    The snippet is enough for an operator to understand the attack vector.
    """
    detected: bool
    label: str            # which pattern matched
    confidence: float     # 0.0–1.0
    snippet: str          # ≤80 chars of context around the match


def scan_for_injection(text: str) -> InjectionResult | None:
    """
    Scan LLM output text for prompt-injection markers.

    Returns an InjectionResult if any pattern matches, else None.
    The caller MUST discard `text` after this call — it is not retained here.

    Performance: O(P × N) where P = number of patterns (8), N = text length.
    Patterns short-circuit on first match — most injections hit early patterns.
    """
    if not text:
        return None

    # Find the highest-confidence match across all patterns
    best: InjectionResult | None = None

    for label, pattern, confidence in _PATTERNS:
        match = pattern.search(text)
        if match:
            # Extract context window: 20 chars before + 20 chars after match
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            snippet = text[start:end].replace("\n", " ").replace("\r", " ")[:80]

            result = InjectionResult(
                detected=True,
                label=label,
                confidence=confidence,
                snippet=snippet,
            )
            # Keep the highest-confidence match
            if best is None or confidence > best.confidence:
                best = result

    return best
