# Changelog

All notable changes to **AgentGuard** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-08-17

### Added
- **Semantic Taint Tracking (`agentguard.taint`)**: Provenance-aware n-gram hash fingerprinting for multi-step data flow tracking and clean prefix rollback calculation.
- **Predictive Failure Forecasting (`agentguard.signals`)**: Statistical zero-LLM trajectory forecasting using token acceleration, output entropy decay, and Z-score anomaly baselining.
- **Semantic Loop Detection (`agentguard.semantic_loop`)**: Evasion-resistant character n-gram cosine similarity detector with `sentence-transformers` vector fallback.
- **UCB1 Bandit Recovery (`agentguard.memory`)**: Multi-Armed Bandit router learning optimal recovery actions per `(signal_type, agent_type)` tuple.
- **Multi-Agent Topology (`agentguard.topology`)**: Inter-agent DAG boundary scanning, active snippet redaction, and shared multi-agent token budget enforcement.
- **Adversarial Red-Teaming (`agentguard.self_test`)**: Built-in self-testing suite evaluating obfuscation payloads (Unicode confusables, role confusion, persona hijack).
- **Clean `src/` Layout**: Refactored project architecture to standard enterprise `src/agentguard` packaging layout.

### Hardened
- Cryptographic SHA-256 event chaining for tamper-evident audit logs.
- Path traversal protection on SQLite checkpoint/baseline file operations.
- Non-destructive `SpanProcessor` attachment to host OpenTelemetry pipelines.
- Automated multi-version GitHub Actions CI test matrix (Python 3.11, 3.12, 3.13) and Ruff static analysis.

---

## [0.2.0] - 2026-08-01

### Added
- Baseline store using Welford's online variance calculation.
- Token budget guardian with cost projections.
- Basic regex prompt injection defense scanner.

---

## [0.1.0] - 2026-07-25

### Added
- Initial OpenTelemetry span interception engine (`AgentGuardProcessor`).
- `Guard.watch()` context manager lifecycle.
- SQLite state checkpoint storage.
