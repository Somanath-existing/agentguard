# AgentGuard

**Lightweight Runtime Security & Predictive Defense for Autonomous LLM Agents**

[![CI](https://github.com/Somanath-existing/agentguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Somanath-existing/agentguard/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg?style=flat-square)](https://www.python.org/)
[![OpenTelemetry Native](https://img.shields.io/badge/OTel-Native-orange.svg?style=flat-square)](https://opentelemetry.io/)

---

## Executive Summary

**AgentGuard** is an early-stage, in-process runtime guardrail framework designed for autonomous multi-agent LLM systems. Operating directly on standard OpenTelemetry (`OTel`) telemetry spans, AgentGuard monitors agent execution trajectories, prevents token-burn loops, and provides post-mortem causal analysis with **zero additional hosted LLM calls**.

While content guardrails focus on single-prompt safety policies, API gateways manage model traffic, and observability platforms help inspect traces, AgentGuard analyzes multi-step execution trajectories using local statistical signals, semantic similarity, and information-flow taint tracking. It offers an emerging, complementary approach for developers experimenting with in-process runtime monitoring and adaptive recovery strategies in Python-based agent workflows.

```
                  ┌─────────────────────────────────────────┐
                  │          Host Agent Framework           │
                  │   (LangChain / LangGraph / CrewAI)      │
                  └────────────────────┬────────────────────┘
                                       │ Emits Spans
                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AgentGuard In-Process OTel Guard                     │
│                                                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────┐  │
│  │   Semantic Taint   │  │ Predictive Failure │  │  Semantic Loop    │  │
│  │  Tracking (n-gram) │  │ Engine (Z-Score)   │  │ (all-MiniLM-L6-v2)│  │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬─────────┘  │
│            └───────────────────────┼───────────────────────┘            │
│                                    ▼                                    │
│                    ┌───────────────────────────────┐                    │
│                    │  UCB1 Bandit Recovery Engine  │                    │
│                    └───────────────┬───────────────┘                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ Abort / Isolate / Rollback
                                     ▼
                   ┌───────────────────────────────────┐
                   │  Tamper-Evident SHA-256 Audit Log │
                   └───────────────────────────────────┘
```

---

## Key Technical Primitives

### 1. 🛡️ Semantic Taint Tracking (Information Flow Provenance)
*   **Problem:** An untrusted tool output (e.g., scraped web page or database output) contains a hidden prompt injection. The agent reads it and passes infected outputs downstream.
*   **Solution:** Calculates overlapping word n-grams and byte-level hash fingerprints of untrusted tool payloads. Builds a directed acyclic graph (DAG) of step provenance. When an injection is detected at step $N$, AgentGuard computes the exact **clean execution prefix** (defining the safe rollback boundary while keeping clean compute).

### 2. ⚡ Zero-LLM Predictive Failure Forecasting
*   **Problem:** Agents collapse into unrecoverable states 3–5 steps before an explicit crash (e.g., token consumption spikes, output entropy decays, or info-gain collapses).
*   **Solution:** Uses Welford's online variance algorithm to maintain dynamic per-agent behavioral baselines ($Z$-scores). Evaluates output size slope, hash entropy, and token acceleration in real-time to forecast failures 2–3 steps **before** they manifest—without calling an external LLM.

### 3. 🔄 Evasion-Resistant Semantic Loop Detection
*   **Problem:** Adversaries or rogue agent loops bypass exact string hash-matching by adding minor variations (whitespace, punctuation, or paraphrasing). Furthermore, in multi-turn architectures, message buffers continually accumulate prior turns and tool outputs; comparing full historical buffers causes >95% false-positive similarity.
*   **Solution:** Uses dense vector embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) with an offline character n-gram TF-IDF fallback. Includes turn isolation (`extract_active_prompt`) to strip cumulative conversation history so similarity checks compare the *actual active turn* rather than the growing chat buffer. Detects semantic stagnation and velocity collapses across sliding windows without false alarms.

### 4. 🧠 UCB1 Multi-Armed Bandit Recovery Router
*   **Problem:** Static recovery strategies (always retrying or always replanning) fail in dynamic production environments.
*   **Solution:** Implements Upper Confidence Bound (UCB1) reinforcement learning to evaluate historical recovery success per `(signal_type, agent_type)` tuple. Dynamically selects optimal recovery policies to maximize session resolution probability.

### 5. 🔒 Cryptographically Chained Forensic Audit Trail
*   **Problem:** Standard logging can be tampered with or truncated during security incidents.
*   **Solution:** Maintains an immutable audit log where each event record is cryptographically signed and chained with the SHA-256 digest of the preceding entry ($\text{Hash}_t = H(\text{Event}_t \parallel \text{Hash}_{t-1})$). Designed for zero-knowledge privacy: raw prompts are **never stored**.

---

## How It Works (OTel Span Processing Flow)

AgentGuard hooks into the host application's OpenTelemetry pipeline:

1. **Span Capture:** The `AgentGuardProcessor` intercepts active telemetry spans emitted by auto-instrumented libraries (LangChain, OpenAI, Anthropic).
2. **Analysis Pipeline:** As spans end, AgentGuard computes input hashes, records token usage, runs regex injection scans, and feeds metrics to the `SignalEngine`.
3. **Policy Enforcement:** If a failure signal exceeds threshold confidence:
   - If `enforce_abort` is active, it raises `AgentGuardAbort` to halt the run immediately.
   - If a recovery strategy is found, it logs a recovery action and prepares rollback instructions.
4. **Clean Detach:** The context manager ensures the span processor is cleanly detached from the global tracer provider when execution finishes, preventing memory leaks.

---

## Detection & Enforcement Matrix

| Signal Type | Detection Vector | Default Action | Recovery Options |
| :--- | :--- | :--- | :--- |
| **`loop`** | Exact matches + n-gram Cosine Similarity | `replan` | Re-plan prompt, custom retry, or abort |
| **`drift`** | Unicode confusable / Prompt injection regex | `abort` | Raise `AgentGuardAbort`, quarantine run |
| **`budget_pressure`** | Tokens consumed > Warning Threshold | `warn` / `downgrade` | Shift to cheaper model class, notify operator |
| **`info_collapse`** | Progressive entropy decay of output lengths | `escalate` | Alert webhook, redirect to human operator |
| **`tool_error`** | Intercepted tool span with error status code | `retry` | Exponential backoff up to `max_retries` |
| **`unauthorized_tool`**| Deny/Allowlist mismatch or bad tool arguments | `abort` | Stop execution, log alert event |

---

## Supported Integrations

AgentGuard integrates out-of-the-box with any python package instrumented with standard OpenTelemetry GenAI semantics:
* **OpenAI SDK** (via `opentelemetry-instrumentation-openai`)
* **Anthropic SDK** (via `opentelemetry-instrumentation-anthropic`)
* **LangChain & LangGraph** (via `opentelemetry-instrumentation-langchain`)
* **CrewAI & AutoGen** (via standard SDK OTel exports)

---

## Quickstart

### ⚡ Zero-Dependency Live Smoke Test

You can test AgentGuard immediately in your terminal without any OpenAI or external API keys:

```bash
git clone https://github.com/Somanath-existing/agentguard.git
cd agentguard
pip install -e ".[dev,all]"
python examples/demo.py
```
This simulates real-time OpenTelemetry LLM/tool spans, demonstrating clean execution, prompt injection interception, and root-cause post-mortem generation in rich terminal UI.

### Installation

```bash
# Install directly from GitHub (latest v0.3.0)
pip install git+https://github.com/Somanath-existing/agentguard.git

# With full OpenTelemetry auto-instrumentation
pip install "agentguard[all] @ git+https://github.com/Somanath-existing/agentguard.git"

# Or install from source in editable mode
pip install -e ".[dev,all]"
```

### Minimal Integration

Wrap your existing agent invocation inside `guard.watch()`. No architectural refactoring is required:

```python
from agentguard import Guard, AgentGuardAbort

# Initialize production guardrail with token budget & agent profile
guard = Guard(
    budget_tokens=100_000,
    agent_type="financial_analyzer",
    enforce_abort=True
)

try:
    with guard.watch(run_id="prod-session-9842") as session:
        # Your unmodified agent code executes here
        response = agent.run("Analyze Q3 earnings report")

except AgentGuardAbort as e:
    print(f"🚨 Security Enforcer Triggered: {e.reason}")
    print(f"Root Cause Analysis: {session.explain_failure()}")

# Print structured session audit report
print(session.report())
```

---

## Production Integration

### 1. Production State Persistence (Redis & SQLite WAL)
By default, AgentGuard operates in SQLite WAL mode. For horizontally scaled microservices (Kubernetes / serverless workers), configure the Redis distributed backend by providing a pre-configured `redis.Redis` instance:

```python
import redis
from agentguard import Guard

# Connect to your enterprise Redis cluster
redis_client = redis.Redis(host='redis-cluster.internal', port=6379, db=0)

guard = Guard(
    agent_type="k8s_worker",
    store_backend="redis",
    redis_client=redis_client,
    redis_ttl_seconds=86400  # Automatically expire state after 24h
)
```

### 2. Multi-Agent DAG Topology Protection
Ensure total budget safety across complex multi-agent hierarchies using a shared topology controller:

```python
from agentguard import Guard, TopologyTracker

# Central shared budget controller
topology = TopologyTracker(total_budget=500_000)

orchestrator = Guard(agent_type="orchestrator", topology=topology)
sub_agent    = Guard(agent_type="researcher", topology=topology, parent_run_id="orch-001")

with orchestrator.watch("orch-001"):
    # Sub-agent execution shares budget and checks hierarchy loops
    with sub_agent.watch("sub-001"):
        sub_agent_execution()
```

---

## Security & Privacy Architecture

*   **Privacy-Preserving Forensic Logging:** AgentGuard avoids storing full user prompts or completion streams on disk. State matching and taint tracking rely on SHA-256 cryptographic digests, with only bounded diagnostic excerpts (up to 80 chars) captured in forensic audit trails upon security rule violations.
*   **Data Minimization Compliance:** Aligns with enterprise data residency standards by design. Only execution metadata (latencies, token counts, error classes, status codes) is collected during standard operations.
*   **Idempotent Integration:** Attaches to existing `TracerProvider` pipelines without overwriting host telemetry configuration.

---

## 🗺️ Roadmap & Contributing

We actively welcome community contributions. Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

* [ ] **CrewAI & AutoGen Adapters:** Native callback and listener wrappers for seamless drop-in integration.
* [ ] **PostgreSQL / AWS RDS Backend:** Durable SQL audit log storage alongside SQLite and Redis.
* [ ] **LLM-Reflection Recovery:** Secondary prompt sanitization via lightweight local models.
* [ ] **Agent Benchmark Suite:** Standardized runner against GAIA and AgentBench failure datasets.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local dev setup, coding guidelines, and submitting pull requests.

---

## License

AgentGuard is licensed under the [Apache 2.0 License](LICENSE).
