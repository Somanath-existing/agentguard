# Contributing to AgentGuard

Thank you for your interest in contributing to **AgentGuard**. We welcome bug reports, framework adapters, recovery heuristics, documentation updates, and pull requests.

---

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/Somanath-existing/agentguard.git
   cd agentguard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install editable package with dev dependencies:**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Run the zero-dependency live smoke test:**
   ```bash
   python examples/demo.py
   ```

---

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ --cov=agentguard --cov-report=term-missing
```

---

## Code Quality & Formatting

We use `ruff` and `mypy` to enforce formatting and typing:

```bash
# Code linting
ruff check src/ tests/

# Type checking
mypy src/
```

---

## Areas We Welcome Contributions (Good First Issues)

We actively encourage contributions in these areas (see [open issues](https://github.com/Somanath-existing/agentguard/issues?q=label%3A%22good+first+issue%22)):

* **Framework Adapters:**
  * Native callback integration for **CrewAI**
  * Interceptor for **AutoGen / AG2** conversational loops
  * Instrumentation hook for **Hugging Face smolagents**
  * LlamaIndex query engine handler
* **Recovery Strategies:**
  * LLM-reflection prompt sanitization strategy
  * Context summarizer recovery strategy using local models
* **Benchmarking & Datasets:**
  * Evaluation runner against public agent failure benchmarks (GAIA, AgentBench)
* **Storage Backends:**
  * Native PostgreSQL / RDS adapter for `CheckpointStore`

---

## Pull Request Guidelines

1. Ensure all **138+ unit and integration tests** pass (`pytest tests/ -v`).
2. Add new unit tests for any added features or bug fixes.
3. Update `CHANGELOG.md` with a summary of your changes.
4. Keep pull requests focused on a single topic.

---

## Security Vulnerability Reporting

If you discover a potential security vulnerability in AgentGuard, please report it privately via [GitHub Security Advisories](https://github.com/Somanath-existing/agentguard/security/advisories) or contact the maintainers directly.
