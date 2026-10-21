"""
examples/langchain_example.py
------------------------------
Shows how to wrap a LangChain agent with AgentGuard in 2 lines.

Run with:
    OPENAI_API_KEY=sk-... python examples/langchain_example.py

AgentGuard will:
  - Intercept every tool call and LLM completion via OTel
  - Detect loops / info-gain collapse / error streaks in real time
  - Enforce the token budget mid-run
  - Print a terminal report when the run ends

No changes to the agent code are required beyond the two Guard lines.
"""

import os

from agentguard import Guard, AgentGuardAbort


def main():
    # ------------------------------------------------------------------
    # 1. Configure the guard (do this once at application startup)
    # ------------------------------------------------------------------
    guard = Guard(
        budget_tokens=50_000,           # stop the agent if it burns more than this
        on_failure="replan",            # loop → inject replanning prompt
        # alert_webhook="https://hooks.your-app.com/agentguard",  # optional
        # webhook_secret="change-me-in-prod",
        max_retries=3,
        allow_model_downgrade=False,    # set True to auto-swap to haiku on budget pressure
        model="claude-sonnet-4-6",
        store_base_dir="./runs",
        enforce_abort=True,             # NEW: actually kill the agent on abort signals
    )

    # ------------------------------------------------------------------
    # 2. Wrap the agent run — these 2 lines are all you add
    # ------------------------------------------------------------------
    try:
        with guard.watch(run_id="langchain-demo-001") as session:
            # ── Your existing agent code, completely unchanged ──────────────
            try:
                from langchain.agents import AgentType, initialize_agent
                from langchain.tools import DuckDuckGoSearchRun
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                tools = [DuckDuckGoSearchRun()]
                agent = initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=10,
                )
                result = agent.run("What were the top AI research papers published this week?")
                print("\nAgent result:", result)

            except ImportError:
                # LangChain / OpenAI not installed — run a mock to show the guard works
                print("LangChain/OpenAI not installed — running mock agent to demonstrate guard.")
                _mock_agent_run(session)

    except AgentGuardAbort as e:
        # AgentGuard killed the run — this is the ENFORCEMENT mechanism
        print(f"\n🛑 AgentGuard ENFORCED abort: {e}")

    # ------------------------------------------------------------------
    # 3. After the run — inspect what happened
    # ------------------------------------------------------------------
    summary = session.report()
    print("\n" + summary)


def _mock_agent_run(session):
    """
    Simulates span events so you can see AgentGuard fire signals
    without installing LangChain or having an API key.
    """
    from agentguard.models import StepEvent
    from agentguard.signals import SignalEngine

    engine = SignalEngine(run_id=session.run_id)

    # Simulate a loop: same input hash 3 times
    repeated_hash = StepEvent.make_input_hash("search: what is the capital of France")
    for i in range(1, 5):
        event = StepEvent(
            run_id=session.run_id,
            step_number=i,
            span_id=f"mock-span-{i}",
            span_name="tool.web_search",
            tool_name="web_search",
            input_hash=repeated_hash if i <= 3 else StepEvent.make_input_hash(f"different-{i}"),
            output_size=max(0, 1000 - (i * 200)),  # shrinking output
            tokens_in=200,
            tokens_out=300,
        )
        session._store.save_step(event)
        signal = engine.evaluate(event)
        if signal and signal.recommended_action in ("warn", "redirect", "abort"):
            session.on_signal(signal)

    print("Mock run complete — check the report above for signals fired.")


if __name__ == "__main__":
    main()
