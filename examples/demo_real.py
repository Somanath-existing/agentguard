# -*- coding: utf-8 -*-
"""
demo_real.py
------------
Real-world AgentGuard demo using the Groq API (free, fast, no LangChain needed).

Set your key first:
    $env:GROQ_API_KEY = "gsk_..."

Then run:
    python demo_real.py

This exercises 3 scenarios:
  1. Normal agent run  → guard stays quiet, prints clean report
  2. Loop simulation   → guard fires WARN then ABORT
  3. Budget breach     → guard fires budget warning signal
"""
from __future__ import annotations

import os
import sys

from groq import Groq

from agentguard import AgentGuardAbort, Guard
from agentguard.models import StepEvent
from agentguard.signals import SignalEngine

# -- helpers ----------------------------------------------------------------

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    sys.exit(
        "\nERROR: GROQ_API_KEY not set.\n"
        "    Run:  $env:GROQ_API_KEY = 'gsk_...'\n"
        "    Get a free key at https://console.groq.com\n"
    )

client = Groq(api_key=GROQ_KEY)


def ask_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> tuple[str, int, int]:
    """Call Groq and return (reply_text, tokens_in, tokens_out)."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    msg = response.choices[0].message.content or ""
    usage = response.usage
    return msg, usage.prompt_tokens, usage.completion_tokens


def _inject_step(session, engine, step: int, input_text: str, output_size: int,
                 tokens_in: int, tokens_out: int, tool: str = "llm.chat") -> None:
    """Manually record a StepEvent (simulates what OTel would do automatically)."""
    ev = StepEvent(
        run_id=session.run_id,
        step_number=step,
        span_id=f"span-{step}",
        span_name=tool,
        tool_name=tool,
        input_hash=StepEvent.make_input_hash(input_text),
        output_size=output_size,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    session._store.save_step(ev)
    session._cost.record(ev)

    signal = engine.evaluate(ev)
    if signal and signal.recommended_action in ("warn", "redirect", "abort"):
        print(f"  [SIGNAL] {signal.signal_type} -> {signal.recommended_action}")
        session.on_signal(signal)

    budget_sig = session._cost.should_warn_signal(session.run_id, step)
    if budget_sig:
        print(f"  [BUDGET] {budget_sig.signal_type}")
        session.on_signal(budget_sig)

    if session._cost.is_over_budget():
        session.on_budget_exceeded(session._cost.remaining())


# -- Scenario 1: Normal real call --------------------------------------------

def scenario_normal():
    print("\n" + "="*60)
    print("SCENARIO 1: Normal agent run (real Groq API call)")
    print("="*60)

    guard = Guard(
        budget_tokens=10_000,
        on_failure="replan",
        store_base_dir="./runs",
        enforce_abort=False,   # observe only
    )

    try:
        with guard.watch(run_id="real-normal-001") as session:
            engine = SignalEngine(run_id=session.run_id)

            questions = [
                "What is 2 + 2? Answer in one sentence.",
                "Name one planet in our solar system.",
                "What colour is the sky on a clear day?",
            ]

            for step, q in enumerate(questions, start=1):
                print(f"\n  Step {step}: {q}")
                reply, tin, tout = ask_groq(q)
                print(f"  -> {reply.strip()[:120]}")
                _inject_step(session, engine, step, q, len(reply), tin, tout)

    except AgentGuardAbort as e:
        print(f"\n[ABORT ENFORCED] {e}")

    print("\n" + session.report())


# -- Scenario 2: Loop detection -----------------------------------------------

def scenario_loop():
    print("\n" + "="*60)
    print("SCENARIO 2: Loop detection (same query repeated)")
    print("="*60)

    guard = Guard(
        budget_tokens=10_000,
        on_failure="replan",
        store_base_dir="./runs",
        enforce_abort=True,
    )

    try:
        with guard.watch(run_id="real-loop-001") as session:
            engine = SignalEngine(run_id=session.run_id)

            # First call is real, then we simulate the agent looping
            stuck_query = "Search the web for the latest news about AI."
            print(f"\n  Repeating query 4x to trigger loop detector:")
            print(f"  Query: \"{stuck_query}\"")

            for step in range(1, 5):
                print(f"\n  Step {step}...")
                # Only call Groq once — rest are injected as loop
                if step == 1:
                    reply, tin, tout = ask_groq(stuck_query)
                    print(f"  -> {reply.strip()[:100]}")
                else:
                    # Simulate agent re-issuing the identical request (looping)
                    tin, tout = 80, 120

                _inject_step(session, engine, step,
                             stuck_query,        # same hash every time -> loop!
                             1000 - step * 150,  # shrinking output -> info-gain collapse
                             tin, tout)

    except AgentGuardAbort as e:
        print(f"\n[ABORT ENFORCED] {e}")

    print("\n" + session.report())


# -- Scenario 3: Budget enforcement ------------------------------------------

def scenario_budget():
    print("\n" + "="*60)
    print("SCENARIO 3: Token budget enforcement")
    print("="*60)

    guard = Guard(
        budget_tokens=500,     # very tight budget
        on_failure="abort",
        store_base_dir="./runs",
        enforce_abort=True,
        warn_pct=0.6,          # warn at 60% usage
    )

    try:
        with guard.watch(run_id="real-budget-001") as session:
            engine = SignalEngine(run_id=session.run_id)
            print("\n  Budget: 500 tokens. Making calls until we breach it...")

            for step in range(1, 10):
                q = f"Write a detailed paragraph about topic number {step}."
                print(f"\n  Step {step}: (tokens used so far: {session._cost._tokens_used})")
                reply, tin, tout = ask_groq(q)
                print(f"  -> tokens this call: in={tin} out={tout}")
                _inject_step(session, engine, step, q, len(reply), tin, tout)

    except AgentGuardAbort as e:
        print(f"\n[ABORT ENFORCED] {e}")

    print("\n" + session.report())


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    scenario_normal()
    scenario_loop()
    scenario_budget()

    print("\nDone. All scenarios complete. Check ./runs/ for SQLite audit logs.")
