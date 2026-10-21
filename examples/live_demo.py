"""
live_demo.py
------------
Live AgentGuard integration with Groq (free LLM API).
Run: python live_demo.py

Get a free Groq API key at: https://console.groq.com
Set it: $env:GROQ_API_KEY = "gsk_..."
"""
import os
import uuid
from groq import Groq
from agentguard import Guard, AgentGuardAbort

# ── 1. Initialize guard and client ─────────────────────────────────────────────

guard = Guard(
    budget_tokens=5_000,        # ~5K tokens max per run (very low for demo)
    agent_type="groq_assistant",
    enforce_abort=True,
)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── 2. Reusable function — no code changes between different questions ──────────

def ask(prompt: str) -> str:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    print(f"\n[{run_id}] Prompt: {prompt[:60]}...")
    try:
        with guard.watch(run_id=run_id) as session:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            reply = response.choices[0].message.content
            print(f"[{run_id}] Response: {reply[:120]}...")
            print(session.report())
            return reply

    except AgentGuardAbort as e:
        print(f"\n[!!!] AgentGuard killed the run: {e.reason}")
        cause = session.explain_failure()
        if "error" not in cause:
            print(f"      Root cause at step {cause.get('root_cause_step')}: {cause.get('signal_type')}")
        return "[ABORTED]"

# ── 3. Run different questions — guard protects all of them ─────────────────────

if __name__ == "__main__":
    # Normal questions — AgentGuard monitors silently, no intervention
    ask("What is quantum computing? Explain in 2 sentences.")
    ask("Write a Python function to reverse a string.")
    ask("What are the top 3 uses of transformers in NLP?")
