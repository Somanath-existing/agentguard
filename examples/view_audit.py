"""
view_audit.py
-------------
Read and pretty-print AgentGuard SQLite audit logs.

Usage:
    python view_audit.py                          # list all runs
    python view_audit.py real-loop-001            # view a specific run
    python view_audit.py real-loop-001 --steps    # include raw step events
"""
from __future__ import annotations

import sqlite3
import sys
import os
import glob
from datetime import datetime

RUNS_DIR = "./runs"


def find_db(run_id: str) -> str:
    pattern = os.path.join(RUNS_DIR, f"agentguard_{run_id}.db")
    matches = glob.glob(pattern)
    if not matches:
        # try partial match
        all_dbs = glob.glob(os.path.join(RUNS_DIR, "*.db"))
        matches = [d for d in all_dbs if run_id in d]
    if not matches:
        sys.exit(f"No database found for run_id '{run_id}'. Use: python view_audit.py to list all runs.")
    return matches[0]


def list_runs():
    dbs = glob.glob(os.path.join(RUNS_DIR, "*.db"))
    if not dbs:
        print("No audit logs found in ./runs/")
        return
    print(f"\n{'='*55}")
    print(f"  AgentGuard Audit Logs  ({len(dbs)} runs)")
    print(f"{'='*55}")
    print(f"  {'Run ID':<35} {'Size':>8}  {'Modified'}")
    print(f"  {'-'*35} {'-'*8}  {'-'*19}")
    for db in sorted(dbs):
        run_id = os.path.basename(db).replace("agentguard_", "").replace(".db", "")
        size   = os.path.getsize(db)
        mtime  = datetime.fromtimestamp(os.path.getmtime(db)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {run_id:<35} {size:>6}B  {mtime}")
    print(f"\n  Usage: python view_audit.py <run-id>")
    print(f"         python view_audit.py <run-id> --steps\n")


def view_run(run_id: str, show_steps: bool = False):
    db_path = find_db(run_id)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    print(f"\n{'='*65}")
    print(f"  Audit Log: {run_id}")
    print(f"  Database : {db_path}")
    print(f"{'='*65}")

    # ── Step Events (stored as JSON in data column) ────────────────
    step_rows = con.execute(
        "SELECT step_number, data, trust_score, created_at FROM steps ORDER BY step_number"
    ).fetchall()

    if show_steps:
        import json
        print(f"\n  STEP EVENTS ({len(step_rows)} total)")
        print(f"  {'Step':>4}  {'Span':<22} {'Tool':<18} {'In':>6} {'Out':>6} {'Status':<8} {'Trust':>6}")
        print(f"  {'-'*4}  {'-'*22} {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
        for r in step_rows:
            d    = json.loads(r["data"])
            span = (d.get("span_name") or "-")[:22]
            tool = (d.get("tool_name") or "-")[:18]
            tin  = d.get("tokens_in", 0)
            tout = d.get("tokens_out", 0)
            stat = d.get("status", "ok")
            trust = f"{r['trust_score']:.2f}" if r["trust_score"] is not None else "-"
            print(f"  {r['step_number']:>4}  {span:<22} {tool:<18} {tin:>6} {tout:>6} {stat:<8} {trust:>6}")

    # ── Signals (stored in audit_log as signal_fired entries) ──────
    signals = con.execute(
        "SELECT step_number, entry_type, action, triggered_by, detail, created_at "
        "FROM audit_log WHERE entry_type='signal_fired' ORDER BY created_at"
    ).fetchall()

    print(f"\n  SIGNALS FIRED ({len(signals)} total)")
    if signals:
        print(f"  {'Step':>4}  {'Triggered By':<25} {'Action':<12}  Detail")
        print(f"  {'-'*4}  {'-'*25} {'-'*12}  {'-'*40}")
        for s in signals:
            step   = str(s["step_number"] or "-")
            by     = (s["triggered_by"] or "-")[:25]
            action = (s["action"] or "-")[:12]
            detail = (s["detail"] or "")[:60]
            print(f"  {step:>4}  {by:<25} {action:<12}  {detail}")
    else:
        print("  (none — clean run)")

    # ── Full Audit Trail ───────────────────────────────────────────
    audit = con.execute(
        "SELECT entry_type, action, triggered_by, step_number, detail, created_at "
        "FROM audit_log ORDER BY created_at"
    ).fetchall()

    print(f"\n  FULL AUDIT TRAIL ({len(audit)} entries)")
    if audit:
        print(f"  {'Timestamp':<26} {'Type':<22} {'Action':<12} {'Step':>4}  Detail")
        print(f"  {'-'*26} {'-'*22} {'-'*12} {'-'*4}  {'-'*35}")
        for a in audit:
            ts     = (a["created_at"] or "")[:19]
            action = (a["action"] or "-")[:12]
            detail = (a["detail"] or "")[:45]
            step   = str(a["step_number"]) if a["step_number"] is not None else "-"
            print(f"  {ts:<26} {a['entry_type']:<22} {action:<12} {step:>4}  {detail}")
    else:
        print("  (none)")

    # ── Summary ────────────────────────────────────────────────────
    import json
    total_in  = sum(json.loads(r["data"]).get("tokens_in",  0) for r in step_rows)
    total_out = sum(json.loads(r["data"]).get("tokens_out", 0) for r in step_rows)

    print(f"\n  SUMMARY")
    print(f"  Steps recorded : {len(step_rows)}")
    print(f"  Signals fired  : {len(signals)}")
    print(f"  Audit entries  : {len(audit)}")
    print(f"  Tokens in/out  : {total_in} / {total_out}  (total: {total_in+total_out})")
    print(f"{'='*65}\n")

    con.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if not args:
        list_runs()
    else:
        view_run(args[0], show_steps="--steps" in flags)
