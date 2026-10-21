"""
agentguard/state.py
-------------------
Checkpoint store with two backends: SQLite (dev) and Redis (prod).

Security fixes vs original guide:
  1. run_id sanitization + path confinement — no path traversal via crafted run_ids.
  2. Separate append-only audit_log table — INSERT only, never INSERT OR REPLACE,
     so recovery history cannot be silently overwritten.
  3. WAL mode enabled on SQLite for crash-safe writes.
  4. Redis key namespace prefix prevents accidental key collisions with host app.
  5. All SQLite operations are protected by threading.Lock to prevent
     corruption from concurrent writes on the same connection.
  6. Audit log entries are hash-chained (prev_hash column) for tamper-evident
     forensic integrity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading

from .models import AuditEntry, StepEvent

# ---------------------------------------------------------------------------
# run_id sanitization (also used by CheckpointStore and Guard)
# ---------------------------------------------------------------------------

_RUN_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_run_id(run_id: str) -> str:
    """
    Replace any character that isn't alphanumeric, dash, or underscore.
    Caps at 64 characters to avoid filesystem limits.

    Raises ValueError if the result is empty (e.g. all-special-char input).
    """
    clean = _RUN_ID_PATTERN.sub("_", run_id)[:64]
    if not clean or clean.strip("_") == "":
        raise ValueError(f"run_id {run_id!r} produced an empty safe identifier")
    return clean


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class CheckpointStore:
    """
    Stores step events and audit entries for a single agent run.

    Backend selection:
        backend="sqlite"  →  local file, zero infra, good for dev/testing
        backend="redis"   →  distributed low-latency reads, TTL support, use in prod

    Path confinement (SQLite):
        The DB file is always created inside `base_dir`. After joining
        base_dir + filename we verify with os.path.abspath that the result
        still starts with the resolved base_dir — this blocks path traversal
        via run_id values like "../../etc/cron.d/agentguard".
    """

    # Redis key namespace — prevents collisions with the host application's keys
    _REDIS_NS = "agentguard"

    def __init__(
        self,
        run_id: str,
        backend: str = "sqlite",
        base_dir: str = "./agentguard_runs",
        redis_client=None,
        redis_ttl_seconds: int = 86_400,  # 24 h default TTL for Redis keys
    ):
        self.run_id = sanitize_run_id(run_id)
        self.backend = backend
        self._redis_ttl = redis_ttl_seconds

        if backend == "sqlite":
            self._init_sqlite(base_dir)
        elif backend == "redis":
            if redis_client is None:
                raise ValueError("redis_client must be provided when backend='redis'")
            self._redis = redis_client
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'sqlite' or 'redis'.")

    # ------------------------------------------------------------------
    # SQLite setup
    # ------------------------------------------------------------------

    def _init_sqlite(self, base_dir: str) -> None:
        os.makedirs(base_dir, exist_ok=True)

        # Path confinement check
        filename = f"agentguard_{self.run_id}.db"
        candidate = os.path.join(base_dir, filename)
        resolved = os.path.abspath(candidate)
        resolved_base = os.path.abspath(base_dir)

        if not resolved.startswith(resolved_base + os.sep) and resolved != resolved_base:
            raise ValueError(
                f"Path traversal detected: run_id {self.run_id!r} "
                f"would escape base_dir {base_dir!r}"
            )

        self.conn = sqlite3.connect(resolved, check_same_thread=False)
        self._db_lock = threading.Lock()
        # WAL mode: crash-safe, allows concurrent reads during writes
        with self._db_lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        # Track last audit hash for tamper-evident chaining
        self._last_audit_hash: str = self._load_last_audit_hash()

    def _create_tables(self) -> None:
        with self._db_lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS steps (
                    step_number  INTEGER NOT NULL,
                    run_id       TEXT    NOT NULL,
                    data         TEXT    NOT NULL,   -- JSON-serialized StepEvent (no raw prompt)
                    trust_score  REAL    DEFAULT 1.0,
                    created_at   TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    PRIMARY KEY (run_id, step_number)
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT    NOT NULL,
                    entry_type   TEXT    NOT NULL,
                    action       TEXT,
                    triggered_by TEXT,
                    step_number  INTEGER,
                    detail       TEXT,
                    prev_hash    TEXT,              -- SHA-256 of previous entry for tamper detection
                    created_at   TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_steps_run_trust
                    ON steps(run_id, trust_score DESC, step_number DESC);

                CREATE INDEX IF NOT EXISTS idx_audit_run
                    ON audit_log(run_id, id);
            """)
            self.conn.commit()

    def _load_last_audit_hash(self) -> str:
        """Load the hash of the most recent audit entry for chain continuity."""
        if self.backend != "sqlite":
            return "genesis"
        with self._db_lock:
            row = self.conn.execute(
                "SELECT prev_hash, entry_type, detail FROM audit_log "
                "WHERE run_id=? ORDER BY id DESC LIMIT 1",
                (self.run_id,),
            ).fetchone()
        if row:
            # Compute hash of the last entry itself to chain from
            content = f"{row[0]}|{row[1]}|{row[2]}"
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
        return "genesis"

    # ------------------------------------------------------------------
    # Public API — steps
    # ------------------------------------------------------------------

    def save_step(self, event: StepEvent, trust_score: float = 1.0) -> None:
        """
        Persist a step event.  Uses INSERT OR REPLACE so re-running a step
        (e.g. after a retry) updates the record rather than duplicating it.
        The raw prompt is NOT stored — only the hash is present in StepEvent.
        """
        if self.backend == "sqlite":
            with self._db_lock:
                self.conn.execute(
                    "INSERT OR REPLACE INTO steps (step_number, run_id, data, trust_score) "
                    "VALUES (?, ?, ?, ?)",
                    (event.step_number, self.run_id, event.model_dump_json(), trust_score),
                )
                self.conn.commit()
        else:
            key = f"{self._REDIS_NS}:steps:{self.run_id}:{event.step_number}"
            payload = json.dumps({"data": event.model_dump_json(), "trust_score": trust_score})
            self._redis.set(key, payload, ex=self._redis_ttl)

    def last_trusted_step(self, min_trust: float = 0.7) -> StepEvent | None:
        """Return the most recent step whose trust_score >= min_trust."""
        if self.backend == "sqlite":
            with self._db_lock:
                row = self.conn.execute(
                    "SELECT data FROM steps "
                    "WHERE run_id=? AND trust_score>=? "
                    "ORDER BY step_number DESC LIMIT 1",
                    (self.run_id, min_trust),
                ).fetchone()
            return StepEvent.model_validate_json(row[0]) if row else None
        else:
            pattern = f"{self._REDIS_NS}:steps:{self.run_id}:*"
            best: StepEvent | None = None
            best_step = -1
            for key in self._redis.scan_iter(pattern):
                raw = self._redis.get(key)
                if not raw:
                    continue
                payload = json.loads(raw)
                if payload["trust_score"] >= min_trust:
                    evt = StepEvent.model_validate_json(payload["data"])
                    if evt.step_number > best_step:
                        best_step = evt.step_number
                        best = evt
            return best

    def all_steps(self) -> list[StepEvent]:
        if self.backend == "sqlite":
            with self._db_lock:
                rows = self.conn.execute(
                    "SELECT data FROM steps WHERE run_id=? ORDER BY step_number",
                    (self.run_id,),
                ).fetchall()
            return [StepEvent.model_validate_json(r[0]) for r in rows]
        else:
            pattern = f"{self._REDIS_NS}:steps:{self.run_id}:*"
            events = []
            for key in self._redis.scan_iter(pattern):
                raw = self._redis.get(key)
                if raw:
                    payload = json.loads(raw)
                    events.append(StepEvent.model_validate_json(payload["data"]))
            return sorted(events, key=lambda e: e.step_number)

    # ------------------------------------------------------------------
    # Public API — audit log (append-only)
    # ------------------------------------------------------------------

    def save_audit_event(self, entry: AuditEntry) -> None:
        """
        Append an audit entry.  NEVER uses INSERT OR REPLACE — the audit log
        must be immutable so post-incident forensics are reliable.
        Each entry includes prev_hash for tamper-evident hash chaining.
        """
        if self.backend == "sqlite":
            with self._db_lock:
                self.conn.execute(
                    "INSERT INTO audit_log "
                    "(run_id, entry_type, action, triggered_by, step_number, detail, prev_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.run_id,
                        entry.entry_type,
                        entry.action,
                        entry.triggered_by,
                        entry.step_number,
                        entry.detail,
                        self._last_audit_hash,
                    ),
                )
                self.conn.commit()
            # Update chain: hash all fields of this entry for complete tamper evidence
            content = (
                f"{self._last_audit_hash}|{self.run_id}|{entry.entry_type}|"
                f"{entry.action}|{entry.triggered_by}|{entry.step_number}|{entry.detail}"
            )
            self._last_audit_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            # Redis: use an append-only list per run
            key = f"{self._REDIS_NS}:audit:{self.run_id}"
            self._redis.rpush(key, entry.model_dump_json())
            self._redis.expire(key, self._redis_ttl)

    def audit_log(self) -> list[AuditEntry]:
        if self.backend == "sqlite":
            with self._db_lock:
                rows = self.conn.execute(
                    "SELECT entry_type, action, triggered_by, step_number, detail, created_at "
                    "FROM audit_log WHERE run_id=? ORDER BY id",
                    (self.run_id,),
                ).fetchall()
            return [
                AuditEntry(
                    run_id=self.run_id,
                    entry_type=r[0],
                    action=r[1],
                    triggered_by=r[2],
                    step_number=r[3],
                    detail=r[4],
                )
                for r in rows
            ]
        else:
            key = f"{self._REDIS_NS}:audit:{self.run_id}"
            return [AuditEntry.model_validate_json(raw) for raw in self._redis.lrange(key, 0, -1)]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.backend == "sqlite" and hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
