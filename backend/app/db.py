"""Read-only access to the active analytics database. The raw dataset is never opened by the API."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ml import config
from ml.pii import redact


class DatabaseUnavailable(RuntimeError):
    pass


def db_path() -> Path:
    # Prefer explicitly activated upload batch, then ANALYTICS_DB env, then default.
    try:
        from backend.app import batches as batch_reg

        active = batch_reg.get_active_db_path()
        if active and active.exists():
            return active
    except Exception:
        pass
    p = Path(os.environ.get("ANALYTICS_DB", str(config.ANALYTICS_DB)))
    return p if p.is_absolute() else config.REPO_ROOT / p


@contextmanager
def connect():
    p = db_path()
    if not p.exists():
        raise DatabaseUnavailable("analytics database not found")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def report(con: sqlite3.Connection, key: str) -> dict | None:
    row = con.execute("SELECT data FROM reports WHERE key=?", (key,)).fetchone()
    return json.loads(row["data"]) if row else None


def meta(con: sqlite3.Connection) -> dict:
    return {r["key"]: json.loads(r["value"]) for r in con.execute("SELECT key, value FROM meta")}


def safe_text(text: str) -> str:
    """Defence in depth: text is already redacted in the DB; re-running the (idempotent) redactor guarantees
    that nothing matching a PII rule can leave through the API even if the DB were produced incorrectly."""
    return redact(text).text


def review_dict(row: sqlite3.Row, theme_names: dict | None = None) -> dict:
    d = {
        "review_id": row["review_id"],
        "created_at": row["created_at"],
        "rating": row["rating"],
        "platform": row["platform"],
        "app_version": row["app_version"],
        "text": safe_text(row["text_redacted"]),
        "sentiment": row["sentiment"],
        "confidence": row["confidence"],
        "theme_id": row["theme_id"],
        "theme_assignment": row["theme_assignment"],
        "pii_redactions": row["pii_redaction_count"],
        "source": row["source"],
    }
    if theme_names is not None:
        d["theme_name"] = theme_names.get(row["theme_id"])
    return d


def theme_names(con: sqlite3.Connection) -> dict:
    return {r["theme_id"]: r["name"] for r in con.execute("SELECT theme_id, name FROM themes")}
