"""Uploaded batch registry: each upload gets its own analytics DB."""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ml import config

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_active_batch_id: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index() -> dict:
    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not config.BATCHES_INDEX.exists():
        return {"batches": [], "active_batch_id": None}
    return json.loads(config.BATCHES_INDEX.read_text(encoding="utf-8"))


def save_index(idx: dict) -> None:
    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    config.BATCHES_INDEX.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def list_batches() -> list[dict]:
    with _lock:
        idx = load_index()
        active = idx.get("active_batch_id") or _active_batch_id
        out = []
        for b in idx.get("batches", []):
            out.append({**b, "active": b["batch_id"] == active})
        return out


def get_active_db_path() -> Path | None:
    with _lock:
        idx = load_index()
        active = _active_batch_id or idx.get("active_batch_id")
        if not active:
            # fall back to default analytics.db if present
            if config.ANALYTICS_DB.exists():
                return config.ANALYTICS_DB
            return None
        for b in idx.get("batches", []):
            if b["batch_id"] == active:
                p = Path(b["db_path"])
                return p if p.is_absolute() else config.REPO_ROOT / p
        return None


def activate(batch_id: str) -> dict:
    global _active_batch_id
    with _lock:
        idx = load_index()
        hit = next((b for b in idx.get("batches", []) if b["batch_id"] == batch_id), None)
        if not hit:
            raise KeyError(batch_id)
        idx["active_batch_id"] = batch_id
        _active_batch_id = batch_id
        save_index(idx)
        return {**hit, "active": True}


def job_status(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)


def start_upload(filename: str, content: bytes) -> dict:
    """Save CSV and run pipeline in a background thread. Returns job stub."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:80] or "upload.csv"
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    batch_id = uuid.uuid4().hex[:12]
    job_id = uuid.uuid4().hex[:12]
    upload_dir = config.UPLOADS_DIR / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = upload_dir / safe
    csv_path.write_bytes(content)
    db_path = config.BATCHES_DIR / f"{batch_id}.db"
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "running",
            "filename": safe,
            "started_at_utc": _now(),
            "error": None,
            "db_path": str(db_path),
        }

    def _run():
        global _active_batch_id
        try:
            from ml.pipeline import run

            res = run(str(csv_path), None, db_path, None, True)
            entry = {
                "batch_id": batch_id,
                "name": safe,
                "filename": safe,
                "db_path": str(db_path),
                "created_at_utc": _now(),
                "reviews": (res.get("overview") or {}).get("total_reviews"),
                "dates_available": (res.get("overview") or {}).get("dates_available"),
                "n_themes": (res.get("overview") or {}).get("n_themes"),
            }
            with _lock:
                idx = load_index()
                idx.setdefault("batches", []).insert(0, entry)
                idx["active_batch_id"] = batch_id
                _active_batch_id = batch_id
                save_index(idx)
                _jobs[job_id].update(status="done", finished_at_utc=_now(), batch=entry)
        except Exception as e:
            with _lock:
                _jobs[job_id].update(status="error", error=str(e)[:500], finished_at_utc=_now())

    threading.Thread(target=_run, daemon=True).start()
    with _lock:
        return dict(_jobs[job_id])
