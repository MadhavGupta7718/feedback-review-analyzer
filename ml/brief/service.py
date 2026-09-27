"""Chooses how a product brief is produced and always reports which path was used.

Paths
  qwen_live         Qwen2.5-3B generated it now on this machine (requires installed model + suitable hardware)
  qwen_precomputed  a Qwen brief generated offline by scripts/generate_brief.py and stored in the DB
  template          deterministic template (always available; the fallback)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from ml.brief.facts import build_facts
from ml.brief.template import template_brief

_lock = threading.Lock()
_writer = None
_writer_error: str | None = None


def qwen_status() -> dict:
    from ml import config
    from ml.models import registry

    mode = os.environ.get("BRIEF_MODE", "auto")
    installed = registry.locate(config.QWEN_MODEL).installed
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    return {"brief_mode": mode, "qwen_installed": installed, "cuda_available": cuda,
            "qwen_loaded": _writer is not None, "last_error": _writer_error,
            "live_generation_possible": mode != "template" and installed and cuda}


def _get_writer():
    global _writer, _writer_error
    if _writer is None and _writer_error is None:
        try:
            from ml.brief.qwen_writer import QwenBriefWriter

            _writer = QwenBriefWriter()
        except Exception as exc:  # noqa: BLE001
            _writer_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return _writer


def precomputed(db_path: Path) -> dict | None:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT data FROM reports WHERE key='qwen_brief'").fetchone()
        return json.loads(row[0]) if row else None
    finally:
        con.close()


def generate_brief(db_path: Path, engine: str = "auto") -> dict:
    facts = build_facts(db_path)
    reasons: list[str] = []
    if engine in ("auto", "qwen"):
        st = qwen_status()
        if st["live_generation_possible"]:
            with _lock:
                writer = _get_writer()
                if writer is not None:
                    res = writer.write(facts)
                    if res["ok"]:
                        return {**res["brief"], "generation_path": "qwen_live", "validation": res["validation"]}
                    reasons.append("Qwen output failed validation: " + "; ".join(res["validation"]["errors"][:3]))
                else:
                    reasons.append(f"Qwen could not be loaded: {_writer_error}")
        else:
            reasons.append("live Qwen unavailable (" + ", ".join(
                k for k, v in (("BRIEF_MODE=template", st["brief_mode"] == "template"), ("model not installed", not st["qwen_installed"]),
                               ("no CUDA GPU", not st["cuda_available"])) if v) + ")")
        pre = precomputed(db_path)
        if pre is not None:
            return {**pre, "generation_path": "qwen_precomputed", "fallback_reasons": reasons}
        reasons.append("no precomputed Qwen brief in the database")
    brief = template_brief(facts)
    if reasons:
        brief["fallback_reasons"] = reasons
    return brief
