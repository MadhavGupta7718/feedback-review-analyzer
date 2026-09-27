"""Phase 0 environment check. Writes artifacts/reports/env_check.json with measured values."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402  (sets HF cache env vars)
from ml.hardware import detect_hardware  # noqa: E402

PACKAGES = [
    "torch", "transformers", "sentence-transformers", "huggingface-hub", "accelerate",
    "bitsandbytes", "hdbscan", "scikit-learn", "numpy", "pandas", "fastapi", "uvicorn",
    "pydantic", "scipy", "psutil", "pytest",
]


def _cmd(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30, shell=False)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def _disk(path: str) -> dict:
    try:
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1024**3, 1), "free_gb": round(u.free / 1024**3, 1)}
    except OSError:
        return {"error": "not found"}


def main() -> int:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None

    hw = detect_hardware()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "node": _cmd(["node", "--version"]),
        "git": _cmd(["git", "--version"]),
        "disk": {"C:": _disk("C:\\"), "D:": _disk("D:\\")},
        "hf_env": {k: os.environ.get(k) for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]},
        "hf_cache_exists": config.HF_HUB_CACHE.exists(),
        "dataset": {
            "path": str(config.S140_PATH),
            "exists": config.S140_PATH.exists(),
            "size_bytes": config.S140_PATH.stat().st_size if config.S140_PATH.exists() else None,
        },
        "hardware": hw.to_dict(),
        "packages": versions,
    }
    out = config.ARTIFACTS_DIR / "reports" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    problems = []
    if not report["dataset"]["exists"]:
        problems.append("dataset missing")
    if not str(report["hf_env"]["HF_HOME"]).upper().startswith("D:"):
        problems.append("HF_HOME not on D:")
    if versions["torch"] is None:
        problems.append("torch not installed")
    print("\nPHASE 0:", "PASS" if not problems else f"FAILED -> {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
