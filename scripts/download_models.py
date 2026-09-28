"""One-time model download (the ONLY code path in this project that touches the network for models).

Downloads anonymously (public models, no Hugging Face token) into D:\\huggingface, verifies files,
then runs scripts/verify_models.py in a subprocess with networking disabled to prove the models
load and infer offline. Exits non-zero if any required model cannot be downloaded or verified.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402  (pins HF cache to D:)
from ml.models import registry  # noqa: E402

REQUIRED_LIBS = ["torch", "transformers", "sentence-transformers", "huggingface_hub", "accelerate", "bitsandbytes"]
# Conservative space needed per model (GB) before download, including headroom.
SPACE_NEEDED_GB = {config.SENTIMENT_MODEL: 1.0, config.EMBEDDING_MODEL: 0.5, config.QWEN_MODEL: 7.5, config.NER_MODEL: 1.0,
                   config.COMPARISON_MODELS[0]: 2.0, config.COMPARISON_MODELS[1]: 0.5}
BASE_PATTERNS = ["*.json", "*.txt", "*.model", "tokenizer*", "vocab*", "merges.txt", "1_Pooling/*", "*.md"]


def gb(n: float) -> float:
    return round(n / 1024**3, 2)


def choose_patterns(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, token=False)
    weights = [f for f in files if f.endswith(".safetensors") and "/" not in f]
    if not weights:
        weights = [f for f in files if f in ("pytorch_model.bin",) or (f.startswith("pytorch_model-") and f.endswith(".bin"))]
    return BASE_PATTERNS + weights


def download(repo_id: str) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    rec: dict = {"repo_id": repo_id, "status": "FAILED"}
    t0 = time.perf_counter()
    try:
        info = HfApi().model_info(repo_id, token=False)
        rec["remote_sha"] = info.sha
        rec["gated"] = bool(getattr(info, "gated", False))
        patterns = choose_patterns(repo_id)
        rec["allow_patterns"] = patterns
        path = snapshot_download(repo_id, allow_patterns=patterns, token=False, cache_dir=str(config.HF_HUB_CACHE))
        rec["download_seconds"] = round(time.perf_counter() - t0, 1)
        local = registry.locate(repo_id)
        rec["local_path"] = str(path)
        rec["revision"] = local.revision
        files = sorted(p.relative_to(path).as_posix() for p in Path(path).rglob("*") if p.is_file())
        rec["files"] = files
        rec["disk_bytes"] = registry.dir_size_bytes(Path(path))
        rec["disk_gb"] = gb(rec["disk_bytes"])
        has_cfg = "config.json" in files
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
        has_tok = any(f.startswith(("tokenizer", "vocab")) for f in files)
        rec["file_checks"] = {"config": has_cfg, "weights": has_weights, "tokenizer": has_tok}
        rec["status"] = "DOWNLOADED" if has_cfg and has_weights and has_tok else "INCOMPLETE"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-optional", action="store_true", help="also download dslim/bert-base-NER")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--comparison", action="store_true", help="only the evaluation-only sentiment comparison models")
    args = ap.parse_args()
    if args.comparison:
        args.skip_verify = True

    failures: list[str] = []
    manifest: dict = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0]}

    libs = {}
    for lib in REQUIRED_LIBS:
        try:
            libs[lib] = md.version(lib)
        except md.PackageNotFoundError:
            libs[lib] = None
            failures.append(f"library missing: {lib}")
    manifest["libraries"] = libs

    if not Path("D:\\").exists():
        failures.append("D: drive not found")
    if not str(config.HF_HOME).upper().startswith("D:"):
        failures.append(f"HF_HOME must be on D:, got {config.HF_HOME}")
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("NOTE: an HF token is set in the environment; it is NOT used (token=False on every call).")
    if failures:
        print("FAILED preflight:", failures)
        return 1

    config.HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    models = list(config.COMPARISON_MODELS) if args.comparison else \
        list(config.REQUIRED_MODELS) + (list(config.OPTIONAL_MODELS) if args.with_optional else [])
    free_before = shutil.disk_usage("D:\\").free
    manifest["hf_env_in_python"] = {k: os.environ.get(k) for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]}
    manifest["d_free_gb_before"] = gb(free_before)
    c_free_before = shutil.disk_usage("C:\\").free
    manifest["c_free_gb_before"] = gb(c_free_before)

    needed = sum(SPACE_NEEDED_GB[m] for m in models if not registry.locate(m).installed)
    print(f"D: free {gb(free_before)} GB, needed (upper bound) {needed} GB")
    if gb(free_before) < needed:
        print("FAILED: insufficient disk space on D:")
        return 1

    records = []
    for repo in models:
        print(f"\n=== {repo}")
        rec = download(repo)
        rec["required"] = repo in config.REQUIRED_MODELS
        print(json.dumps({k: v for k, v in rec.items() if k not in ("files", "allow_patterns")}, indent=2))
        records.append(rec)
        if rec["status"] != "DOWNLOADED" and rec["required"]:
            failures.append(f"{repo}: {rec['status']} {rec.get('error', '')}")

    free_after = shutil.disk_usage("D:\\").free
    manifest["d_free_gb_after"] = gb(free_after)
    manifest["c_free_gb_after"] = gb(shutil.disk_usage("C:\\").free)
    manifest["d_used_by_download_gb"] = gb(free_before - free_after)
    manifest["models"] = records

    if not args.skip_verify and not failures:
        print("\n=== Offline verification (network disabled in a subprocess)")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="")
        proc = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_models.py"))], env=env)
        manifest["offline_verification_exit_code"] = proc.returncode
        if proc.returncode != 0:
            failures.append("offline verification failed (see artifacts/reports/model_verification.json)")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    manifest["status"] = "PASS" if not failures else "FAILED"
    out = config.ARTIFACTS_DIR / "reports" / ("model_download_comparison.json" if args.comparison else "model_download.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nMODEL SETUP: {manifest['status']}  failures={failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
