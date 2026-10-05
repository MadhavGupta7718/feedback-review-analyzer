"""Safe hyperparameter sweep on teacher-labeled Amazon data. Saves each run; restores best vs baseline."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "amazon_clothing" / "reviews_teacher_roberta.csv"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
FINETUNE = ROOT / "scripts" / "finetune_amazon_sentiment.py"
OUT_DIR = ROOT / "artifacts" / "models" / "amazon_roberta_sentiment"
REPORT = ROOT / "artifacts" / "reports" / "amazon_sentiment_eval.json"
SWEEP_DIR = ROOT / "artifacts" / "reports" / "hyperparam_sweep"
MODEL_SWEEP = ROOT / "artifacts" / "models" / "hyperparam_sweep"

BASELINE = {
    "name": "baseline_current",
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 128,
    "note": "Teacher run before sweep (reference)",
}

RUNS = [
    {"name": "A_lr1e-5", "lr": 1e-5, "batch_size": 16, "max_length": 128},
    {"name": "B_lr3e-5", "lr": 3e-5, "batch_size": 16, "max_length": 128},
    {"name": "C_bs8", "lr": 2e-5, "batch_size": 8, "max_length": 128},
    {"name": "D_bs32", "lr": 2e-5, "batch_size": 32, "max_length": 128},
    {"name": "E_max256", "lr": 2e-5, "batch_size": 16, "max_length": 256},
]


def metrics_from_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    m = r.get("metrics") or {}
    test = m.get("test") or {}
    return {
        "val_macro_recall": r.get("best_val_macro_recall"),
        "test_accuracy": test.get("accuracy") or m.get("headline_accuracy"),
        "test_macro_recall": test.get("macro_recall") or m.get("headline_macro_recall"),
        "history": r.get("history"),
        "hyperparams": r.get("hyperparams"),
    }


def run_one(cfg: dict) -> dict:
    cmd = [
        str(PY),
        str(FINETUNE),
        "--data",
        str(DATA),
        "--epochs",
        "3",
        "--lr",
        str(cfg["lr"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--max-length",
        str(cfg["max_length"]),
    ]
    print(f"\n=== RUN {cfg['name']} ===", flush=True)
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    out = {
        "name": cfg["name"],
        "config": cfg,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
    }
    if proc.returncode != 0:
        out["error"] = "finetune failed (OOM or crash)"
        return out
    met = metrics_from_report(REPORT)
    out.update(met or {})
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SWEEP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT, SWEEP_DIR / f"{cfg['name']}.json")
    if OUT_DIR.exists():
        dest = MODEL_SWEEP / cfg["name"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(OUT_DIR, dest)
    return out


def main() -> int:
    if not DATA.exists():
        print(f"Missing {DATA}", file=sys.stderr)
        return 1
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        shutil.copy2(REPORT, SWEEP_DIR / "00_baseline_before_sweep.json")
    if OUT_DIR.exists():
        dest = MODEL_SWEEP / "00_baseline_before_sweep"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(OUT_DIR, dest)

    baseline_met = metrics_from_report(SWEEP_DIR / "00_baseline_before_sweep.json")
    if baseline_met:
        baseline_met["name"] = BASELINE["name"]
        baseline_met["config"] = BASELINE

    results: list[dict] = []
    if baseline_met:
        results.append(baseline_met)

    for cfg in RUNS:
        results.append(run_one(cfg))

    # Pick best by val macro recall, tie-break test macro recall
    ok = [r for r in results if r.get("ok", True) and r.get("val_macro_recall") is not None]
    best = max(
        ok,
        key=lambda r: (float(r["val_macro_recall"]), float(r.get("test_macro_recall") or 0)),
    )
    ref = baseline_met or {}
    ref_val = float(ref.get("val_macro_recall") or 0)
    ref_test = float(ref.get("test_macro_recall") or 0)

    summary = {
        "baseline_val_macro_recall": ref_val,
        "baseline_test_macro_recall": ref_test,
        "baseline_test_accuracy": ref.get("test_accuracy"),
        "best_run": best.get("name"),
        "best_val_macro_recall": best.get("val_macro_recall"),
        "best_test_macro_recall": best.get("test_macro_recall"),
        "best_test_accuracy": best.get("test_accuracy"),
        "beats_baseline": (
            float(best.get("val_macro_recall") or 0) > ref_val + 1e-6
            or float(best.get("test_macro_recall") or 0) > ref_test + 1e-6
        ),
        "results": [
            {
                "name": r.get("name"),
                "config": r.get("config"),
                "ok": r.get("ok", True),
                "val_macro_recall": r.get("val_macro_recall"),
                "test_macro_recall": r.get("test_macro_recall"),
                "test_accuracy": r.get("test_accuracy"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    summary_path = SWEEP_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    winner = best.get("name")
    if winner and winner != BASELINE["name"]:
        src_model = MODEL_SWEEP / winner
        src_rep = SWEEP_DIR / f"{winner}.json"
        if src_model.exists() and summary["beats_baseline"]:
            if OUT_DIR.exists():
                shutil.rmtree(OUT_DIR)
            shutil.copytree(src_model, OUT_DIR)
            shutil.copy2(src_rep, REPORT)
            print(f"Restored production model from sweep winner: {winner}", flush=True)
        elif not summary["beats_baseline"]:
            # restore baseline snapshot
            bmodel = MODEL_SWEEP / "00_baseline_before_sweep"
            brep = SWEEP_DIR / "00_baseline_before_sweep.json"
            if bmodel.exists():
                if OUT_DIR.exists():
                    shutil.rmtree(OUT_DIR)
                shutil.copytree(bmodel, OUT_DIR)
            if brep.exists():
                shutil.copy2(brep, REPORT)
            print("No run beat baseline; restored pre-sweep model.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
