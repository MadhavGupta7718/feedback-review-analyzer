"""End-to-end pipeline benchmark at 100 / 1,000 / full (~10K) reviews on GPU and CPU.

Each configuration runs in a fresh subprocess (clean GPU memory, no warm caches) with the embedding cache disabled,
writing to a throwaway database under artifacts/cache/. Results: artifacts/reports/pipeline_benchmark.json.

  python scripts/benchmark_pipeline.py                    # synthetic: GPU 100/1K/10K, CPU 100/1K/10K
  python scripts/benchmark_pipeline.py --skip-cpu-full    # skip the slow CPU 10K run
  python scripts/benchmark_pipeline.py --source sentiment140 --sizes 10000 --devices cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one(source: str, limit: int | None, device: str, out: Path) -> None:
    from ml.pipeline import run

    db = ROOT / "artifacts" / "cache" / f"bench_{source}_{limit or 'full'}_{device}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    res = run(source, limit, db, device, use_embedding_cache=False)
    perf = res["performance"]
    out.write_text(json.dumps({
        "source": source, "limit": limit, "device": device, "n_reviews": perf["n_reviews"],
        "total_seconds": perf["total_seconds"], "stages": {k: v["seconds"] for k, v in perf["stages"].items()},
        "peak_rss_mb": perf["peak_rss_mb"], "peak_gpu_mem_gb": perf["peak_gpu_mem_gb"],
        "sentiment_reviews_per_sec": perf["sentiment_reviews_per_sec"],
        "embedding_seconds": perf["embedding"].get("seconds"),
        "n_themes": res["overview"]["n_themes"], "emerging": [e["name"] for e in res["overview"]["emerging"]],
        "traceability": res["traceability"]["status"],
    }), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "sentiment140"])
    ap.add_argument("--sizes", default="100,1000,full")
    ap.add_argument("--devices", default="cuda,cpu")
    ap.add_argument("--skip-cpu-full", action="store_true")
    ap.add_argument("--one", nargs=3, metavar=("LIMIT", "DEVICE", "OUT"))
    ap.add_argument("--out", default=None, help="report file name under artifacts/reports/")
    args = ap.parse_args()
    if args.one:
        lim, dev, out = args.one
        one(args.source, None if lim == "full" else int(lim), dev, Path(out))
        return 0

    results = []
    tmp = ROOT / "artifacts" / "cache" / "bench_result.json"
    for dev in args.devices.split(","):
        for size in args.sizes.split(","):
            if dev == "cpu" and size == "full" and args.skip_cpu_full:
                continue
            print(f"--- {args.source} size={size} device={dev}", flush=True)
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--source", args.source, "--one", size, dev, str(tmp)],
                               cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                results.append({"source": args.source, "limit": size, "device": dev, "error": p.stderr[-1500:]})
                print("FAILED:", p.stderr[-800:])
                continue
            r = json.loads(tmp.read_text(encoding="utf-8"))
            r["wall_seconds_including_process_start"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            print(json.dumps({k: r[k] for k in ("n_reviews", "total_seconds", "sentiment_reviews_per_sec", "embedding_seconds",
                                                "n_themes", "traceability")}), flush=True)
    import torch

    from ml.hardware import detect_hardware

    report = {"hardware": detect_hardware().to_dict(), "torch": torch.__version__, "embedding_cache": "disabled",
              "note": "total_seconds excludes Python/torch import; wall_seconds includes process start and imports.",
              "results": results}
    out = ROOT / "artifacts" / "reports" / (args.out or f"pipeline_benchmark{'' if args.source == 'synthetic' else '_' + args.source}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
