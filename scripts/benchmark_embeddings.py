"""Phase 5: MiniLM embedding checks + GPU/CPU benchmarks at 100 / 1,000 / 10,000 reviews.
Writes artifacts/reports/embedding_benchmark.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.embeddings.encoder import Embedder  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402


def main() -> int:
    import psutil
    import torch

    rev = synthetic.generate(seed=config.SEED)
    cleaned, _ = clean_batch([r.__dict__ for r in rev])
    texts = [c["text_redacted"] for c in cleaned][:10000]
    report: dict = {"n_available": len(texts), "runs": []}
    proc = psutil.Process()

    gpu = Embedder("cuda") if torch.cuda.is_available() else None
    if gpu:
        gpu.encode(texts[:256])
        for n in (100, 1000, 10000):
            for bs in (64, 128, 256):
                torch.cuda.reset_peak_memory_stats()
                o = gpu.encode(texts[:n], bs)
                report["runs"].append({"device": "cuda", "n": n, "batch_size": bs, "seconds": round(o.seconds, 3),
                                       "reviews_per_sec": round(n / o.seconds, 1),
                                       "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
        a = gpu.encode(texts[:2000], 128).vectors
        b = gpu.encode(texts[:2000], 64).vectors
        report["gpu_reproducibility_max_abs_diff"] = float(np.abs(a - b).max())
        report["dim"] = int(a.shape[1])
        report["norms_min_max"] = [float(np.linalg.norm(a, axis=1).min()), float(np.linalg.norm(a, axis=1).max())]
        del gpu
        torch.cuda.empty_cache()

    cpu = Embedder("cpu")
    cpu.encode(texts[:64])
    rss0 = proc.memory_info().rss
    for n in (100, 1000):
        o = cpu.encode(texts[:n], 64)
        report["runs"].append({"device": "cpu", "n": n, "batch_size": 64, "seconds": round(o.seconds, 3),
                               "reviews_per_sec": round(n / o.seconds, 1)})
    report["cpu_rss_delta_mb"] = round((proc.memory_info().rss - rss0) / 1024**2, 1)
    c = cpu.encode(texts[:500], 64).vectors
    if "dim" in report:
        g = Embedder("cuda").encode(texts[:500], 64).vectors if torch.cuda.is_available() else c
        report["cpu_vs_gpu_max_abs_diff"] = float(np.abs(c - g).max())
    report["checks"] = {"dim_384": report.get("dim", c.shape[1]) == 384,
                        "reproducible": report.get("gpu_reproducibility_max_abs_diff", 0) < 1e-4}
    dest = config.ARTIFACTS_DIR / "reports" / "embedding_benchmark.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
