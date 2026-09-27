"""Parameter sweep for theme discovery on the synthetic batch (ground truth available).
HDBSCAN is run once per (pca, min_cluster_size) and the merge threshold is swept on top of it.
Writes artifacts/reports/theme_sweep.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.embeddings.encoder import encode_cached  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402
from ml.themes.discovery import merge_micro_clusters  # noqa: E402


def main() -> int:
    import hdbscan
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score

    rev = synthetic.generate(seed=config.SEED)
    cleaned, _ = clean_batch([r.__dict__ for r in rev])
    gt = {r.review_id: r.gt_theme for r in rev}
    truth = np.array([gt[c["review_id"]] for c in cleaned])
    X, _ = encode_cached([c["text_redacted"] for c in cleaned], "cuda")
    rows = []
    for pca in (10, 20, 50):
        Xr = PCA(n_components=pca, random_state=config.SEED).fit_transform(X)
        for mcs in (25, 40, 80):
            t0 = time.perf_counter()
            raw = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10, core_dist_n_jobs=1).fit_predict(Xr)
            secs = time.perf_counter() - t0
            for thr in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0):
                lab = merge_micro_clusters(X, raw, thr)
                a = lab >= 0
                rows.append({"pca": pca, "min_cluster_size": mcs, "merge_threshold": thr, "hdbscan_seconds": round(secs, 1),
                             "micro": int(len(set(raw)) - (1 if -1 in raw else 0)),
                             "themes": int(len(set(lab)) - (1 if -1 in lab else 0)),
                             "noise": round(float((raw < 0).mean()), 3),
                             "ari": round(adjusted_rand_score(truth[a], lab[a]), 3),
                             "homogeneity": round(homogeneity_score(truth[a], lab[a]), 3),
                             "completeness": round(completeness_score(truth[a], lab[a]), 3)})
                print(rows[-1], flush=True)
    dest = config.ARTIFACTS_DIR / "reports" / "theme_sweep.json"
    dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    best = max(rows, key=lambda r: r["ari"])
    print("BEST by ARI:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
