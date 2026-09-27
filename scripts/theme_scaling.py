"""How theme discovery behaves at different batch sizes (synthetic batch, planted ground truth).

For evenly spaced subsamples of 100 ... 10K reviews, runs the full discover() procedure over a grid of
min_cluster_size / min_samples and records theme count, ARI, homogeneity, completeness and unassigned share.
Writes artifacts/reports/theme_scaling.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.embeddings.encoder import encode_cached  # noqa: E402
from ml.pipeline import subsample  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402
from ml.themes.discovery import ThemeParams, discover, embedding_texts  # noqa: E402


def main() -> int:
    from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score

    rev = synthetic.generate(seed=config.SEED)
    gt = {r.review_id: r.gt_theme for r in rev}
    rows_all = [r.__dict__ for r in rev]
    out = []
    for n in (100, 250, 500, 1000, 2500, 10192):
        cleaned, _ = clean_batch(subsample(rows_all, n))
        texts = [c["text_redacted"] for c in cleaned]
        truth = np.array([gt[c["review_id"]] for c in cleaned])
        X, _ = encode_cached(embedding_texts(texts), "cuda")
        for mcs in (3, 5, 8, 10, 15, 25, 40):
            if mcs > len(texts) // 4:
                continue
            for ms in (3, 5, 10):
                res = discover(X, texts, ThemeParams(min_cluster_size=mcs, min_samples=ms))
                lab = res.labels
                a = lab >= 0
                row = {"n": len(texts), "min_cluster_size": mcs, "min_samples": min(ms, mcs), "themes": len(res.themes),
                       "unassigned": round(float((~a).mean()), 3),
                       "ari": round(adjusted_rand_score(truth[a], lab[a]), 3) if a.sum() > 1 else 0.0,
                       "homogeneity": round(homogeneity_score(truth[a], lab[a]), 3) if a.sum() > 1 else 0.0,
                       "completeness": round(completeness_score(truth[a], lab[a]), 3) if a.sum() > 1 else 0.0,
                       "n_true_themes": int(len(set(truth)))}
                out.append(row)
                print(row, flush=True)
    dest = config.ARTIFACTS_DIR / "reports" / "theme_scaling.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for n in sorted({r["n"] for r in out}):
        best = max((r for r in out if r["n"] == n), key=lambda r: (r["ari"] * (1 - r["unassigned"])))
        print("BEST", n, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
