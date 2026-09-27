"""Phase 6: run theme discovery on a batch, score against planted ground truth (synthetic), and print every
theme with keywords + representative redacted reviews for manual inspection.
Usage: python scripts/theme_experiment.py [--source synthetic|sentiment140] [--min-cluster-size N] ..."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.embeddings.encoder import encode_cached  # noqa: E402
from ml.preprocessing.clean import clean_batch  # noqa: E402
from ml.themes.discovery import ThemeParams, discover, embedding_texts, evaluate_against_truth  # noqa: E402


def load(source: str):
    if source == "synthetic":
        rev = synthetic.generate(seed=config.SEED)
        cleaned, _ = clean_batch([r.__dict__ for r in rev])
        gt = {r.review_id: r.gt_theme for r in rev}
        return cleaned, [gt[c["review_id"]] for c in cleaned]
    with open(config.DATA_DIR / "interim" / "s140_batch_10k.csv", encoding="utf-8") as fh:
        rows = [{"review_id": r["id"], "text": r["text"], "created_at": r["date"]} for r in csv.DictReader(fh)]
    cleaned, _ = clean_batch(rows)
    return cleaned, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=ThemeParams.min_samples)
    ap.add_argument("--pca", type=int, default=ThemeParams.pca_components)
    ap.add_argument("--method", default=ThemeParams.cluster_selection_method)
    ap.add_argument("--noise-threshold", type=float, default=ThemeParams.noise_assign_threshold)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    import torch

    cleaned, truth = load(args.source)
    texts = [c["text_redacted"] for c in cleaned]
    emb, emb_info = encode_cached(embedding_texts(texts), "cuda" if torch.cuda.is_available() else "cpu")
    p = ThemeParams(pca_components=args.pca, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                    cluster_selection_method=args.method, noise_assign_threshold=args.noise_threshold)
    t0 = time.perf_counter()
    res = discover(emb, texts, p)
    secs = time.perf_counter() - t0
    out = {"source": args.source, "embedding": emb_info, "cluster_seconds": round(secs, 2), "stats": res.stats}
    if truth:
        out["truth_eval"] = evaluate_against_truth(res.labels, truth, res.themes)
    themes_view = []
    for t in res.themes:
        themes_view.append({**t.to_dict(), "representatives": [texts[i] for i in t.representative_indices[:3]]})
    out["themes"] = themes_view
    dest = config.ARTIFACTS_DIR / "reports" / f"theme_experiment_{args.source}.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in out.items() if k not in ("themes",)} | (
        {"truth_eval": {k: v for k, v in out.get("truth_eval", {}).items() if k != "per_theme"}} if truth else {}), indent=1))
    if not args.quiet:
        pur = {p["theme_id"]: p for p in out.get("truth_eval", {}).get("per_theme", [])}
        for t in themes_view:
            extra = f" | truth={pur[t['theme_id']]['majority_truth']} purity={pur[t['theme_id']]['purity']}" if pur else ""
            print(f"\n{t['theme_id']} {t['name']!r} size={t['size']} coh={t['coherence']}{extra}")
            print("   kw:", ", ".join(t["keywords"]))
            for r in t["representatives"]:
                print("   -", r[:120].encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
