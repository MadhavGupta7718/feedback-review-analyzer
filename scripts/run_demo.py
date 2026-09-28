"""Offline demo. No network, no Hugging Face token, no raw dataset exposure.

  python scripts/run_demo.py                  console walkthrough of the precomputed analytics (artifacts/analytics.db)
  python scripts/run_demo.py --fresh 1000     first re-run the full pipeline on a 1,000-review synthetic batch (GPU if available)
  python scripts/run_demo.py --serve          also start the API (:8000) and the dashboard (:5173); Ctrl+C stops both

Network access is blocked for this process and its children (dead proxy + HF offline flags), so the demo proves
that everything runs from local files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HTTP_PROXY": "http://127.0.0.1:9",
               "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost"}
os.environ.update(OFFLINE_ENV)

W = 100


def h(title: str) -> None:
    print("\n" + "=" * W + f"\n{title}\n" + "=" * W)


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, W, initial_indent=indent, subsequent_indent=indent)


def walkthrough(db: Path) -> None:
    os.environ["ANALYTICS_DB"] = str(db)
    os.environ.setdefault("BRIEF_MODE", "precomputed")
    from fastapi.testclient import TestClient

    from backend.app.main import app

    c = TestClient(app)
    m = c.get("/metrics").json()
    ov = m["overview"]
    h(f"1. OVERVIEW  ({m['dataset']['name']})")
    print(wrap(f"{ov['total_reviews']:,} reviews from {ov['date_range'][0][:10]} to {ov['date_range'][1][:10]}. "
               f"Sentiment: {ov['sentiment_pct']['negative']}% negative, {ov['sentiment_pct']['neutral']}% neutral, "
               f"{ov['sentiment_pct']['positive']}% positive. {ov['n_themes']} themes ({ov['n_complaint_themes']} complaint themes). "
               f"{ov['pii_redactions_total']} personal-data items redacted before storage. Drift: {ov['drift_status']}."))
    if m["dataset"].get("kind") == "synthetic":
        print(wrap("NOTE: " + m["dataset"]["note"]))

    h("2. COMPLAINT RADAR (current 14-day window vs previous)")
    issues = c.get("/issues").json()
    for i in issues["issues"]:
        if i["status"] in ("NEW", "EMERGING", "STABLE", "DECLINING"):
            print(f"  {i['status']:<10} {i['name']:<28} {i['previous_mentions']:>5} -> {i['current_mentions']:<5} "
                  f"{i['growth_label']:>6}  {i['negative_ratio'] * 100:5.1f}% neg  priority {i['priority']:.1f}")
    em = [i for i in issues["issues"] if i["status"] in ("NEW", "EMERGING")]
    if em:
        top = c.get(f"/issues/{em[0]['theme_id']}").json()
        print(f"\n  VIEW WHY -> {top['name']} [{top['status']}]")
        for r in top["reasons"]:
            print(wrap("reason: " + r, "    "))
        print(wrap(f"growth: {top['calculation']['growth']}", "    "))
        print(wrap(f"negative share: {top['calculation']['negative_ratio']}", "    "))
        print(wrap(f"priority: {top['calculation']['priority']}   ({top['formula']})", "    "))
        ev = c.get(f"/issues/{em[0]['theme_id']}/evidence").json()["evidence"]
        print("    evidence (redacted verbatims):")
        for e in [x for x in ev if x["evidence_kind"] == "radar_evidence"][:3]:
            print(wrap(f"[{e['review_id']}] \"{e['text']}\"", "      "))

    h("3. SENTIMENT VALIDATION (held-out labelled Sentiment140 test split)")
    sv = c.get("/sentiment/validation").json()
    mt = sv["metrics"]
    print(wrap(f"{sv['model']} on {sv['sample']['size']:,} labelled tweets: binary accuracy {mt['binary_forced']['accuracy'] * 100:.1f}% "
               f"(macro F1 {mt['binary_forced']['macro_f1']:.3f}); abstaining on neutral {mt['abstain']['accuracy'] * 100:.1f}% at "
               f"{mt['abstain']['coverage'] * 100:.1f}% coverage; strict 3-class {mt['strict_3class']['accuracy'] * 100:.1f}%."))

    h("4. DATA HEALTH")
    dh = c.get("/data-health").json()
    print(wrap(f"input rows {dh['input_rows']:,} -> processed {dh['processed_reviews']:,}; rejected {dh['rejected']}; "
               f"duplicates removed {dh['ingestion_duplicates_removed']}; mojibake repaired {dh['mojibake_repaired']}."))
    print(wrap(f"PII redactions by type: {dh['pii_redactions']}; synthetic PII recall {dh['pii_audit']['overall_recall']}."))
    if dh.get("traceability"):
        print(wrap(f"traceability audit: {dh['traceability']['status']} ({dh['traceability']['links_checked']} evidence links checked)."))

    h("5. PRODUCT BRIEF")
    b = c.post("/product-brief", json={"engine": "auto"}).json()
    print(f"  generation path: {b['generation_path']}" + (f"   (fallback: {'; '.join(b['fallback_reasons'])})" if b.get("fallback_reasons") else ""))
    print(wrap(b["executive_summary"]))
    print("  Suggested investigation areas:")
    for s in b["suggested_investigation_areas"]:
        print(wrap("- " + s, "    "))
    print("\nDemo walkthrough complete. Every number above comes from the deterministic pipeline.")


def wait_http(url: str, seconds: int = 90) -> bool:
    t0 = time.time()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() - t0 < seconds:
        try:
            with opener.open(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def serve(db: Path) -> None:
    env = {**os.environ, **OFFLINE_ENV, "ANALYTICS_DB": str(db)}
    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                              cwd=ROOT, env=env)]
    npm = shutil.which("npm")
    if npm and (ROOT / "frontend" / "node_modules").exists():
        procs.append(subprocess.Popen([npm, "run", "dev", "--", "--host", "127.0.0.1"], cwd=ROOT / "frontend",
                                      env={**env, "VITE_API_BASE_URL": "http://127.0.0.1:8000"}))
    else:
        print("Frontend not started (run `npm ci` in frontend/ first). API only.")
    try:
        ok_api = wait_http("http://127.0.0.1:8000/health")
        print(f"\nAPI: http://127.0.0.1:8000/docs  ({'up' if ok_api else 'NOT RESPONDING'})")
        if len(procs) > 1:
            ok_ui = wait_http("http://127.0.0.1:5173/")
            print(f"Dashboard: http://127.0.0.1:5173/  ({'up' if ok_ui else 'NOT RESPONDING'})")
        print("Press Ctrl+C to stop.")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", type=int, default=None, metavar="N", help="re-run the pipeline on N synthetic reviews first")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args()
    from ml import config

    db = args.db or config.ANALYTICS_DB
    if args.fresh:
        from ml.pipeline import run

        db = ROOT / "artifacts" / "cache" / f"demo_{args.fresh}.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        h(f"0. PIPELINE on {args.fresh:,} synthetic reviews (offline)")
        res = run("synthetic", args.fresh, db)
        print(f"  total {res['performance']['total_seconds']} s on {res['performance']['device']}; traceability {res['traceability']['status']}")
    if not Path(db).exists():
        print(f"Analytics database not found: {db}. Run `python -m ml.pipeline` first.")
        return 1
    walkthrough(Path(db))
    if args.serve:
        serve(Path(db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
