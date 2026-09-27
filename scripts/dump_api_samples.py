"""Dump one response per endpoint.

  python scripts/dump_api_samples.py             -> artifacts/cache/api_samples/ (gitignored, trimmed for reading)
  python scripts/dump_api_samples.py --fixtures  -> frontend/src/test/fixtures/ (frontend test fixtures; lists capped,
                                                     content is synthetic and already PII-redacted)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402
from ml.pii.leak_scan import scan  # noqa: E402


def trim(obj, cap: int, marker: bool):
    if isinstance(obj, list):
        out = [trim(x, cap, marker) for x in obj[:cap]]
        if marker and len(obj) > cap:
            out.append(f"... {len(obj) - cap} more")
        return out
    if isinstance(obj, dict):
        return {k: trim(v, cap, marker) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true")
    args = ap.parse_args()
    out = ROOT / ("frontend/src/test/fixtures" if args.fixtures else "artifacts/cache/api_samples")
    cap = 40 if args.fixtures else 3
    out.mkdir(parents=True, exist_ok=True)
    c = TestClient(app)
    issues = c.get("/issues").json()["issues"]
    tid = issues[0]["theme_id"]
    rid = c.get("/reviews?limit=1").json()["reviews"][0]["review_id"]
    calls = {
        "health": ("get", "/health"), "metrics": ("get", "/metrics"), "themes": ("get", "/themes"),
        "theme": ("get", f"/themes/{tid}"), "issues": ("get", "/issues"), "issue": ("get", f"/issues/{tid}"),
        "evidence": ("get", f"/issues/{tid}/evidence"), "reviews": ("get", "/reviews?limit=5"),
        "review": ("get", f"/reviews/{rid}"), "sentiment": ("get", "/sentiment/validation"), "drift": ("get", "/drift"),
        "data_health": ("get", "/data-health"), "model_info": ("get", "/model-info"),
    }
    for name, (m, url) in calls.items():
        body = getattr(c, m)(url).json()
        if args.fixtures and name == "model_info":
            body.pop("hardware_at_pipeline_run", None)
        (out / f"{name}.json").write_text(json.dumps(trim(body, cap, not args.fixtures), indent=1), encoding="utf-8")
    brief = c.post("/product-brief", json={"engine": "template"}).json()
    (out / "brief.json").write_text(json.dumps(brief, indent=1), encoding="utf-8")
    if args.fixtures:
        leaks = [(p.name, scan(p.read_text(encoding="utf-8"))) for p in out.glob("*.json")]
        print("leak scan (whole-file, digit runs include timestamps):", {n: h for n, h in leaks if set(h) - {"long_digit_run"}})
    print("wrote", out)


if __name__ == "__main__":
    main()
