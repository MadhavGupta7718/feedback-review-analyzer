import re

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from ml import config
from ml.pii.leak_scan import scan

pytestmark = pytest.mark.skipif(not config.ANALYTICS_DB.exists(), reason="run `python -m ml.pipeline` first")


@pytest.fixture(scope="module")
def client():
    import os

    os.environ["BRIEF_MODE"] = "template"
    with TestClient(app) as c:
        yield c


ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}(T[\d:.+]+)?(Z|[+-]\d{2}:\d{2})?$")
HEX = re.compile(r"^[0-9a-f]{12,64}$")


MACHINE_METADATA_KEYS = {"hardware_at_pipeline_run"}  # OS/driver version strings, not user content


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in MACHINE_METADATA_KEYS:
                yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def assert_no_pii(payload):
    """Every string in the response (review texts, briefs, names, notes) must be free of PII patterns.
    Timestamps and model revision hashes are the only strings exempt from the digit-run rule."""
    for s in _strings(payload):
        if ISO_TS.match(s) or HEX.match(s):
            continue
        assert not scan(s), (scan(s), s[:200])


GET_ENDPOINTS = ["/health", "/metrics", "/themes", "/issues", "/reviews", "/sentiment/validation", "/drift", "/data-health", "/model-info"]


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoints_ok_and_pii_free(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert_no_pii(r.json())


def test_health_reports_real_data(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["reviews"] > 1000


def test_metrics_schema_and_consistency(client):
    ov = client.get("/metrics").json()["overview"]
    for k in ("total_reviews", "sentiment_pct", "n_themes", "top_complaint", "emerging", "pii_redactions_total", "drift_status"):
        assert k in ov
    assert abs(sum(ov["sentiment_pct"].values()) - 100) < 0.5
    assert sum(ov["sentiment_counts"].values()) == ov["total_reviews"]


def test_themes_list_sort_filter_search(client):
    all_t = client.get("/themes").json()
    assert all_t["count"] == len(all_t["themes"]) > 0
    sizes = [t["size"] for t in all_t["themes"]]
    assert sizes == sorted(sizes, reverse=True)
    comp = client.get("/themes", params={"complaints_only": True}).json()["themes"]
    assert comp and all(t["is_complaint"] for t in comp)
    neg = [t["negative_pct"] for t in client.get("/themes", params={"sort": "negative_pct"}).json()["themes"]]
    assert neg == sorted(neg, reverse=True)
    assert client.get("/themes", params={"q": "zzzz-nothing"}).json()["count"] == 0  # empty result


def test_theme_detail_traceable(client):
    tid = client.get("/themes").json()["themes"][0]["theme_id"]
    t = client.get(f"/themes/{tid}").json()
    assert t["representative_reviews"]
    for r in t["representative_reviews"]:
        assert r["theme_id"] == tid
        assert client.get(f"/reviews/{r['review_id']}").json()["text"] == r["text"]
    assert "weekly_counts" in t and "radar" in t and "segments" in t


def test_issue_list_detail_evidence(client):
    body = client.get("/issues", params={"status": "NEW,EMERGING"}).json()
    assert body["count"] >= 1 and body["formula"]
    assert all(i["status"] in ("NEW", "EMERGING") for i in body["issues"])
    it = body["issues"][0]
    detail = client.get(f"/issues/{it['theme_id']}").json()
    assert "calculation" in detail and "growth" in detail["calculation"]
    ev = client.get(f"/issues/{it['theme_id']}/evidence").json()
    radar_ids = [e["review_id"] for e in ev["evidence"] if e["evidence_kind"] == "radar_evidence"]
    assert radar_ids == detail["evidence_review_ids"]
    for e in ev["evidence"]:
        assert e["theme_id"] == it["theme_id"]


def test_reviews_pagination_and_filters(client):
    p1 = client.get("/reviews", params={"limit": 5}).json()
    p2 = client.get("/reviews", params={"limit": 5, "offset": 5}).json()
    assert len(p1["reviews"]) == 5 and p1["total"] > 10
    assert not {r["review_id"] for r in p1["reviews"]} & {r["review_id"] for r in p2["reviews"]}
    neg = client.get("/reviews", params={"sentiment": "negative", "limit": 20}).json()["reviews"]
    assert all(r["sentiment"] == "negative" for r in neg)
    assert client.get("/reviews", params={"q": "%_%"}).status_code == 200  # LIKE wildcards are escaped


@pytest.mark.parametrize("path", ["/themes/theme_999", "/issues/theme_999", "/issues/theme_999/evidence", "/reviews/NOPE123"])
def test_missing_resources_404(client, path):
    r = client.get(path)
    assert r.status_code == 404 and "detail" in r.json()


@pytest.mark.parametrize("path,params", [
    ("/themes/../../etc/passwd", None), ("/themes/theme_1", None), ("/themes/DROP TABLE", None),
    ("/reviews/" + "x" * 50, None), ("/reviews", {"limit": 0}), ("/reviews", {"limit": 500}),
    ("/reviews", {"sentiment": "angry"}), ("/themes", {"sort": "evil"}), ("/issues", {"status": "new;drop"}),
    ("/reviews", {"theme_id": "1 OR 1=1"}),
])
def test_malformed_input_rejected(client, path, params):
    r = client.get(path, params=params)
    assert r.status_code in (404, 422), (path, r.status_code)
    assert "Traceback" not in r.text and "sqlite" not in r.text.lower()


def test_product_brief_template(client):
    r = client.post("/product-brief", json={"engine": "template"})
    assert r.status_code == 200
    b = r.json()
    assert b["generation_path"] == "template"
    for k in ("executive_summary", "top_complaints", "emerging_complaints", "evidence", "possible_associations",
              "suggested_investigation_areas", "caveats"):
        assert k in b
    assert_no_pii(b)
    ids = {e["review_id"] for e in b["evidence"]}
    for rid in ids:
        assert client.get(f"/reviews/{rid}").status_code == 200  # no fabricated evidence


def test_product_brief_precomputed_qwen_is_served_without_live_model(client, monkeypatch):
    monkeypatch.setenv("BRIEF_MODE", "precomputed")
    b = client.post("/product-brief", json={"engine": "auto"}).json()
    if b["generation_path"] == "template":
        pytest.skip("no precomputed Qwen brief in this database (run scripts/generate_brief.py)")
    assert b["generation_path"] == "qwen_precomputed"
    assert any("BRIEF_MODE=precomputed" in r for r in b["fallback_reasons"])
    assert b["validation"]["passed"] is True
    assert_no_pii(b)
    t = client.post("/product-brief", json={"engine": "template"}).json()
    for k in ("top_complaints", "emerging_complaints", "evidence", "caveats"):
        assert b[k] == t[k]  # only the prose sections come from the model
    nums = set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?%?", b["executive_summary"]))
    m = client.get("/metrics").json()["overview"]
    issues = client.get("/issues").json()["issues"]
    themes = client.get("/themes").json()["themes"]
    known = {f"{m['total_reviews']:,}", f"{m['sentiment_pct']['negative']:g}%", "14"}
    for i in issues:
        known |= {f"{i['current_mentions']:,}", f"{i['previous_mentions']:,}", f"{round(i['negative_ratio'] * 100, 1):g}%", i["growth_label"].lstrip("+-")}
    for t_ in themes:
        known |= {f"{t_['negative_count']:,}", f"{t_['size']:,}", f"{t_['negative_pct']:g}%"}
    assert nums <= known, nums - known


def test_product_brief_template_mode_disables_qwen(client, monkeypatch):
    monkeypatch.setenv("BRIEF_MODE", "template")
    b = client.post("/product-brief", json={"engine": "auto"}).json()
    assert b["generation_path"] == "template"
    assert any("BRIEF_MODE=template" in r for r in b["fallback_reasons"])


def test_product_brief_invalid_engine(client):
    assert client.post("/product-brief", json={"engine": "gpt-9"}).status_code == 422
    assert client.post("/product-brief", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_cors_allows_configured_origin_only(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_backend_failure_returns_safe_503(monkeypatch, client):
    monkeypatch.setenv("ANALYTICS_DB", "Z:/definitely/missing.db")
    for path in ("/metrics", "/themes", "/issues"):
        r = client.get(path)
        assert r.status_code == 503
        assert "Z:/" not in r.text and "Traceback" not in r.text
    assert client.get("/health").status_code == 503


def test_all_theme_and_issue_details_pii_free(client):
    for t in client.get("/themes").json()["themes"]:
        assert_no_pii(client.get(f"/themes/{t['theme_id']}").json())
        assert_no_pii(client.get(f"/issues/{t['theme_id']}/evidence").json())


def test_no_raw_dataset_route(client):
    for path in ("/dataset", "/raw", "/download", "/data/raw/synthetic_reviews.csv"):
        assert client.get(path).status_code == 404
