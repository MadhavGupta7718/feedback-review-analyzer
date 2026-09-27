"""Generate the Qwen product brief locally (GPU) and store it as the precomputed 'qwen_brief' report.

  python scripts/generate_brief.py            generate, validate, store in artifacts/analytics.db
  python scripts/generate_brief.py --dry-run  generate and validate only

The deployed backend has no GPU; it serves this stored brief (generation_path "qwen_precomputed").
Only a brief that passed validation is ever stored. Run log: artifacts/reports/qwen_brief_run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml import config  # noqa: E402
from ml.brief.facts import build_facts, facts_json_for_llm  # noqa: E402
from ml.brief.qwen_writer import QwenBriefWriter, build_prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.ANALYTICS_DB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    db = Path(args.db)
    facts = build_facts(db)
    writer = QwenBriefWriter()
    res = writer.write(facts)
    v = res["validation"]
    run = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.QWEN_MODEL, "plan": writer.plan.to_dict(), "load": writer.load_stats,
        "ok": res["ok"], "attempts": v.get("attempts"), "errors": v.get("errors"), "checks": v.get("checks"),
        "raw_output": v.get("raw_output"), "prompt_chars": len(build_prompt(facts)),
        "fact_sheet_chars": len(facts_json_for_llm(facts)),
    }
    out = ROOT / "artifacts" / "reports" / "qwen_brief_run.json"
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("ok", "attempts", "errors")}, indent=2, ensure_ascii=False))
    print("\nRAW OUTPUT:\n" + (run["raw_output"] or ""))
    if res["ok"]:
        brief = res["brief"]
        brief["validation"] = {k: v[k] for k in ("passed", "errors", "checks", "slots_used")}
        brief["validation"]["attempts"] = len(v["attempts"])
        print("\nFINAL SUMMARY:\n" + brief["executive_summary"])
        print("\nINVESTIGATE:\n" + "\n".join("- " + i for i in brief["suggested_investigation_areas"]))
        if not args.dry_run:
            con = sqlite3.connect(db)
            with con:
                con.execute("INSERT OR REPLACE INTO reports(key, data) VALUES (?, ?)", ("qwen_brief", json.dumps(brief, ensure_ascii=False)))
            con.close()
            print(f"\nstored qwen_brief in {db}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
