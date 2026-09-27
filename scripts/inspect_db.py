"""Print a compact summary of artifacts/analytics.db (radar table + drift) for manual inspection."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else config.ANALYTICS_DB
    con = sqlite3.connect(db)
    print(f"{'theme':10} {'name':24} {'status':22} {'cur':>5} {'prev':>5} {'growth':>7} {'neg%':>5} {'trend':8} assoc")
    for (d,) in con.execute("SELECT data FROM issues ORDER BY priority DESC"):
        d = json.loads(d)
        print(f"{d['theme_id']:10} {d['name'][:24]:24} {d['status']:22} {d['current_mentions']:>5} {d['previous_mentions']:>5} "
              f"{d['growth_label']:>7} {round(d['negative_ratio'] * 100):>5} {d['trend']:8} {[a['value'] for a in d['associations']]}")
    drift = json.loads(con.execute("SELECT data FROM reports WHERE key='drift'").fetchone()[0])
    for k, v in drift["current_vs_previous"]["metrics"].items():
        print(f"drift {k:14} value={v['value']} status={v['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
