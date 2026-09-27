# Demo

Everything below runs from local files. `scripts/run_demo.py` blocks network access for itself and its child processes
(a dead proxy plus the Hugging Face offline flags), so the demo also shows that no model is downloaded and no token is needed.

## 1. Console walkthrough (30 seconds, no GPU needed)

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

This prints the overview, Complaint Radar with the **View why** calculation and evidence, sentiment validation, data
health and the product brief. It reads the committed `artifacts/analytics.db` through the real API code.

## 2. Dashboard

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --serve     # API on :8000 + dashboard on :5173, Ctrl+C stops both
```

Open http://127.0.0.1:5173 (first time only: `cd frontend; npm ci`).

## 3. Re-run the whole pipeline live

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py --fresh 10000   # full batch, about 45 s on the RTX 4050
.\.venv\Scripts\python.exe scripts\run_demo.py --fresh 1000    # quick, about 17 s
```

The run writes to `artifacts/cache/demo_N.db`, so the committed database is untouched. A fresh database has no stored Qwen brief, so
the brief falls back to the template unless live Qwen runs. In a 1,000-review batch no issue reaches NEW or EMERGING. The
radar needs at least 30 current mentions, so it does not raise alarms on thin evidence.

To analyse your own reviews, see "Bring your own reviews" in `docs/DATASET.md`.

## Suggested 5-minute storyline

1. **Executive Overview:** 10,104 reviews and the sentiment split. The top complaints are ranked by priority, not size, and two
   issues are flagged. The problem statement's question, "what do customers complain about?", is answered on one screen.
2. **Complaint Radar:** Failing Payment is **NEW** (0 → 140 mentions in 14 days, 96% negative) and Battery is **EMERGING**
   (111 → 243, +119%). Crashes Phone is larger but **STABLE**, so it ranks lower. Click **View why** to show:
   - the exact rule that fired and the thresholds;
   - the calculation `135 x 4.00 = 540.0`;
   - the last two weekly bars highlighted;
   - real negative reviews from the current window, each with its review ID.
3. **Evidence:** click a review ID. The drawer shows the redacted text (PII as `[EMAIL]`, `[PHONE]` …), the class probabilities
   and which theme or issue uses it as evidence. Every number on the dashboard traces back to rows like this.
4. **Sentiment Validation:** accuracy 0.7646 on 5,000 labelled tweets, shown honestly with three scorings, the confusion
   matrix, and the weakness that neutral recall is low.
5. **Data Health:**
   - ingestion rejects and duplicates;
   - PII redactions by type, with 100% recall on 628 planted rows;
   - traceability audit PASS;
   - drift: theme mix PSI 0.59 is significant, while sentiment is stable.
6. **Product Brief:** choose `auto`. Offline or on a CPU host it serves the validated Qwen brief (`qwen_precomputed`) and says so.
   On the GPU laptop, `qwen_live` takes 1–3 minutes. Point out that the numbers were filled in by code, not by the model, and
   that every check the validator ran is listed.

## Talking points for questions

- **"Can the LLM hallucinate numbers?"** It never sees them. It writes `{E1.current}` placeholders. The validator rejects
  digits, unknown slots, reversed directions, causal claims and invented theme names, and code then fills in the values. The
  live tests include a prompt-injection review that asks for "921 reviews", and it was never accepted.
- **"Is the sentiment model any good?"** 76.5% binary accuracy on noisy emoticon-labelled tweets, measured, not claimed.
  The weaknesses are documented in `docs/MODEL_CARD.md`.
- **"Why synthetic data?"** No public dataset has product reviews with known themes, trends and PII, which you need to measure theme
  recovery, radar correctness and PII recall. Real Sentiment140 data is used where ground truth exists (sentiment).
- **"GPU?"** The full 10K pipeline takes 46.7 s on the RTX 4050 vs 455.4 s on the CPU (9.8×). The deployed API needs no GPU.
