# Demo

## Storyline (upload + Amazon model)

1. **Model Validation:** open `/sentiment`. Show TEST accuracy and macro recall from the Amazon clothing fine-tune
   (held-out 10%). Emphasize this is the product model’s global score — not tied to any upload.
2. **Upload & batches:** drop a CSV on `/batches`. Wait for the job to finish; click **View**.
3. **Executive Overview:** sentiment split and top complaint themes for that batch.
4. **Themes / Evidence:** drill into a theme; open a review ID (redacted text + probabilities).
5. **Complaint Radar / Data Health drift:**
   - If the CSV has dates → show NEW/EMERGING and drift tables.
   - If not → empty state: “This view needs review dates; none were found in this dataset.”

## Run

```powershell
# once: train model
.\.venv\Scripts\python.exe scripts\prepare_amazon_clothing.py
.\.venv\Scripts\python.exe scripts\finetune_amazon_sentiment.py --epochs 3 --batch-size 16

# API + UI
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
cd frontend; npm run dev
```

Open http://127.0.0.1:5173.

Legacy: `scripts/run_demo.py` can still serve a precomputed DB for offline smoke tests, but the product path is
**train Amazon model → upload CSV → activate batch**.
