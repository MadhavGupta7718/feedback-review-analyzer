"""Central configuration. Importing this module pins the Hugging Face cache to D: before
any transformers / sentence-transformers import can pick a default location on C:."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

HF_HOME = Path(os.environ.setdefault("HF_HOME", r"D:\huggingface"))
HF_HUB_CACHE = Path(os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub")))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

S140_PATH = Path(
    os.environ.get(
        "S140_PATH",
        r"D:\Microsoft\sentiment analysis\archive\training.1600000.processed.noemoticon.csv",
    )
)
S140_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
S140_ENCODING = "latin-1"
S140_EXPECTED_ROWS = 1_600_000

SEED = 42

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", ARTIFACTS_DIR / "analytics.db"))
if not ANALYTICS_DB.is_absolute():
    ANALYTICS_DB = REPO_ROOT / ANALYTICS_DB

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
NER_MODEL = "dslim/bert-base-NER"

REQUIRED_MODELS = [SENTIMENT_MODEL, EMBEDDING_MODEL, QWEN_MODEL]
OPTIONAL_MODELS = [NER_MODEL]
# evaluation-only baselines for the sentiment study; never used by the product pipeline
COMPARISON_MODELS = ["siebert/sentiment-roberta-large-english", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"]
FINETUNED_SENTIMENT_DIR = ARTIFACTS_DIR / "models" / "roberta-s140-binary"
# Product sentiment model after Amazon clothing fine-tune (3-class). Used when present.
AMAZON_SENTIMENT_DIR = ARTIFACTS_DIR / "models" / "amazon_roberta_sentiment"
AMAZON_SENTIMENT_EVAL = ARTIFACTS_DIR / "reports" / "amazon_sentiment_eval.json"
# Binary decision on P(pos) / (P(pos) + P(neg)) when a 3-class prediction must be scored against 2-class truth.
# Tuned for accuracy on the Sentiment140 VALIDATION split only (artifacts/reports/sentiment_study_val.json).
SENTIMENT_BINARY_THRESHOLD = 0.725

BATCHES_DIR = ARTIFACTS_DIR / "cache" / "batches"
BATCHES_INDEX = BATCHES_DIR / "index.json"
UPLOADS_DIR = DATA_DIR / "uploads"

EMBEDDING_DIM = 384
