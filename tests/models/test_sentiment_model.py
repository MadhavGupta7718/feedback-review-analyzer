"""Regression tests for the product sentiment model (and the experimental fine-tuned model if present).
Run: pytest -m models tests/models/test_sentiment_model.py

Network access is blocked for the module so every load must come from the local cache.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.models]

torch = pytest.importorskip("torch")

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.models import registry  # noqa: E402

if not registry.locate(config.SENTIMENT_MODEL).installed:
    pytest.skip("sentiment model not installed (run scripts/download_models.py)", allow_module_level=True)

from ml.sentiment.model import LABELS, SentimentModel  # noqa: E402

TEXTS = [
    "I absolutely love this, best day ever!",
    "This is terrible, I hate it so much.",
    "The meeting is at 3pm on Tuesday.",
    "not bad at all, actually pretty good",
    "[USER] thanks for the help [URL]",
    "ugh my phone died again and I lost everything",
    "can't wait for the weekend!!!",
    "worst customer service I have ever had",
]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module", autouse=True)
def offline():
    saved = {k: os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.update({"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module", params=DEVICES)
def model(request):
    m = SentimentModel(request.param)
    yield m
    del m
    if request.param == "cuda":
        torch.cuda.empty_cache()


def test_model_and_tokenizer_load_with_expected_labels(model):
    assert model.model.config.num_labels == 3
    assert [model.model.config.id2label[i].lower() for i in model.order] == list(LABELS)
    assert model.tok("hello world")["input_ids"]
    assert next(model.model.parameters()).device.type == model.device


def test_probabilities_sum_to_one_and_labels_match_argmax(model):
    out = model.predict(TEXTS, batch_size=4)
    assert out.probs.shape == (len(TEXTS), 3)
    np.testing.assert_allclose(out.probs.sum(1), 1.0, atol=1e-3)
    assert out.labels == [LABELS[i] for i in out.probs.argmax(1)]
    assert out.labels[0] == "positive" and out.labels[1] == "negative"


def test_batching_does_not_change_predictions(model):
    a = model.predict(TEXTS, batch_size=1)
    b = model.predict(TEXTS, batch_size=len(TEXTS))
    assert a.labels == b.labels
    np.testing.assert_allclose(a.probs, b.probs, atol=5e-3 if model.device == "cuda" else 1e-4)


def test_repeated_runs_are_identical(model):
    a = model.predict(TEXTS, batch_size=4)
    b = model.predict(TEXTS, batch_size=4)
    assert a.labels == b.labels
    np.testing.assert_array_equal(a.probs, b.probs)


def test_empty_batch():
    m = SentimentModel("cpu")
    out = m.predict([], batch_size=8)
    assert out.probs.shape == (0, 3) and out.labels == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cpu_and_gpu_agree():
    cpu = SentimentModel("cpu").predict(TEXTS, batch_size=4)
    gpu = SentimentModel("cuda").predict(TEXTS, batch_size=4)
    assert cpu.labels == gpu.labels
    np.testing.assert_allclose(cpu.probs, gpu.probs, atol=5e-3)


def test_threshold_decision_is_consistent_with_nearest_class_at_05():
    probs = SentimentModel("cpu").predict(TEXTS, batch_size=8).probs
    nearest = np.where(probs[:, M.POS] > probs[:, M.NEG], M.POS, M.NEG)
    assert (M.apply_threshold(M.positive_score(probs), 0.5) == nearest).all()
    tuned = M.apply_threshold(M.positive_score(probs), config.SENTIMENT_BINARY_THRESHOLD)
    assert set(np.unique(tuned)) <= {M.NEG, M.POS}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_metrics_reproduce_the_validation_study():
    """The product class reproduces the study's cached validation probabilities (same model, same preprocessing)."""
    cache = config.ARTIFACTS_DIR / "cache" / "sentiment_exp" / "val_current.npy"
    from ml.evaluation.s140_split import SPLIT_DIR

    if not cache.exists() or not (SPLIT_DIR / "val.csv").exists():
        pytest.skip("validation study not run")
    from ml.sentiment import experiment as X

    val = X.load_split("val")
    idx = np.linspace(0, len(val.y) - 1, 2000).astype(int)
    texts = [X.prepare(val.raw[i]).current for i in idx]
    probs = SentimentModel("cuda").predict(texts, batch_size=64).probs
    ref = np.load(cache)[idx]
    np.testing.assert_allclose(probs, ref, atol=1e-2)
    nearest = lambda p: np.where(p[:, M.POS] > p[:, M.NEG], M.POS, M.NEG)  # noqa: E731
    assert (nearest(probs) == nearest(ref)).mean() > 0.998


@pytest.mark.skipif(not (config.FINETUNED_SENTIMENT_DIR / "config.json").exists(), reason="fine-tuned model not trained")
def test_finetuned_model_is_separate_and_binary():
    from ml.sentiment.experiment import Predictor

    p = Predictor(config.FINETUNED_SENTIMENT_DIR, "cpu", max_length=64, dtype="float32")
    assert p.model.config.num_labels == 2 and p.cols["neutral"] is None
    probs = p.predict(TEXTS[:4], batch_size=2)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-4)
    assert (probs[:, M.NEU] == 0).all()
    assert probs[0, M.POS] > 0.5 and probs[1, M.NEG] > 0.5
    # the product model is untouched: still the 3-class hub checkpoint
    assert SentimentModel("cpu").model.config.num_labels == 3
