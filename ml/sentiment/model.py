"""RoBERTa sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest), local-only, batched."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ml import config
from ml.models import registry
from ml.preprocessing.clean import model_text

LABELS = ("negative", "neutral", "positive")


@dataclass
class SentimentOutput:
    labels: list[str]
    confidences: np.ndarray  # max probability per text
    probs: np.ndarray        # (n, 3) in LABELS order
    seconds: float
    device: str
    batch_size: int
    model: str
    revision: str | None


class SentimentModel:
    def __init__(self, device: str = "cpu", max_length: int = 128):
        import torch

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tok, self.model, local = registry.load_sentiment(device)
        self.revision = local.revision
        self.model_id = local.repo_id
        id2label = {i: str(self.model.config.id2label[i]).lower() for i in range(self.model.config.num_labels)}
        self.order = []
        for lab in LABELS:
            try:
                self.order.append(next(i for i, l in id2label.items() if l == lab or l.startswith(lab[:3])))
            except StopIteration:
                self.order.append(LABELS.index(lab) if LABELS.index(lab) < self.model.config.num_labels else 0)

    def predict(self, texts: list[str], batch_size: int = 64) -> SentimentOutput:
        torch = self.torch
        inputs = [model_text(t) for t in texts]
        order = np.argsort([len(t) for t in inputs])  # length-sorted batches = less padding
        probs = np.zeros((len(inputs), 3), dtype=np.float32)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tok([inputs[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                p = torch.softmax(logits, dim=-1)[:, self.order].cpu().numpy()
                probs[idx] = p
        if self.device == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        return SentimentOutput(
            labels=[LABELS[i] for i in probs.argmax(1)],
            confidences=probs.max(1),
            probs=probs,
            seconds=secs,
            device=self.device,
            batch_size=batch_size,
            model=getattr(self, "model_id", None) or config.SENTIMENT_MODEL,
            revision=self.revision,
        )
