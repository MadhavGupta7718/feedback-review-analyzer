"""Helpers for the Sentiment140 accuracy study (scripts/sentiment_experiments.py, scripts/finetune_sentiment.py).

Every variant starts from the raw tweet. Variants marked `pii_safe=False` feed unredacted text to the model; they are
measured as references only (text is never stored or logged) and cannot be selected for the product pipeline, which
must redact before inference.
"""
from __future__ import annotations

import csv
import html
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml import config
from ml.evaluation.metrics import NEG, POS
from ml.evaluation.s140_split import SPLIT_DIR
from ml.pii import redact
from ml.preprocessing.clean import _CONTROL, _WS, model_text, repair, tidy

NEGATION = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without|n't|\w+n't|dont|cant|wont|didnt|isnt|doesnt|"
    r"wasnt|werent|havent|hasnt|hadnt|shouldnt|wouldnt|couldnt|aint|arent)\b", re.IGNORECASE)


@dataclass
class Split:
    name: str
    lines: np.ndarray
    ids: list[str]
    y: np.ndarray        # NEG / POS
    raw: list[str]


def load_split(name: str) -> Split:
    with open(SPLIT_DIR / f"{name}.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return Split(name, np.array([int(r["line"]) for r in rows]), [r["id"] for r in rows],
                 np.array([NEG if r["target"] == "0" else POS for r in rows]), [r["text"] for r in rows])


@dataclass
class Prepared:
    """Intermediate forms computed once per tweet so variants differ only in the step under test."""
    repaired: str
    redacted: str        # repaired + PII placeholders ([USER], [URL] ...)
    current: str         # production form: repaired -> redacted -> tidy (placeholders still in brackets)


def prepare(raw: str) -> Prepared:
    rep = repair(raw)
    red = redact(rep).text
    return Prepared(rep, red, tidy(red))


def _no_ftfy(raw: str) -> str:
    s = unicodedata.normalize("NFC", html.unescape(raw))
    return tidy(redact(_CONTROL.sub(" ", s)).text)


VARIANTS: dict[str, tuple[bool, str]] = {
    "current": (True, "production: ftfy + HTML entities + NFC -> PII redaction -> tidy (runs of 4+ !?. -> 3) -> @user/http"),
    "no_punct_tidy": (True, "as current but runs of punctuation kept"),
    "no_ftfy": (True, "as current but without ftfy mojibake repair (HTML entities + NFC only)"),
    "urls_removed": (True, "as current but links removed instead of 'http'"),
    "mentions_removed": (True, "as current but user mentions removed instead of '@user'"),
    "hashtag_plain": (True, "as current but '#word' -> 'word'"),
    "bracket_placeholders": (True, "as current but placeholders left as [USER]/[URL] (no model-card mapping)"),
    "raw": (False, "reference: raw tweet, no processing (unredacted)"),
    "raw_html_ws": (False, "reference: HTML entities decoded + whitespace normalised (unredacted)"),
}


def variant_texts(name: str, raw: list[str], prepared: list[Prepared]) -> list[str]:
    if name == "raw":
        return list(raw)
    if name == "raw_html_ws":
        return [_WS.sub(" ", html.unescape(r)).strip() for r in raw]
    if name == "no_ftfy":
        return [model_text(_no_ftfy(r)) for r in raw]
    out = []
    for p in prepared:
        s = p.current
        if name == "no_punct_tidy":
            s = _WS.sub(" ", p.redacted).strip()
        elif name == "urls_removed":
            s = _WS.sub(" ", s.replace("[URL]", "")).strip()
        elif name == "mentions_removed":
            s = _WS.sub(" ", s.replace("[USER]", "")).strip()
        elif name == "hashtag_plain":
            s = re.sub(r"#(\w+)", r"\1", s)
        out.append(s if name == "bracket_placeholders" else model_text(s))
    return out


def negation_count(text: str) -> int:
    return len(NEGATION.findall(html.unescape(text)))


class Predictor:
    """Local-only sequence classifier returning (n, 3) probabilities in (negative, neutral, positive) order.
    Binary models get a zero neutral column. Predictions are cached per input string."""

    def __init__(self, path: Path | str, device: str = "cuda", max_length: int = 128, dtype: str = "float16"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.max_length = torch, device, max_length
        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).eval().to(device)
        if device == "cuda" and dtype == "float16":
            self.model = self.model.to(torch.float16)
        id2 = {i: self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)}
        self.cols = {lab: next((i for i, l in id2.items() if l.startswith(lab[:3])), None) for lab in ("negative", "neutral", "positive")}
        assert self.cols["negative"] is not None and self.cols["positive"] is not None, id2
        self.cache: dict[str, np.ndarray] = {}
        self.seconds = 0.0
        self.inferred = 0

    def predict(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        torch = self.torch
        todo = sorted({t for t in texts if t not in self.cache}, key=len)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(0, len(todo), batch_size):
                chunk = todo[s:s + batch_size]
                enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                out = np.zeros((len(chunk), 3), dtype=np.float32)
                for j, lab in enumerate(("negative", "neutral", "positive")):
                    if self.cols[lab] is not None:
                        out[:, j] = p[:, self.cols[lab]]
                for t, row in zip(chunk, out):
                    self.cache[t] = row
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.inferred += len(todo)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 3), dtype=np.float32)


def sentiment_model_path() -> Path:
    from ml.models import registry

    return registry.require(config.SENTIMENT_MODEL).path
