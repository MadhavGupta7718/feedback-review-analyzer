"""MiniLM sentence embeddings (sentence-transformers/all-MiniLM-L6-v2), L2-normalised, 384-d, local-only."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from ml import config
from ml.models import registry


@dataclass
class EmbeddingOutput:
    vectors: np.ndarray
    seconds: float
    device: str
    batch_size: int
    revision: str | None


class Embedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, local = registry.load_embedder(device)
        self.revision = local.revision

    def encode(self, texts: list[str], batch_size: int = 128) -> EmbeddingOutput:
        t0 = time.perf_counter()
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return EmbeddingOutput(vecs.astype(np.float32), time.perf_counter() - t0, self.device, batch_size, self.revision)


def cache_key(texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(config.EMBEDDING_MODEL.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode_cached(texts: list[str], device: str, batch_size: int = 128) -> tuple[np.ndarray, dict]:
    """Embeddings are an artifact (artifacts/embeddings/*.npy), not database rows."""
    d = config.ARTIFACTS_DIR / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cache_key(texts)}.npy"
    if path.exists():
        return np.load(path), {"cached": True, "path": str(path)}
    emb = Embedder(device)
    out = emb.encode(texts, batch_size)
    np.save(path, out.vectors)
    return out.vectors, {"cached": False, "path": str(path), "seconds": round(out.seconds, 2), "device": device,
                         "batch_size": batch_size, "revision": out.revision}
