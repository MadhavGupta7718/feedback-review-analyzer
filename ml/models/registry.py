"""Local-only model loading.

Application code never downloads a model: every loader passes local_files_only=True and a missing
model raises ModelNotInstalledError with an actionable message. The three states the rest of the
system reports are NOT_INSTALLED, LOAD_FAILED and LOADED.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ml import config


class ModelState(str, Enum):
    NOT_INSTALLED = "NOT_INSTALLED"
    LOAD_FAILED = "LOAD_FAILED"
    LOADED = "LOADED"


class ModelNotInstalledError(RuntimeError):
    def __init__(self, repo_id: str):
        super().__init__(
            f"MODEL NOT INSTALLED: '{repo_id}' is not in the local cache ({config.HF_HUB_CACHE}). "
            f"Run `python scripts/download_models.py` once with internet access. "
            f"The application never downloads models at runtime."
        )
        self.repo_id = repo_id


@dataclass
class LocalModel:
    repo_id: str
    path: Path | None
    revision: str | None

    @property
    def installed(self) -> bool:
        return self.path is not None


def locate(repo_id: str) -> LocalModel:
    """Find the cached snapshot for repo_id without touching the network."""
    repo_dir = config.HF_HUB_CACHE / ("models--" + repo_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if not ref.exists():
        return LocalModel(repo_id, None, None)
    revision = ref.read_text().strip()
    snap = repo_dir / "snapshots" / revision
    if not (snap / "config.json").exists():
        return LocalModel(repo_id, None, revision)
    return LocalModel(repo_id, snap, revision)


def require(repo_id: str) -> LocalModel:
    m = locate(repo_id)
    if not m.installed:
        raise ModelNotInstalledError(repo_id)
    return m


def dir_size_bytes(path: Path) -> int:
    """Size of a snapshot, following the symlinks / copies HF places in snapshots/."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.resolve().stat().st_size
    return total


def load_sentiment(device: str = "cpu"):
    """Load Amazon fine-tuned 3-class model when present; else pretrained cardiffnlp."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    amazon = config.AMAZON_SENTIMENT_DIR
    if (amazon / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(str(amazon), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(amazon), local_files_only=True)
        model.eval().to(device)
        if device == "cuda":
            model = model.to(torch.float16)
        local = LocalModel(repo_id=f"local:{amazon.name}", path=amazon, revision="amazon-finetune")
        return tok, model, local

    m = require(config.SENTIMENT_MODEL)
    tok = AutoTokenizer.from_pretrained(m.path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(m.path, local_files_only=True)
    model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.float16)
    return tok, model, m


def load_embedder(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    m = require(config.EMBEDDING_MODEL)
    model = SentenceTransformer(str(m.path), device=device, local_files_only=True)
    return model, m


def load_qwen(plan):
    """Load Qwen according to a hardware plan (see ml.hardware.plan_qwen). Returns (tok, model, local, stats)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = require(config.QWEN_MODEL)
    if plan.device == "none":
        raise RuntimeError(f"Qwen disabled by hardware plan: {plan.reason}")
    dtype = getattr(torch, plan.dtype)
    kwargs: dict = {"local_files_only": True, "low_cpu_mem_usage": True}
    if plan.quantization == "bnb-4bit-nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = dtype
        kwargs["device_map"] = {"": 0} if plan.device == "cuda" else {"": "cpu"}

    if plan.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(m.path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(m.path, **kwargs)
    model.eval()
    stats = {"load_seconds": round(time.perf_counter() - t0, 2)}
    if plan.device == "cuda":
        stats["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        stats["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    return tok, model, m, stats
