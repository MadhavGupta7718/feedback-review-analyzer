"""Verify every locally installed model: files, loading with local_files_only=True, real inference,
output structure, and (for Qwen) GPU/VRAM detection plus basic fact preservation.

Run it with networking disabled to prove offline operation, e.g.
  $env:HF_HUB_OFFLINE=1; $env:HTTPS_PROXY="http://127.0.0.1:9"; python scripts/verify_models.py
(download_models.py does exactly this in a subprocess).
"""
from __future__ import annotations

import gc
import importlib.metadata as md
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.hardware import detect_hardware, plan_qwen  # noqa: E402
from ml.models import registry  # noqa: E402


def network_blocked() -> bool:
    try:
        import httpx

        httpx.get("https://huggingface.co", timeout=5)
        return False
    except Exception:  # noqa: BLE001
        return True


def file_checks(repo_id: str) -> dict:
    m = registry.locate(repo_id)
    if not m.installed:
        return {"state": registry.ModelState.NOT_INSTALLED.value}
    files = [p.name for p in m.path.rglob("*") if p.is_file()]
    return {
        "local_path": str(m.path),
        "revision": m.revision,
        "disk_gb": round(registry.dir_size_bytes(m.path) / 1024**3, 3),
        "config": "config.json" in files,
        "tokenizer": any(f.startswith(("tokenizer", "vocab")) for f in files),
        "weights": any(f.endswith((".safetensors", ".bin")) for f in files),
    }


def verify_sentiment(device: str) -> dict:
    import torch

    rec = file_checks(config.SENTIMENT_MODEL)
    t0 = time.perf_counter()
    tok, model, _ = registry.load_sentiment(device)
    rec["load_seconds"] = round(time.perf_counter() - t0, 2)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    rec["labels"] = labels
    texts = ["I love this app, it works perfectly!", "This is the worst update ever, it crashes constantly.", "The app was updated on Tuesday."]
    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
    rec["inference_seconds"] = round(time.perf_counter() - t0, 3)
    preds = [{"text": t, "label": labels[int(p.argmax())], "confidence": round(float(p.max()), 4)} for t, p in zip(texts, probs)]
    rec["predictions"] = preds
    rec["checks"] = {
        "labels_valid": sorted(labels) == ["negative", "neutral", "positive"],
        "probabilities_sum_to_1": bool(torch.allclose(probs.sum(-1), torch.ones(len(texts)), atol=1e-3)),
        "positive_example_positive": preds[0]["label"] == "positive",
        "negative_example_negative": preds[1]["label"] == "negative",
    }
    del model
    return rec


def verify_embeddings(device: str) -> dict:
    import numpy as np

    rec = file_checks(config.EMBEDDING_MODEL)
    t0 = time.perf_counter()
    model, _ = registry.load_embedder(device)
    rec["load_seconds"] = round(time.perf_counter() - t0, 2)
    sents = ["battery drains quickly", "battery dies fast", "payment failed"]
    t0 = time.perf_counter()
    e1 = model.encode(sents, batch_size=3, normalize_embeddings=True)
    rec["inference_seconds"] = round(time.perf_counter() - t0, 3)
    e2 = model.encode(sents, batch_size=1, normalize_embeddings=True)
    sim_close = float(np.dot(e1[0], e1[1]))
    sim_far = float(np.dot(e1[0], e1[2]))
    rec["embedding_dim"] = int(e1.shape[1])
    rec["similarity"] = {
        "battery drains quickly ~ battery dies fast": round(sim_close, 4),
        "battery drains quickly ~ payment failed": round(sim_far, 4),
    }
    rec["max_abs_diff_batch1_vs_batch3"] = float(np.abs(e1 - e2).max())
    rec["checks"] = {
        "dim_384": rec["embedding_dim"] == config.EMBEDDING_DIM,
        "semantic_ordering": sim_close > sim_far,
        "reproducible_across_batch_sizes": rec["max_abs_diff_batch1_vs_batch3"] < 1e-4,
    }
    del model
    return rec


QWEN_FACTS = {
    "total_reviews": 842,
    "theme": "Battery Drain",
    "mentions_current_period": 142,
    "mentions_previous_period": 50,
    "growth_percent": 184.0,
    "negative_percent": 81.0,
    "evidence_review_ids": ["R01002", "R01044", "R01209"],
}


def verify_qwen() -> dict:
    import torch

    rec = file_checks(config.QWEN_MODEL)
    try:
        import bitsandbytes  # noqa: F401

        bnb = True
    except Exception:  # noqa: BLE001
        bnb = False
    hw = detect_hardware()
    plan = plan_qwen(hw, bnb)
    rec["hardware"] = hw.to_dict()
    rec["plan"] = plan.to_dict()
    rec["bitsandbytes_available"] = bnb
    tok, model, _, stats = registry.load_qwen(plan)
    rec.update(stats)
    messages = [
        {"role": "system", "content": "You rewrite verified analytics into two plain sentences. Use only the numbers and IDs given. Never add numbers, never claim causes."},
        {"role": "user", "content": "Verified facts (JSON):\n" + json.dumps(QWEN_FACTS)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = {"max_new_tokens": 120, "do_sample": False, "repetition_penalty": 1.05}
    rec["generation_config"] = gen_cfg
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg, pad_token_id=tok.eos_token_id)
    rec["generation_seconds"] = round(time.perf_counter() - t0, 2)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    rec["tokens_generated"] = int(new_tokens.shape[0])
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    rec["output"] = text
    if plan.device == "cuda":
        rec["gpu_peak_memory_gb_after_generate"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    allowed = {"842", "142", "50", "184", "81", "01002", "01044", "01209", "2", "184.0", "81.0"}
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    rec["numbers_in_output"] = sorted(numbers)
    rec["checks"] = {
        "non_empty": bool(text),
        "on_gpu": plan.device == "cuda" and next(model.parameters()).is_cuda,
        "no_invented_numbers": numbers <= allowed,
        "no_921": "921" not in text,
        "no_causal_language": not re.search(r"\b(caus(e|ed|es|ing)|because of|due to|led to|result(ed)? in)\b", text, re.I),
    }
    del model
    return rec


def verify_missing_model_is_reported() -> dict:
    fake = "nonexistent-org/definitely-not-downloaded"
    try:
        registry.require(fake)
        return {"checks": {"raises_not_installed": False}}
    except registry.ModelNotInstalledError as exc:
        return {"message": str(exc), "state": registry.locate(fake).installed, "checks": {"raises_not_installed": True}}


def main() -> int:
    from ml.hardware import select_device

    device = select_device()
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "offline_env": {k: os.environ.get(k) for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HTTPS_PROXY"]},
        "network_blocked": network_blocked(),
        "device": device,
        "versions": {p: md.version(p) for p in ["torch", "transformers", "sentence-transformers", "huggingface_hub", "bitsandbytes"]},
        "models": {},
    }
    steps = [
        (config.SENTIMENT_MODEL, lambda: verify_sentiment(device)),
        (config.EMBEDDING_MODEL, lambda: verify_embeddings(device)),
        (config.QWEN_MODEL, verify_qwen),
        ("missing-model-handling", verify_missing_model_is_reported),
    ]
    failures = []
    for name, fn in steps:
        print(f"=== {name}", flush=True)
        if name != "missing-model-handling" and not registry.locate(name).installed:
            report["models"][name] = {"state": registry.ModelState.NOT_INSTALLED.value}
            failures.append(f"{name}: NOT_INSTALLED")
            continue
        try:
            rec = fn()
            rec["state"] = registry.ModelState.LOADED.value if name != "missing-model-handling" else "n/a"
        except Exception as exc:  # noqa: BLE001
            rec = {"state": registry.ModelState.LOAD_FAILED.value, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:]}
        report["models"][name] = rec
        bad = [k for k, v in rec.get("checks", {}).items() if not v]
        if rec["state"] == registry.ModelState.LOAD_FAILED.value or bad or "checks" not in rec:
            failures.append(f"{name}: {rec.get('error') or bad}")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("traceback",)}, indent=2, default=str), flush=True)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAILED"
    out = config.ARTIFACTS_DIR / "reports" / "model_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nnetwork_blocked={report['network_blocked']}  MODEL VERIFICATION: {report['status']}  {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
