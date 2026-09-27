"""Runtime hardware detection. Nothing here assumes a particular GPU: every decision is made
from what torch / psutil actually report on the running machine."""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
    python_version: str
    platform: str
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_vram_gb: float | None
    gpu_free_vram_gb: float | None
    compute_capability: str | None
    system_ram_total_gb: float
    system_ram_available_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    vm = psutil.virtual_memory()
    info = HardwareInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=None,
        torch_cuda_version=None,
        cuda_available=False,
        gpu_name=None,
        gpu_total_vram_gb=None,
        gpu_free_vram_gb=None,
        compute_capability=None,
        system_ram_total_gb=round(vm.total / 1024**3, 2),
        system_ram_available_gb=round(vm.available / 1024**3, 2),
    )
    try:
        import torch
    except ImportError:
        return info
    info.torch_version = torch.__version__
    info.torch_cuda_version = torch.version.cuda
    if torch.cuda.is_available():
        info.cuda_available = True
        info.gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        info.gpu_total_vram_gb = round(total / 1024**3, 2)
        info.gpu_free_vram_gb = round(free / 1024**3, 2)
        major, minor = torch.cuda.get_device_capability(0)
        info.compute_capability = f"{major}.{minor}"
    return info


def select_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class QwenPlan:
    device: str
    dtype: str
    quantization: str | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# Qwen2.5-3B has ~3.09B parameters: ~6.2 GB of weights in fp16, ~2.0 GB in 4-bit NF4.
QWEN_FP16_GB = 6.2
QWEN_4BIT_GB = 2.0
QWEN_HEADROOM_GB = 1.0


def plan_qwen(hw: HardwareInfo, bnb_available: bool) -> QwenPlan:
    if hw.cuda_available and hw.gpu_free_vram_gb is not None:
        bf16_ok = hw.compute_capability is not None and float(hw.compute_capability) >= 8.0
        half = "bfloat16" if bf16_ok else "float16"
        if hw.gpu_free_vram_gb >= QWEN_FP16_GB + QWEN_HEADROOM_GB:
            return QwenPlan("cuda", half, None, f"{hw.gpu_free_vram_gb} GB free VRAM fits fp16 weights")
        if bnb_available and hw.gpu_free_vram_gb >= QWEN_4BIT_GB + QWEN_HEADROOM_GB:
            return QwenPlan(
                "cuda",
                half,
                "bnb-4bit-nf4",
                f"{hw.gpu_free_vram_gb} GB free VRAM < {QWEN_FP16_GB + QWEN_HEADROOM_GB} GB needed for fp16; "
                "using 4-bit NF4 quantization",
            )
    if hw.system_ram_available_gb >= 13.0:
        return QwenPlan("cpu", "float32", None, "no suitable GPU; CPU float32 (slow)")
    if hw.system_ram_available_gb >= 7.5:
        return QwenPlan("cpu", "bfloat16", None, "no suitable GPU; CPU bfloat16 (slow)")
    return QwenPlan("none", "none", None, "insufficient GPU VRAM and system RAM; template fallback")
