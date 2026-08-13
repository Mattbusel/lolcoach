"""GPU-aware, conservative QLoRA hyperparameter selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainConfig


@dataclass(frozen=True)
class TrainingPlan:
    gpu_name: str
    memory_gib: float
    max_seq_len: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    precision: str

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


def choose_training_plan(cfg: TrainConfig) -> TrainingPlan:
    """Choose a stable single-GPU plan, respecting explicit config values."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Install project dependencies first.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. QLoRA training is configured for an NVIDIA 4070/4090-class GPU.")
    device = torch.cuda.get_device_properties(0)
    memory = device.total_memory / 1024**3
    # Keep 20--25% VRAM headroom for CUDA allocator fragmentation and long examples.
    if memory >= 22:
        defaults = dict(max_seq_len=4096, batch_size=2, grad_accum=8, learning_rate=1.5e-4, lora_r=32, lora_alpha=64)
    elif memory >= 15:
        defaults = dict(max_seq_len=3072, batch_size=1, grad_accum=16, learning_rate=1.8e-4, lora_r=24, lora_alpha=48)
    elif memory >= 10:
        # 12 GB cards running a 7B model in 4-bit sit close to the limit. On
        # Windows the driver does not OOM when VRAM is oversubscribed -- it
        # silently pages to system RAM, and step time collapses from ~7s to
        # ~40s with GPU utilisation stuck near 30%. A shorter sequence keeps
        # the whole step resident, which is far faster than a longer one that
        # spills.
        defaults = dict(max_seq_len=1024, batch_size=1, grad_accum=16, learning_rate=2.0e-4, lora_r=16, lora_alpha=32)
    else:
        defaults = dict(max_seq_len=1536, batch_size=1, grad_accum=24, learning_rate=2.0e-4, lora_r=16, lora_alpha=32)
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return TrainingPlan(
        gpu_name=device.name, memory_gib=round(memory, 1),
        max_seq_len=cfg.max_seq_len or defaults["max_seq_len"],
        batch_size=cfg.batch_size or defaults["batch_size"],
        grad_accum=cfg.grad_accum or defaults["grad_accum"],
        learning_rate=cfg.lr or defaults["learning_rate"],
        lora_r=cfg.lora_r or defaults["lora_r"],
        lora_alpha=cfg.lora_alpha or defaults["lora_alpha"],
        precision="bf16" if bf16 else "fp16",
    )
