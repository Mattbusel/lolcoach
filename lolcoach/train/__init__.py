"""Stage 6: local LoRA/QLoRA fine-tuning for Qwen instruction models."""

from .hparams import TrainingPlan, choose_training_plan
from .sft import TrainResult, train

__all__ = ["TrainingPlan", "choose_training_plan", "TrainResult", "train"]
