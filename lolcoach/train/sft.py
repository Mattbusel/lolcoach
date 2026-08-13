"""Resumable supervised QLoRA fine-tuning with assistant-only loss masking."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from .hparams import TrainingPlan, choose_training_plan

log = get_logger("train")


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    checkpoint: str | None
    examples: int
    validation_examples: int
    plan: TrainingPlan
    used_unsloth: bool


def train(cfg: Config, *, resume_from: str | None = None) -> TrainResult:
    """Fine-tune ``cfg.train.base_model`` on the prepared chat JSONL corpus.

    Checkpoints are written at every epoch and training resumes from the last
    valid checkpoint by default.  Only assistant tokens receive loss, so the
    model learns answers rather than memorising the prompts themselves.
    """
    plan = choose_training_plan(cfg.train)
    _seed(cfg.train.seed)
    train_path = cfg.paths.datasets / "train.jsonl"
    validation_path = cfg.paths.datasets / "validation.jsonl"
    if not train_path.is_file() or train_path.stat().st_size == 0:
        raise RuntimeError("Training dataset is absent or empty. Run `lolcoach prepare_dataset` first.")
    try:
        from datasets import load_dataset
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("datasets and transformers are required for training. Install project dependencies first.") from exc
    data_files: dict[str, str] = {"train": str(train_path)}
    if validation_path.is_file() and validation_path.stat().st_size:
        data_files["validation"] = str(validation_path)
    raw = load_dataset("json", data_files=data_files)
    model, tokenizer, used_unsloth = _load_model(cfg, plan)
    train_set = raw["train"].map(
        lambda record: _tokenize_record(record, tokenizer, plan.max_seq_len),
        remove_columns=raw["train"].column_names, desc="tokenizing train examples",
    )
    validation_set = None
    if "validation" in raw and len(raw["validation"]):
        validation_set = raw["validation"].map(
            lambda record: _tokenize_record(record, tokenizer, plan.max_seq_len),
            remove_columns=raw["validation"].column_names, desc="tokenizing validation examples",
        )
    output = cfg.paths.models / cfg.train.output_name
    output.mkdir(parents=True, exist_ok=True)
    # ``warmup_ratio`` is deprecated in transformers 5 in favour of an explicit
    # step count, so the ratio is resolved against the real schedule here.
    steps_per_epoch = max(1, math.ceil(
        len(train_set) / max(1, plan.batch_size * plan.grad_accum)))
    total_steps = max(1, int(steps_per_epoch * cfg.train.epochs))
    warmup_steps = max(0, int(total_steps * cfg.train.warmup_ratio))

    # ``overwrite_output_dir`` was removed in transformers 5; checkpoints are
    # kept regardless, which is what resume relies on.
    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=plan.batch_size, per_device_eval_batch_size=plan.batch_size,
        gradient_accumulation_steps=plan.grad_accum, learning_rate=plan.learning_rate,
        warmup_steps=warmup_steps, weight_decay=cfg.train.weight_decay,
        logging_steps=10, save_strategy="epoch" if cfg.train.save_every_epoch else "steps",
        save_steps=500, save_total_limit=3, eval_strategy="epoch" if validation_set is not None else "no",
        bf16=plan.precision == "bf16", fp16=plan.precision == "fp16", tf32=True,
        gradient_checkpointing=cfg.train.gradient_checkpointing, report_to=[],
        seed=cfg.train.seed, data_seed=cfg.train.seed, optim="paged_adamw_8bit" if cfg.train.load_in_4bit else "adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_set, eval_dataset=validation_set,
        data_collator=AssistantDataCollator(tokenizer),
    )
    checkpoint = resume_from or (_last_checkpoint(output) if cfg.train.resume else None)
    log.info("training %s on %s (%.1f GiB, seq=%d, effective batch=%d)%s", cfg.train.base_model, plan.gpu_name, plan.memory_gib, plan.max_seq_len, plan.effective_batch_size, f"; resuming {checkpoint}" if checkpoint else "")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    _write_manifest(output / "training_manifest.json", {
        "created_at": time.time(), "base_model": cfg.train.base_model, "examples": len(train_set),
        "validation_examples": len(validation_set) if validation_set is not None else 0,
        "plan": asdict(plan), "used_unsloth": used_unsloth, "last_checkpoint": _last_checkpoint(output),
    })
    return TrainResult(output, _last_checkpoint(output), len(train_set), len(validation_set) if validation_set is not None else 0, plan, used_unsloth)


def _load_model(cfg: Config, plan: TrainingPlan) -> tuple[Any, Any, bool]:
    if cfg.train.use_unsloth:
        loaded = _try_unsloth(cfg, plan)
        if loaded is not None:
            return (*loaded, True)
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__ as transformers_version
    except ImportError as exc:
        raise RuntimeError("transformers, PEFT, and bitsandbytes are required for QLoRA training.") from exc
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.base_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Transformers 5 renamed ``torch_dtype`` to ``dtype``; retain 4.44+ support.
    dtype = torch.bfloat16 if plan.precision == "bf16" else torch.float16
    dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {"device_map": "auto"}
    kwargs[dtype_key] = dtype
    if cfg.train.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.train.base_model, **kwargs)
    if cfg.train.gradient_checkpointing:
        model.config.use_cache = False
    if cfg.train.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)
    adapter = LoraConfig(r=plan.lora_r, lora_alpha=plan.lora_alpha, lora_dropout=cfg.train.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, adapter), tokenizer, False


def _unsloth_is_usable() -> tuple[bool, str]:
    """Check that Unsloth can actually train here, not merely that it imports.

    Unsloth's fused kernels are compiled by Triton at the first backward pass,
    which needs a working C compiler. On a machine without one, importing and
    loading succeed and training then dies several minutes in with
    ``Failed to find C compiler``. Detecting that up front is the difference
    between falling back cleanly and losing a run.

    Returns ``(usable, reason_if_not)``.
    """
    import importlib.util
    import os
    import shutil

    if importlib.util.find_spec("unsloth") is None:
        return False, "unsloth is not installed"

    # unsloth_zoo pins a torch range and refuses to import outside it.
    try:
        import unsloth_zoo  # noqa: F401
    except Exception as exc:
        return False, f"unsloth_zoo rejected this environment ({exc})"

    compiler = (os.environ.get("CC") or shutil.which("cc") or shutil.which("clang")
                or shutil.which("gcc"))
    if not compiler:
        return False, ("no C compiler on PATH, which Triton needs to build "
                       "Unsloth's fused kernels")
    return True, ""


def _try_unsloth(cfg: Config, plan: TrainingPlan) -> tuple[Any, Any] | None:
    """Use Unsloth only when it is importable *and* able to compile kernels."""
    usable, reason = _unsloth_is_usable()
    if not usable:
        log.info("Unsloth not used (%s); training with PEFT + bitsandbytes QLoRA", reason)
        return None
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        log.info("Unsloth import failed (%s); using standard QLoRA", exc)
        return None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.train.base_model, max_seq_length=plan.max_seq_len,
            load_in_4bit=cfg.train.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=plan.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=plan.lora_alpha, lora_dropout=cfg.train.lora_dropout,
            use_gradient_checkpointing="unsloth" if cfg.train.gradient_checkpointing else False,
            random_state=cfg.train.seed,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as exc:
        log.info("Unsloth unavailable for this configuration; using standard QLoRA (%s)", exc)
        return None


def _tokenize_record(record: Mapping[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    messages = list(record["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each training record must end with an assistant message")
    # ``return_dict`` defaults to True in transformers 5.x, which would make
    # these BatchEncodings rather than the token-id lists this function slices.
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    prefix, full = list(prefix), list(full)

    if len(full) > max_length:
        # Preserve the whole response and trim the prompt from the left, so a
        # long retrieved context never costs us the answer we train on.
        response = full[len(prefix):][-max_length:]
        available_prefix = max(0, max_length - len(response))
        # ``prefix[-0:]`` is the entire list, so an exhausted budget has to be
        # handled explicitly rather than by negative slicing.
        kept_prefix = prefix[len(prefix) - available_prefix:] if available_prefix else []
        full = kept_prefix + response
        prefix_length = len(kept_prefix)
    else:
        prefix_length = len(prefix)
    labels = list(full)
    labels[:prefix_length] = [-100] * prefix_length
    return {"input_ids": list(full), "attention_mask": [1] * len(full), "labels": labels}


class AssistantDataCollator:
    """Pad tokenized records while retaining ``-100`` on non-loss positions."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        import torch
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad = int(self.tokenizer.pad_token_id)
        return {
            "input_ids": torch.tensor([list(f["input_ids"]) + [pad] * (max_length - len(f["input_ids"])) for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([list(f["attention_mask"]) + [0] * (max_length - len(f["attention_mask"])) for f in features], dtype=torch.long),
            "labels": torch.tensor([list(f["labels"]) + [-100] * (max_length - len(f["labels"])) for f in features], dtype=torch.long),
        }


def _last_checkpoint(output: Path) -> str | None:
    checkpoints = [path for path in output.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write_manifest(path: Path, content: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        temporary = Path(fh.name)
        json.dump(content, fh, indent=2)
        fh.write("\n")
    os.replace(temporary, path)
