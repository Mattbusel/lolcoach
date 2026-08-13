"""Grounded local inference for the ``lolcoach chat`` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .rag import LocalRAG, RagHit


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    sources: list[RagHit]


class LocalCoach:
    """Qwen adapter plus local RAG context, loaded once for an interactive session."""

    def __init__(self, cfg: Config, *, adapter: Path | None = None) -> None:
        self.cfg = cfg
        self.rag = LocalRAG(cfg)
        self.model, self.tokenizer = _load_adapter(cfg, adapter or cfg.paths.models / cfg.train.output_name / "adapter")

    def _prompt(self, question: str, patch: str | None, top_k: int | None,
                max_context_chars: int = 3000,
                system: str | None = None) -> tuple[Any, list[RagHit]]:
        """Build the tokenized prompt and return it with its retrieval hits.

        Prompt length is the dominant cost here. A 7B model in 4-bit on a
        12 GB card prefills a few thousand tokens slowly, and the default
        12,000-character retrieval block turned every question into a
        multi-minute wait that looked like a hang. Match questions already
        carry the evidence that matters in their moment context, so the
        retrieved block is kept deliberately small.
        """
        import torch

        context, hits = self.rag.context_for(
            question, top_k=top_k or 4, patch=patch, max_chars=max_context_chars)
        prompt = (
            "Use the retrieved local evidence below. If it is incomplete, say what cannot be known from the evidence. "
            "Do not invent exact cooldowns, ward locations, or enemy positions.\n\n"
            f"Retrieved evidence:\n{context or '(No relevant local evidence was found.)'}\n\nQuestion: {question}"
        )
        # The persona changes the voice only. The evidence rules above are part
        # of the user turn, so no tone can talk the model out of them.
        messages = [{"role": "system",
                     "content": system or "You are a practical League of Legends coach."},
                    {"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=False).to(self.model.device)
        return inputs, hits

    def stream(self, question: str, *, patch: str | None = None,
               top_k: int | None = None, max_new_tokens: int = 320,
               max_context_chars: int = 3000, system: str | None = None):
        """Yield the answer incrementally as the model produces it.

        A 7B model takes roughly half a minute to finish an answer on a
        12 GB card. Streaming turns that from a frozen UI into readable
        output arriving at reading speed, which is the single biggest
        perceived-latency win available here.

        Generation runs on a worker thread because ``generate`` blocks; the
        streamer hands tokens back to this generator as they are produced.
        """
        import threading

        import torch
        from transformers import TextIteratorStreamer

        inputs, _hits = self._prompt(question, patch, top_k, max_context_chars,
                                     system=system)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            inputs=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)

        worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        worker.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        worker.join(timeout=1.0)

    def ask(self, question: str, *, patch: str | None = None, top_k: int | None = None, max_new_tokens: int = 600) -> ChatAnswer:
        context, hits = self.rag.context_for(question, top_k=top_k, patch=patch)
        prompt = (
            "Use the retrieved local evidence below. If it is incomplete, say what cannot be known from the evidence. "
            "Do not invent exact cooldowns, ward locations, or enemy positions.\n\n"
            f"Retrieved evidence:\n{context or '(No relevant local evidence was found.)'}\n\nQuestion: {question}"
        )
        messages = [{"role": "system", "content": "You are a practical League of Legends coach."}, {"role": "user", "content": prompt}]
        # transformers 5 returns a BatchEncoding unless return_dict is off, and
        # the slicing below indexes the tensor directly.
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=False).to(self.model.device)
        # Qwen reuses the eos token as its pad token, so transformers cannot
        # infer the mask and warns that generation may misbehave. The prompt is
        # a single unpadded sequence, so the mask is all ones.
        import torch

        attention_mask = torch.ones_like(inputs)
        generated = self.model.generate(
            inputs, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(generated[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        return ChatAnswer(answer, hits)


def _load_adapter(cfg: Config, adapter: Path) -> tuple[Any, Any]:
    if not adapter.is_dir():
        raise RuntimeError(f"Local adapter is absent at {adapter}. Run `lolcoach train` first.")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__ as transformers_version
    except ImportError as exc:
        raise RuntimeError("Local chat requires PyTorch, Transformers, and PEFT.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if torch.cuda.is_available():
        dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        kwargs[dtype_key] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(cfg.train.base_model, **kwargs)
    return PeftModel.from_pretrained(base, str(adapter)).eval(), tokenizer
