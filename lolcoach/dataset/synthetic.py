"""Stage 3: synthetic coaching explanations from a stronger teacher model.

The Stage-2 generators produce correct but mechanical answers: they state the
decision and the evidence, in the phrasing of the rule that produced them. A
teacher model turns the same grounded evidence into the kind of explanation a
good coach actually gives -- with the causal chain spelled out and the
alternative addressed.

The critical design choice is that the teacher is **never asked what
happened**. It is given the facts extracted from the timeline (position, gold,
health, wave estimate, what followed) and asked only to explain them. That
keeps hallucination out of the training data: the model cannot invent a dragon
that never died, because it is not the one supplying the events.

Every generated example is then validated before it is kept:

* it must not contradict the recommended action it was asked to explain;
* it must not name champions that are absent from the prompt context;
* it must not invent numbers that do not appear in the context;
* it must be long enough to contain reasoning and short enough to be an answer.

Backends
--------
``anthropic``/``openai``/``deepseek`` use their HTTP APIs and need the matching
key. ``local`` runs a model through transformers on the GPU, which needs no
key and is the default fallback so the stage is runnable offline. ``none``
disables the stage.

The output is JSONL and the run is resumable: completed keys are read back
from the existing file, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..config import Config
from ..logging_utils import get_logger
from ..storage import Database
from . import generators as gen

log = get_logger("dataset.synthetic")

SYSTEM = (
    "You are a top-level League of Legends coach explaining a specific in-game "
    "decision to a player who wants to improve.\n"
    "You will be given facts extracted from a match timeline and the decision "
    "that the analysis recommends.\n"
    "Rules you must follow:\n"
    "1. Explain WHY the recommended decision is right, in two to five sentences.\n"
    "2. Use only the facts given. Never invent champions, items, timers, or "
    "events that are not in the context.\n"
    "3. Name the concrete trade-off: what the player gives up by doing this, "
    "and what goes wrong if they do the opposite.\n"
    "4. Write plainly, the way a coach talks. No headings, no bullet points, "
    "no preamble like 'Great question'.\n"
    "5. If the context says an estimate has low confidence, say what the player "
    "should check on screen instead of asserting it as fact."
)

#: Archetypes worth spending teacher tokens on, in priority order. Wave and
#: death explanations benefit most; static knowledge does not need a teacher.
TEACHABLE = ("wave_decision", "recall", "death_review", "objective_timing",
             "rotation", "teamfight", "trade_window", "jungle_prediction")


@dataclass
class SyntheticResult:
    """Outcome of one synthetic generation run."""

    requested: int = 0
    generated: int = 0
    rejected: int = 0
    resumed: int = 0
    failed: int = 0
    output_path: Path | None = None
    backend: str = ""
    model: str = ""
    skipped: str = ""
    reject_reasons: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TeacherBackend(ABC):
    """Common interface for a model that writes explanations."""

    name = "base"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        """Generate a grounded coaching explanation from the supplied turns."""
        raise RuntimeError("TeacherBackend is abstract; use a configured concrete backend")

    def close(self) -> None:
        """Release any resources (GPU memory for the local backend)."""


class AnthropicBackend(TeacherBackend):
    """Claude via the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(block.text for block in resp.content
                       if getattr(block, "type", "") == "text").strip()


class OpenAICompatibleBackend(TeacherBackend):
    """OpenAI and DeepSeek both speak the chat-completions protocol."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None,
                 name: str = "openai") -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
            else OpenAI(api_key=api_key)
        self.model = model
        self.name = name

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return (resp.choices[0].message.content or "").strip()


class LocalBackend(TeacherBackend):
    """A local instruction model through transformers.

    This exists so Stage 3 is runnable with no API key at all. It is slower
    per example than an API and the explanations are weaker, but it is real
    generation on real grounded prompts rather than a stub.

    Generation is serialised with a lock because a single GPU model cannot be
    called concurrently from several threads safely.
    """

    name = "local"

    def __init__(self, model_id: str, max_seq_len: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs: dict[str, Any] = {
            dtype_key: dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.max_seq_len = max_seq_len
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=self.max_seq_len)
        with self._lock:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id)
            generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def build_backend(cfg: Config) -> tuple[TeacherBackend | None, str]:
    """Construct the configured teacher, or explain why it cannot be built."""
    backend = (cfg.dataset.synthetic_backend or "none").lower()
    model = cfg.dataset.synthetic_model

    if backend == "none":
        return None, "synthetic generation disabled in config"

    if backend == "anthropic":
        key = cfg.secret("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
        if not key:
            return None, "ANTHROPIC_API_KEY is not set"
        return AnthropicBackend(key, model), ""

    if backend == "openai":
        key = cfg.secret("OPENAI_API_KEY")
        if not key:
            return None, "OPENAI_API_KEY is not set"
        return OpenAICompatibleBackend(key, model, name="openai"), ""

    if backend == "deepseek":
        key = cfg.secret("DEEPSEEK_API_KEY")
        if not key:
            return None, "DEEPSEEK_API_KEY is not set"
        return OpenAICompatibleBackend(key, model, base_url="https://api.deepseek.com",
                                       name="deepseek"), ""

    if backend == "local":
        model_id = model if "/" in model else cfg.train.base_model
        try:
            return LocalBackend(model_id), ""
        except Exception as exc:
            return None, f"local teacher unavailable: {type(exc).__name__}: {exc}"

    return None, f"unknown synthetic backend '{backend}'"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\b\d[\d,]*\b")
#: Phrases that mean the teacher answered the meta-question instead of coaching.
_META = ("as an ai", "i cannot", "i don't have enough", "the context does not",
         "based on the information provided, i")

ACTION_CONTRADICTIONS = {
    "FREEZE": ("crash the wave", "shove it in", "push it in"),
    "CRASH": ("hold the freeze", "set up a freeze", "freeze it"),
    "SLOW_PUSH": ("crash it now", "hold the freeze"),
    "RESET": ("stay in lane", "do not back"),
}


def validate(answer: str, context: str, action: str,
             known_champions: set[str]) -> tuple[bool, str]:
    """Check a generated explanation against its own prompt.

    Returns ``(ok, reason_if_rejected)``. The checks are deliberately cheap and
    conservative: they catch the failure modes that would poison training data
    (contradicting the label, naming champions that were never mentioned,
    inventing numbers) without trying to judge writing quality.
    """
    text = answer.strip()
    if len(text) < 80:
        return False, "too_short"
    if len(text) > 2200:
        return False, "too_long"

    lowered = text.lower()
    for marker in _META:
        if marker in lowered:
            return False, "meta_response"

    for bad in ACTION_CONTRADICTIONS.get(action, ()):
        if bad in lowered:
            return False, "contradicts_action"

    # Champions named in the answer must appear in the context.
    context_lower = context.lower()
    for champ in known_champions:
        if len(champ) < 4:
            continue        # skip short names that collide with English words
        if champ.lower() in lowered and champ.lower() not in context_lower:
            return False, "invented_champion"

    # Large numbers should come from the context, not from the model.
    context_numbers = {n.replace(",", "") for n in _NUMBER.findall(context)}
    for token in _NUMBER.findall(text):
        value = token.replace(",", "")
        if len(value) >= 3 and value not in context_numbers:
            # Percentages and small round numbers are normal coaching language.
            if value.endswith("00") or int(value) <= 100:
                continue
            return False, "invented_number"

    return True, ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _teacher_prompt(context_and_question: str, baseline: str) -> str:
    """Assemble the user turn for the teacher model."""
    return (
        f"{context_and_question}\n\n"
        f"The analysis of this timeline recommends the following, and it is "
        f"correct:\n{baseline}\n\n"
        f"Write the coaching explanation for that recommendation."
    )


def iter_candidates(db: Database, cfg: Config,
                    archetypes: Sequence[str] = TEACHABLE) -> Iterator[dict[str, Any]]:
    """Yield grounded prompts worth sending to the teacher.

    Candidates are drawn from the same generators the deterministic dataset
    uses, so the teacher sees exactly the evidence the pipeline trusts.
    """
    streams = {
        "wave_decision": lambda: gen.wave_examples(db, cfg.dataset.min_wave_confidence),
        "trade_window": lambda: gen.trade_examples(db),
        "recall": lambda: gen.recall_examples(db),
        "jungle_prediction": lambda: gen.jungle_examples(db),
        "objective_timing": lambda: gen.objective_examples(db),
        "death_review": lambda: gen.death_examples(db),
        "rotation": lambda: gen.rotation_examples(db),
        "teamfight": lambda: gen.fight_examples(db),
    }
    wanted = [a for a in archetypes if a in streams]
    for archetype in wanted:
        for row in streams[archetype]():
            got_archetype, key, user, assistant, confidence, match_id, patch = row
            if got_archetype not in wanted:
                continue
            yield {
                "archetype": got_archetype, "key": key, "user": user,
                "baseline": assistant, "confidence": float(confidence),
                "match_id": match_id, "patch": patch,
            }


def load_completed(path: Path) -> set[str]:
    """Keys already present in an output file, for resumption."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            key = (record.get("metadata") or {}).get("key")
            if key:
                done.add(key)
    return done


def generate_synthetic(cfg: Config, limit: int | None = None,
                       archetypes: Sequence[str] = TEACHABLE,
                       progress: Callable[[int, int], None] | None = None
                       ) -> SyntheticResult:
    """Run Stage 3 and append validated examples to ``synthetic.jsonl``."""
    result = SyntheticResult()
    backend, why = build_backend(cfg)
    if backend is None:
        result.skipped = why
        log.info("synthetic generation skipped: %s", why)
        return result

    result.backend = backend.name
    result.model = cfg.dataset.synthetic_model
    db = Database(cfg.paths.db)
    out_path = cfg.paths.datasets / "synthetic.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_path = out_path

    completed = load_completed(out_path)
    result.resumed = len(completed)
    target = limit or cfg.dataset.synthetic_max_examples

    champions = {str(r[0]) for r in db.query(
        "SELECT name FROM champions WHERE name IS NOT NULL")}

    # Collect the work list first so progress reporting is meaningful and the
    # per-archetype mix stays balanced rather than exhausting one stream.
    candidates: list[dict[str, Any]] = []
    per_archetype: dict[str, int] = {}
    cap = max(1, target // max(1, len(archetypes)))
    for cand in iter_candidates(db, cfg, archetypes):
        if cand["key"] in completed:
            continue
        seen = per_archetype.get(cand["archetype"], 0)
        if seen >= cap:
            continue
        per_archetype[cand["archetype"]] = seen + 1
        candidates.append(cand)
        if len(candidates) >= target:
            break

    result.requested = len(candidates)
    if not candidates:
        log.info("no new candidates for synthetic generation")
        backend.close()
        return result

    log.info("generating %d synthetic explanations with %s (%s)",
             len(candidates), result.backend, result.model)

    write_lock = threading.Lock()
    reasons: dict[str, int] = {}

    def work(cand: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
        action = _action_of(cand["baseline"])
        prompt = _teacher_prompt(cand["user"], cand["baseline"])
        try:
            answer = backend.complete(SYSTEM, prompt)
        except Exception as exc:
            return False, f"backend_error:{type(exc).__name__}", None
        ok, reason = validate(answer, cand["user"], action, champions)
        if not ok:
            return False, reason, None
        record = {
            "messages": [
                {"role": "system", "content":
                 "You are a League of Legends coach. Give an actionable, concise "
                 "answer, distinguish known facts from uncertainty, and explain "
                 "the relevant tradeoff."},
                {"role": "user", "content": cand["user"]},
                {"role": "assistant", "content": " ".join(answer.split())},
            ],
            "metadata": {
                "archetype": cand["archetype"], "key": cand["key"],
                "source": "synthetic", "teacher": f"{result.backend}:{result.model}",
                "match_id": cand["match_id"], "patch": cand["patch"],
                "confidence": round(cand["confidence"], 3),
                "generated_at": time.time(),
            },
        }
        return True, "", record

    # The local backend serialises internally, so extra threads only add
    # contention; API backends benefit from concurrency.
    workers = 1 if backend.name == "local" else max(1, cfg.dataset.synthetic_concurrency)

    with out_path.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, c): c for c in candidates}
            for done, fut in enumerate(as_completed(futures), 1):
                try:
                    ok, reason, record = fut.result()
                except Exception as exc:
                    result.failed += 1
                    log.debug("synthetic worker failed: %s", exc)
                    continue
                if ok and record is not None:
                    with write_lock:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                    result.generated += 1
                else:
                    if reason.startswith("backend_error"):
                        result.failed += 1
                    else:
                        result.rejected += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                if progress is not None:
                    progress(done, len(candidates))

    result.reject_reasons = reasons
    backend.close()
    db.set_meta("synthetic_latest", {
        "at": time.time(), "generated": result.generated,
        "rejected": result.rejected, "backend": result.backend,
    })
    log.info("synthetic generation finished: %d kept, %d rejected, %d failed",
             result.generated, result.rejected, result.failed)
    return result


def _action_of(baseline: str) -> str:
    """Recover the recommended action from a baseline answer's opening."""
    lowered = baseline.lower()
    if lowered.startswith("crash and reset") or "crash it and back" in lowered:
        return "RESET"
    if lowered.startswith("crash it"):
        return "CRASH"
    if lowered.startswith("hold the freeze"):
        return "FREEZE"
    if lowered.startswith("keep the slow push"):
        return "SLOW_PUSH"
    return ""
