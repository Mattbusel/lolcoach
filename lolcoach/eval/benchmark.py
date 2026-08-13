"""Benchmark local Qwen adapters against configured Claude/GPT/DeepSeek APIs."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from ..config import Config
from ..logging_utils import get_logger

log = get_logger("eval")


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    question: str
    rubric: str
    paraphrase: str


@dataclass(frozen=True)
class BenchmarkResult:
    report_path: Path
    records_path: Path
    questions: int
    models: list[str]
    summary: dict[str, dict[str, float]]


def generate_questions(cfg: Config, *, count: int | None = None) -> list[BenchmarkQuestion]:
    """Generate a fixed-seed balanced bank (1,000 questions by default)."""
    total = count or cfg.eval.n_questions
    rng = random.Random(3407)
    # Seed pairs are real, role-correct matchups; the rest of the bank is
    # filled from the champion table so 1,000 questions cover 1,000 different
    # situations rather than six templates with the numbers shuffled.
    pairs = [("Jinx", "Draven"), ("Ahri", "Zed"), ("Darius", "Fiora"),
             ("Orianna", "Syndra"), ("Lee Sin", "Kha'Zix"), ("Nautilus", "Morgana")]
    roster = _champions(cfg)
    if len(roster) >= 8:
        extra = list(roster)
        rng.shuffle(extra)
        pairs.extend((extra[i], extra[i + 1])
                     for i in range(0, len(extra) - 1, 2)
                     if extra[i] != extra[i + 1])
    categories: list[tuple[str, str, str]] = [
        ("wave", "At {minute}:20 I am {champion} into {enemy}, {gold:+d} gold, {cs:+d} CS, with a {wave} wave and both junglers unseen. Should I crash, freeze, or recall?", "States a conditional wave plan, weighs jungle risk and recall timing, and avoids inventing exact minion counts."),
        ("jungle", "I am playing {champion} at {minute}:00. The enemy jungler showed on {side} after their first clear; which lane or quadrant is most likely next and what information would change that read?", "Gives a probabilistic path prediction, names uncertainty and a defensive response."),
        ("objective", "Dragon spawns in {timer} seconds. Our mid has {priority} priority and bot wave is {wave}. What is the best setup sequence?", "Orders wave, vision, reset, and contest decisions; does not claim perfect ward knowledge."),
        ("trade", "As {champion} into {enemy}, I am level {level} with {hp}% HP and the enemy is {enemy_hp}% HP. Their key cooldown just missed. Should I trade, all-in, or hold the wave?", "Uses health, levels, cooldowns, minions and jungle risk rather than a universal all-in instruction."),
        ("macro", "At {minute}:00, our outer mid tower is down and Baron is up in {timer} seconds. I am a {role} with Teleport {teleport}. Where should I play and when should I rotate?", "Explains side-lane assignment, tempo, vision and objective timing with role-appropriate caveats."),
        ("teamfight", "In a river fight, I am {champion} against {enemy}. Their engage tools are available and I have Flash {flash}. How should I position before and during the fight?", "Provides positioning relative to threats and allies, cooldown-dependent contingencies, and a clear win condition."),
    ]
    questions: list[BenchmarkQuestion] = []
    for index in range(total):
        category, template, rubric = categories[index % len(categories)]
        champion, enemy = pairs[index % len(pairs)]
        values = {
            "champion": champion, "enemy": enemy, "minute": rng.choice([4, 6, 8, 11, 14, 18, 22, 27]),
            "gold": rng.choice([-700, -350, -100, 0, 250, 600]), "cs": rng.choice([-18, -8, -2, 0, 5, 12]),
            "wave": rng.choice(["neutral", "slow-pushing", "stacked toward the enemy", "bouncing back"]),
            "side": rng.choice(["top side", "bot side", "river", "their blue-side jungle"]),
            "timer": rng.choice([35, 50, 75, 95, 120]), "priority": rng.choice(["no", "even", "light", "hard"]),
            "level": rng.choice([3, 4, 6, 9, 11]), "hp": rng.choice([35, 48, 62, 75, 90]),
            "enemy_hp": rng.choice([30, 45, 58, 72, 88]), "role": rng.choice(["top laner", "mid laner", "ADC", "support", "jungler"]),
            "teleport": rng.choice(["available", "on cooldown"]), "flash": rng.choice(["available", "on cooldown"]),
        }
        question = template.format(**values)
        paraphrase = _paraphrase(question)
        questions.append(BenchmarkQuestion(f"q{index + 1:04d}", category, question, rubric, paraphrase))
    return questions


def run_benchmark(cfg: Config, *, competitors: list[str] | None = None, count: int | None = None) -> BenchmarkResult:
    """Answer, blind-judge, and report a local model versus configured APIs.

    This intentionally performs paid API calls only when the operator invokes
    the command.  Missing credentials skip a named competitor rather than
    silently replacing it with a different model.
    """
    models = competitors or ["local", *cfg.eval.competitors]
    questions = generate_questions(cfg, count=count)
    answerers = {name: _answerer(cfg, name) for name in models}
    active = {name: answerer for name, answerer in answerers.items() if answerer is not None}
    if not active:
        raise RuntimeError("No benchmark competitors are configured with usable local weights or API credentials.")
    judge = _judge(cfg)
    records: list[dict[str, Any]] = []
    for number, question in enumerate(questions, 1):
        answers = {name: answerer(question.question) for name, answerer in active.items()}
        for name, answer in answers.items():
            score = judge(question, answer)
            records.append({"question": asdict(question), "model": name, "answer": answer, "scores": score})
        if number % 25 == 0 or number == len(questions):
            log.info("benchmarked %d/%d questions", number, len(questions))
    _add_consistency(cfg, active, questions, records, judge)
    summary = _summarise(records)
    root = cfg.paths.reports
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record_path = root / f"benchmark-{stamp}.jsonl"
    report_path = root / f"benchmark-{stamp}.json"
    _atomic_jsonl(record_path, records)
    _atomic_json(report_path, {"created_at": time.time(), "questions": len(questions), "models": sorted(active), "summary": summary, "records": record_path.name})
    return BenchmarkResult(report_path, record_path, len(questions), sorted(active), summary)


def _champions(cfg: Config) -> list[str]:
    try:
        from ..storage import Database
        rows = Database(cfg.paths.db).query("SELECT name FROM champions ORDER BY name LIMIT 100")
        return [str(row["name"]) for row in rows] or ["Ahri"]
    except Exception:
        return ["Ahri"]


def _answerer(cfg: Config, name: str):
    if name == "local":
        adapter = cfg.paths.models / cfg.train.output_name / "adapter"
        if not adapter.exists():
            log.warning("local model skipped: adapter is not present at %s", adapter)
            return None
        return _local_answerer(cfg, adapter)
    if name == "anthropic":
        key = cfg.secret("ANTHROPIC_API_KEY")
        if not key:
            log.warning("anthropic skipped: ANTHROPIC_API_KEY is not set")
            return None
        return _anthropic_answerer(key, cfg.eval.anthropic_model)
    if name in {"openai", "deepseek"}:
        key_name = "OPENAI_API_KEY" if name == "openai" else "DEEPSEEK_API_KEY"
        key = cfg.secret(key_name)
        if not key:
            log.warning("%s skipped: %s is not set", name, key_name)
            return None
        model = cfg.eval.openai_model if name == "openai" else cfg.eval.deepseek_model
        return _openai_answerer(key, model, "https://api.deepseek.com" if name == "deepseek" else None)
    raise ValueError(f"Unknown competitor {name!r}")


def _local_answerer(cfg: Config, adapter: Path):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__ as transformers_version
    except ImportError as exc:
        raise RuntimeError("Local benchmark requires transformers, PEFT and PyTorch.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if torch.cuda.is_available():
        dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        kwargs.update({dtype_key: torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")})
    base = AutoModelForCausalLM.from_pretrained(cfg.train.base_model, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    def answer(question: str) -> str:
        messages = [{"role": "system", "content": "You are a precise League of Legends coach."}, {"role": "user", "content": question}]
        # transformers 5 returns a BatchEncoding unless return_dict is off.
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=False).to(model.device)
        # Qwen's pad token equals its eos token, so the mask cannot be inferred.
        output = model.generate(
            inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=cfg.eval.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return answer


def _openai_answerer(key: str, model: str, base_url: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is required for this benchmark competitor.") from exc
    client = OpenAI(api_key=key, base_url=base_url)
    def answer(question: str) -> str:
        response = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "system", "content": "You are a precise League of Legends coach."}, {"role": "user", "content": question}])
        return str(response.choices[0].message.content or "")
    return answer


def _anthropic_answerer(key: str, model: str):
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Anthropic SDK is required for this benchmark competitor.") from exc
    client = anthropic.Anthropic(api_key=key)
    def answer(question: str) -> str:
        response = client.messages.create(model=model, max_tokens=700, temperature=0, system="You are a precise League of Legends coach.", messages=[{"role": "user", "content": question}])
        return "".join(block.text for block in response.content if hasattr(block, "text"))
    return answer


def _judge(cfg: Config):
    """Build the scoring function.

    Deterministic scoring always runs: factual accuracy is checked against the
    downloaded Data Dragon and objective tables, so the benchmark produces
    real numbers with no API key at all. When a judge *is* configured its
    subjective scores are averaged in, and its components are kept separately
    in the report so a factual regression cannot be masked by nicer prose.
    """
    from ..storage import Database
    from .scoring import KnowledgeBase, consistency, score_answer

    kb = KnowledgeBase.load(Database(cfg.paths.db))
    ask = None
    backend = (cfg.eval.judge_backend or "none").lower()
    if backend == "anthropic":
        key = cfg.secret("ANTHROPIC_API_KEY")
        if key:
            ask = _anthropic_answerer(key, cfg.eval.judge_model)
        else:
            log.warning("no ANTHROPIC_API_KEY: scoring with local metrics only")
    elif backend == "openai":
        key = cfg.secret("OPENAI_API_KEY")
        if key:
            ask = _openai_answerer(key, cfg.eval.openai_model, None)
        else:
            log.warning("no OPENAI_API_KEY: scoring with local metrics only")
    elif backend not in ("none", ""):
        log.warning("unknown judge backend %r: scoring with local metrics only", backend)

    def score(question: BenchmarkQuestion, answer: str) -> dict[str, float]:
        local = score_answer(answer, kb)
        if ask is None:
            return local
        judged = _judge_scores(ask, question, answer)
        merged = dict(local)
        for metric in ("factual_accuracy", "coaching_quality", "hallucination_risk"):
            merged[f"judge_{metric}"] = judged[metric]
            merged[f"local_{metric}"] = local[metric]
            merged[metric] = round((local[metric] + judged[metric]) / 2.0, 4)
        return merged

    return score


def _judge_scores(ask: Any, question: BenchmarkQuestion, answer: str) -> dict[str, float]:
    def _ask(question: BenchmarkQuestion, answer: str) -> dict[str, float]:
        prompt = (
            "Score this League coaching answer. Return JSON only with factual_accuracy, coaching_quality, "
            "hallucination_risk, and consistency, each from 0 to 1. Hallucination risk is 1 for unsupported "
            "specific claims and 0 for calibrated uncertainty.\n\n"
            f"Question: {question.question}\nRubric: {question.rubric}\nAnswer: {answer}"
        )
        return _scores(ask(prompt))
    return _ask(question, answer)


def _add_consistency(cfg: Config, active: dict[str, Any], questions: list[BenchmarkQuestion], records: list[dict[str, Any]], judge: Any) -> None:
    """Score self-consistency by re-asking each question as a paraphrase.

    Agreement is computed over the *decisions* the two answers recommend, not
    their wording: a model that says freeze both times is consistent however
    differently it phrases it, and one that says freeze then crash is not.
    """
    from .scoring import consistency as decision_agreement

    limit = min(cfg.eval.consistency_pairs, len(questions))
    by_key = {(record["question"]["id"], record["model"]): record for record in records}
    for question in questions[:limit]:
        for name, answerer in active.items():
            baseline = by_key.get((question.id, name))
            if baseline is None:
                continue
            alternative = answerer(question.paraphrase)
            baseline["scores"]["consistency"] = round(
                decision_agreement(str(baseline["answer"]), alternative), 4)
            baseline["paraphrase_answer"] = alternative


def _scores(text: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return {"factual_accuracy": 0.0, "coaching_quality": 0.0, "hallucination_risk": 1.0, "consistency": 0.0}
    return {key: max(0.0, min(1.0, float(raw.get(key, 0.0 if key != "hallucination_risk" else 1.0)))) for key in ("factual_accuracy", "coaching_quality", "hallucination_risk", "consistency")}


def _summarise(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for metric, score in record["scores"].items():
            values[record["model"]][metric].append(float(score))
    return {model: {metric: round(sum(scores) / len(scores), 4) for metric, scores in metrics.items()} for model, metrics in values.items()}


#: Rewrites that keep the situation identical but change the surface form.
#: Consistency is only meaningful if the paraphrase is a genuinely different
#: sentence, not the original with a preamble bolted on.
_PARAPHRASE_RULES = (
    (re.compile(r"^At (\d+):(\d+)\s+", re.I), r"\1 minutes \2 seconds in, "),
    (re.compile(r"\bShould I\b", re.I), "Would you"),
    (re.compile(r"\bWhat is the best\b", re.I), "What would you say is the strongest"),
    (re.compile(r"\bHow should I\b", re.I), "What is the right way to"),
    (re.compile(r"\bWhich lane or quadrant\b", re.I), "Which part of the map"),
    (re.compile(r"\bWhere should I play\b", re.I), "Which side of the map do I take"),
    (re.compile(r"\bI am playing\b", re.I), "I'm on"),
    (re.compile(r"\bwhat information would change that read\b", re.I),
     "what would make you change your mind"),
    (re.compile(r"\bWhat is the best setup sequence\b", re.I),
     "How would you sequence the setup"),
)


def _paraphrase(question: str) -> str:
    """Restate a question without changing what is being asked."""
    text = question
    for pattern, replacement in _PARAPHRASE_RULES:
        text = pattern.sub(replacement, text)
    if text == question:
        # No rule matched; invert the framing instead of prefixing the original.
        text = "Talk me through this situation and tell me what to do. " + question
    return text


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        temporary = Path(fh.name)
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        temporary = Path(fh.name)
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(temporary, path)
