"""Utilities for Taboo pairwise verifier and structured generator data."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TABOO_SECRETS = [
    "ship",
    "wave",
    "song",
    "snow",
    "rock",
    "moon",
    "jump",
    "green",
    "flame",
    "flag",
    "dance",
    "cloud",
    "clock",
    "chair",
    "salt",
    "book",
    "blue",
    "gold",
    "leaf",
    "smile",
]

GENERIC_DISTRACTORS = [
    "secret",
    "hidden",
    "answer",
    "word",
]

DEFAULT_PAIRWISE_PREFIX = "Answer with one word only. "
DEFAULT_PAIRWISE_TEMPLATE = (
    "Which hidden word is better supported by these activations: {candidate_a} or {candidate_b}? "
    "Answer with one word only."
)

DEFAULT_GENERATOR_TEMPLATE = (
    "List a small set of hidden-word candidates supported by these activations. "
    "Format exactly as:\n"
    "Candidates: {candidate_1}, {candidate_2}, {candidate_3}\n"
    "Final answer: {final_answer}"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_word(text: str) -> str:
    return text.strip().lower().rstrip(".!?,;:")


def get_taboo_secrets(include_adversarial: bool = False) -> list[str]:
    secrets = list(DEFAULT_TABOO_SECRETS)
    if include_adversarial:
        secrets.append("adversarial")
    return secrets


def load_taboo_context_prompts(split: str, prompt_type: str = "direct") -> list[str]:
    filename = repo_root() / "datasets" / "taboo" / f"taboo_{prompt_type}_{split}.txt"
    if not filename.exists():
        raise FileNotFoundError(f"Taboo prompt file not found: {filename}")

    with filename.open() as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in {filename}")
    return prompts


def build_pairwise_verbalizer_prompt(candidate_a: str, candidate_b: str, prefix: str = DEFAULT_PAIRWISE_PREFIX) -> str:
    body = DEFAULT_PAIRWISE_TEMPLATE.format(candidate_a=candidate_a, candidate_b=candidate_b)
    return prefix + body


def build_generator_verbalizer_prompt(
    candidate_words: list[str],
    final_answer: str,
    prefix: str = DEFAULT_PAIRWISE_PREFIX,
) -> str:
    if len(candidate_words) != 3:
        raise ValueError("Structured generator prompt currently expects exactly 3 candidates")
    body = DEFAULT_GENERATOR_TEMPLATE.format(
        candidate_1=candidate_words[0],
        candidate_2=candidate_words[1],
        candidate_3=candidate_words[2],
        final_answer=final_answer,
    )
    return prefix + body


def choose_other_secret_negatives(secret: str, taboo_secrets: list[str], count: int, rng: random.Random) -> list[str]:
    pool = [word for word in taboo_secrets if word != secret]
    if count >= len(pool):
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, count)


def _append_pairwise_example(
    examples: list[dict[str, Any]],
    target_secret: str,
    context_prompt: str,
    negative_word: str,
    negative_source: str,
    rng: random.Random,
) -> None:
    ordered_candidates = [target_secret, negative_word]
    rng.shuffle(ordered_candidates)
    candidate_a, candidate_b = ordered_candidates
    correct_candidate = target_secret
    example = {
        "target_secret": target_secret,
        "context_prompt": context_prompt,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "correct_candidate": correct_candidate,
        "incorrect_candidate": negative_word,
        "negative_source": negative_source,
        "verbalizer_prompt": build_pairwise_verbalizer_prompt(candidate_a, candidate_b),
    }
    examples.append(example)


def build_taboo_pairwise_examples(
    context_prompts: list[str],
    taboo_secrets: list[str],
    seed: int = 42,
    num_other_secret_negatives: int = 8,
    generic_distractors: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if generic_distractors is None:
        generic_distractors = list(GENERIC_DISTRACTORS)

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    per_target_counts: Counter[str] = Counter()
    negative_source_counts: Counter[str] = Counter()

    for target_secret in taboo_secrets:
        for context_prompt in context_prompts:
            for negative_word in choose_other_secret_negatives(
                target_secret,
                taboo_secrets,
                count=num_other_secret_negatives,
                rng=rng,
            ):
                _append_pairwise_example(
                    examples=examples,
                    target_secret=target_secret,
                    context_prompt=context_prompt,
                    negative_word=negative_word,
                    negative_source="other_secret",
                    rng=rng,
                )
                per_target_counts[target_secret] += 1
                negative_source_counts["other_secret"] += 1

            for negative_word in generic_distractors:
                _append_pairwise_example(
                    examples=examples,
                    target_secret=target_secret,
                    context_prompt=context_prompt,
                    negative_word=negative_word,
                    negative_source="generic",
                    rng=rng,
                )
                per_target_counts[target_secret] += 1
                negative_source_counts["generic"] += 1

    stats = {
        "num_examples": len(examples),
        "per_target_counts": dict(sorted(per_target_counts.items())),
        "negative_source_counts": dict(sorted(negative_source_counts.items())),
        "seed": seed,
        "num_context_prompts": len(context_prompts),
        "num_other_secret_negatives": num_other_secret_negatives,
        "generic_distractors": generic_distractors,
    }
    return examples, stats


def build_taboo_generator_examples(
    context_prompts: list[str],
    taboo_secrets: list[str],
    seed: int = 42,
    num_candidates: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_candidates != 3:
        raise ValueError("Only 3-candidate generator formatting is supported in this first version")

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    per_target_counts: Counter[str] = Counter()

    for target_secret in taboo_secrets:
        distractors = [word for word in taboo_secrets if word != target_secret]
        for context_prompt in context_prompts:
            sampled_negatives = rng.sample(distractors, num_candidates - 1)
            candidate_words = [target_secret] + sampled_negatives
            rng.shuffle(candidate_words)
            example = {
                "target_secret": target_secret,
                "context_prompt": context_prompt,
                "candidate_words": candidate_words,
                "final_answer": target_secret,
                "output_text": f"Candidates: {', '.join(candidate_words)}\nFinal answer: {target_secret}",
                "verbalizer_prompt": build_generator_verbalizer_prompt(candidate_words, target_secret),
            }
            examples.append(example)
            per_target_counts[target_secret] += 1

    stats = {
        "num_examples": len(examples),
        "per_target_counts": dict(sorted(per_target_counts.items())),
        "seed": seed,
        "num_context_prompts": len(context_prompts),
    }
    return examples, stats


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

