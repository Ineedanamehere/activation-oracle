"""Evaluation helpers for pairwise verifier prompts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from nl_probes.base_experiment import VerbalizerResults
from nl_probes.pairwise_data import normalize_word


def extract_primary_prediction(result: VerbalizerResults) -> str:
    candidates: list[str] = []
    candidates.extend(response for response in result.segment_responses if response)
    candidates.extend(response for response in result.full_sequence_responses if response)
    candidates.extend(response for response in result.token_responses if response)
    if not candidates:
        return ""
    return normalize_word(candidates[0])


def score_pairwise_result(result: VerbalizerResults, example_meta: dict[str, Any]) -> dict[str, Any]:
    prediction = extract_primary_prediction(result)
    correct_candidate = normalize_word(example_meta["correct_candidate"])
    candidate_a = normalize_word(example_meta["candidate_a"])
    candidate_b = normalize_word(example_meta["candidate_b"])

    prediction_in_candidates = prediction in {candidate_a, candidate_b}
    is_correct = prediction == correct_candidate

    return {
        "prediction": prediction,
        "prediction_in_candidates": prediction_in_candidates,
        "is_correct": is_correct,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "correct_candidate": correct_candidate,
    }


def summarize_pairwise_scores(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_rows)
    overall_correct = sum(1 for row in scored_rows if row["is_correct"])
    in_candidate_count = sum(1 for row in scored_rows if row["prediction_in_candidates"])

    per_target_totals: Counter[str] = Counter()
    per_target_correct: Counter[str] = Counter()
    per_source_totals: Counter[str] = Counter()
    per_source_correct: Counter[str] = Counter()

    for row in scored_rows:
        target = row["target_secret"]
        source = row["negative_source"]
        per_target_totals[target] += 1
        per_source_totals[source] += 1
        if row["is_correct"]:
            per_target_correct[target] += 1
            per_source_correct[source] += 1

    return {
        "num_examples": total,
        "accuracy": (overall_correct / total) if total else 0.0,
        "prediction_in_candidates_rate": (in_candidate_count / total) if total else 0.0,
        "per_target_accuracy": {
            target: per_target_correct[target] / count for target, count in sorted(per_target_totals.items())
        },
        "per_negative_source_accuracy": {
            source: per_source_correct[source] / count for source, count in sorted(per_source_totals.items())
        },
        "per_target_counts": dict(sorted(per_target_totals.items())),
        "per_negative_source_counts": dict(sorted(per_source_totals.items())),
    }

