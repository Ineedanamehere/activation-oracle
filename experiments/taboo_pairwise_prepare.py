"""Prepare pairwise verifier training data for Taboo using existing AO splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nl_probes.pairwise_data import (
    build_taboo_pairwise_examples,
    get_taboo_secrets,
    load_taboo_context_prompts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Taboo prompt split to use.")
    parser.add_argument("--prompt-type", default="direct", choices=["direct", "standard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-other-secret-negatives", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pairwise_data/taboo_pairwise_val.jsonl"),
        help="Output JSONL path, relative to repo root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context_prompts = load_taboo_context_prompts(split=args.split, prompt_type=args.prompt_type)
    taboo_secrets = get_taboo_secrets(include_adversarial=False)
    examples, stats = build_taboo_pairwise_examples(
        context_prompts=context_prompts,
        taboo_secrets=taboo_secrets,
        seed=args.seed,
        num_other_secret_negatives=args.num_other_secret_negatives,
    )

    write_jsonl(args.output, examples)

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "assumption": (
            "The repository does not include a Taboo train prompt split, so this first version uses the existing "
            f"{args.split} split as the pairwise data source."
        ),
        "stats": stats,
        "output_path": str(args.output),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {stats['num_examples']} pairwise examples to {args.output}")
    print(f"Per-target counts: {stats['per_target_counts']}")
    print(f"Negative source counts: {stats['negative_source_counts']}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
