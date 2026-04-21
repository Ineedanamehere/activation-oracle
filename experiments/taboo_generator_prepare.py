"""Prepare structured hidden-word generator supervision data for Taboo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nl_probes.pairwise_data import (
    build_taboo_generator_examples,
    get_taboo_secrets,
    load_taboo_context_prompts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--prompt-type", default="direct", choices=["direct", "standard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pairwise_data/taboo_generator_val.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context_prompts = load_taboo_context_prompts(split=args.split, prompt_type=args.prompt_type)
    taboo_secrets = get_taboo_secrets(include_adversarial=False)
    examples, stats = build_taboo_generator_examples(
        context_prompts=context_prompts,
        taboo_secrets=taboo_secrets,
        seed=args.seed,
    )

    write_jsonl(args.output, examples)
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(
            {
                "assumption": (
                    "This is a structured generator data draft only. It reuses the existing Taboo prompt split and "
                    "formats outputs as `Candidates: ...` plus `Final answer: ...`."
                ),
                "stats": stats,
                "output_path": str(args.output),
            },
            f,
            indent=2,
        )

    print(f"Wrote {stats['num_examples']} structured generator examples to {args.output}")
    print(f"Per-target counts: {stats['per_target_counts']}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
