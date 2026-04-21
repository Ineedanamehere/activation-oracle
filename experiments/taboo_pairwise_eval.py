"""Zero-shot pairwise verifier eval for Activation Oracle on Taboo."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo
from nl_probes.pairwise_data import (
    build_pairwise_verbalizer_prompt,
    get_taboo_secrets,
    load_taboo_context_prompts,
)
from nl_probes.pairwise_eval import score_pairwise_result, summarize_pairwise_scores
from nl_probes.utils.common import load_model, load_tokenizer


MODEL_DEFAULTS = {
    "google/gemma-2-9b-it": {
        "verbalizer_lora_path": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        "target_lora_path_template": "bcywinski/gemma-2-9b-it-taboo-{lora_path}",
        "segment_start": -10,
    },
    "Qwen/Qwen3-8B": {
        "verbalizer_lora_path": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        "target_lora_path_template": "adamkarvonen/Qwen3-8B-taboo-{lora_path}_50_mix",
        "segment_start": -10,
    },
}

GENERIC_DISTRACTORS = ["secret", "hidden", "answer", "word"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-9b-it", choices=sorted(MODEL_DEFAULTS))
    parser.add_argument(
        "--verbalizer-lora-path",
        default=None,
        help="Optional verbalizer LoRA path or local checkpoint directory. Defaults to the repo baseline AO checkpoint.",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--prompt-type", default="direct", choices=["direct", "standard"])
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/taboo_pairwise_eval_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    defaults = MODEL_DEFAULTS[args.model_name]
    context_prompts = load_taboo_context_prompts(split=args.split, prompt_type=args.prompt_type)
    context_prompts = context_prompts[: args.max_contexts]
    taboo_secrets = get_taboo_secrets(include_adversarial=False)

    verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
    example_rows: list[dict] = []
    rng = random.Random(args.seed)

    for target_secret in taboo_secrets:
        other_secret = next(word for word in taboo_secrets if word != target_secret)
        for context_prompt in context_prompts:
            for incorrect_candidate, negative_source in [
                (other_secret, "other_secret"),
                (GENERIC_DISTRACTORS[0], "generic"),
            ]:
                candidates = [target_secret, incorrect_candidate]
                rng.shuffle(candidates)
                candidate_a, candidate_b = candidates
                verbalizer_prompt = build_pairwise_verbalizer_prompt(candidate_a, candidate_b)
                verbalizer_prompt_infos.append(
                    VerbalizerInputInfo(
                        context_prompt=[{"role": "user", "content": context_prompt}],
                        ground_truth=target_secret,
                        verbalizer_prompt=verbalizer_prompt,
                    )
                )
                example_rows.append(
                    {
                        "target_secret": target_secret,
                        "context_prompt": context_prompt,
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "correct_candidate": target_secret,
                        "negative_source": negative_source,
                    }
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, dtype)
    model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")

    verbalizer_lora_path = args.verbalizer_lora_path or defaults["verbalizer_lora_path"]
    target_lora_template = defaults["target_lora_path_template"]
    sanitized_verbalizer_name = base_experiment.load_lora_adapter(model, verbalizer_lora_path)

    config = base_experiment.VerbalizerEvalConfig(
        model_name=args.model_name,
        activation_input_types=["lora"],
        verbalizer_input_types=["segment"],
        eval_batch_size=args.eval_batch_size,
        full_seq_repeats=0,
        segment_repeats=1,
        segment_start_idx=defaults["segment_start"],
        verbalizer_generation_kwargs={
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 5,
        },
    )

    all_scored_rows: list[dict] = []
    idx = 0

    for target_secret in taboo_secrets:
        target_rows = [row for row in example_rows if row["target_secret"] == target_secret]
        target_infos = verbalizer_prompt_infos[idx : idx + len(target_rows)]
        idx += len(target_rows)

        target_lora_path = target_lora_template.format(lora_path=target_secret)
        sanitized_target_name = base_experiment.load_lora_adapter(model, target_lora_path)

        results = base_experiment.run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=target_infos,
            verbalizer_lora_path=sanitized_verbalizer_name,
            target_lora_path=sanitized_target_name,
            config=config,
            device=device,
        )

        for result, row in zip(results, target_rows, strict=True):
            scored = score_pairwise_result(result, row)
            scored_row = {
                **row,
                **scored,
                "raw_segment_responses": result.segment_responses,
                "raw_full_sequence_responses": result.full_sequence_responses,
                "act_key": result.act_key,
                "verbalizer_prompt": result.verbalizer_prompt,
            }
            all_scored_rows.append(scored_row)

        if sanitized_target_name in model.peft_config:
            model.delete_adapter(sanitized_target_name)

    if sanitized_verbalizer_name in model.peft_config:
        model.delete_adapter(sanitized_verbalizer_name)

    summary = summarize_pairwise_scores(all_scored_rows)
    failures = [row for row in all_scored_rows if not row["is_correct"]][:20]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name_str = args.model_name.split("/")[-1].replace(".", "_")
    output_json = args.output_dir / f"{model_name_str}_{args.prompt_type}_{args.split}_pairwise_summary.json"
    output_jsonl = args.output_dir / f"{model_name_str}_{args.prompt_type}_{args.split}_pairwise_rows.jsonl"

    with output_json.open("w") as f:
        json.dump(
            {
                "config": asdict(config),
                "model_name": args.model_name,
                "verbalizer_lora_path": verbalizer_lora_path,
                "num_contexts": len(context_prompts),
                "summary": summary,
                "failures": failures,
            },
            f,
            indent=2,
        )

    with output_jsonl.open("w") as f:
        for row in all_scored_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Pairwise accuracy: {summary['accuracy']:.4f}")
    print(f"Prediction-in-candidates rate: {summary['prediction_in_candidates_rate']:.4f}")
    print(f"Per-negative-source accuracy: {summary['per_negative_source_accuracy']}")
    print(f"Per-target accuracy: {summary['per_target_accuracy']}")
    print(f"Saved summary to {output_json}")
    print(f"Saved rows to {output_jsonl}")


if __name__ == "__main__":
    main()
