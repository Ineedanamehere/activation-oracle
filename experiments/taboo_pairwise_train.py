"""Train a pairwise Taboo verifier LoRA using Activation Oracle supervision."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
import nl_probes.sft as sft
from nl_probes.pairwise_data import load_jsonl
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


MODEL_DEFAULTS = {
    "google/gemma-2-9b-it": {
        "target_lora_path_template": "bcywinski/gemma-2-9b-it-taboo-{lora_path}",
        "segment_start": -10,
    },
    "Qwen/Qwen3-8B": {
        "target_lora_path_template": "adamkarvonen/Qwen3-8B-taboo-{lora_path}_50_mix",
        "segment_start": -10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-9b-it", choices=sorted(MODEL_DEFAULTS))
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("experiments/pairwise_data/taboo_pairwise_val.jsonl"),
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Optional pairwise eval JSONL. If omitted, no eval set is used during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/taboo_pairwise"))
    return parser.parse_args()


def build_training_datapoints(
    examples: list[dict],
    model_name: str,
    tokenizer,
    model,
    device: torch.device,
    eval_batch_size: int,
) -> list[TrainingDataPoint]:
    defaults = MODEL_DEFAULTS[model_name]
    config = base_experiment.VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        verbalizer_input_types=["segment"],
        eval_batch_size=eval_batch_size,
        segment_repeats=1,
        full_seq_repeats=0,
        segment_start_idx=defaults["segment_start"],
    )

    grouped_examples: dict[str, list[dict]] = {}
    for example in examples:
        grouped_examples.setdefault(example["target_secret"], []).append(example)

    datapoints: list[TrainingDataPoint] = []
    for target_secret, secret_examples in grouped_examples.items():
        target_lora_path = defaults["target_lora_path_template"].format(lora_path=target_secret)
        sanitized_target_name = base_experiment.load_lora_adapter(model, target_lora_path)

        for start in range(0, len(secret_examples), eval_batch_size):
            batch_examples = secret_examples[start : start + eval_batch_size]
            message_dicts = [[{"role": "user", "content": row["context_prompt"]}] for row in batch_examples]

            inputs_BL = base_experiment.encode_messages(
                tokenizer=tokenizer,
                message_dicts=message_dicts,
                add_generation_prompt=config.add_generation_prompt,
                enable_thinking=config.enable_thinking,
                device=device,
            )

            target_activations = base_experiment.collect_target_activations(
                model=model,
                inputs_BL=inputs_BL,
                config=config,
                target_lora_path=sanitized_target_name,
            )

            acts_dict = target_activations["lora"]
            seq_len = int(inputs_BL["input_ids"].shape[1])

            for batch_idx, example in enumerate(batch_examples):
                attn = inputs_BL["attention_mask"][batch_idx]
                real_len = int(attn.sum().item())
                left_pad = seq_len - real_len
                context_input_ids = inputs_BL["input_ids"][batch_idx, left_pad:].tolist()

                if config.segment_start_idx < 0:
                    segment_start = len(context_input_ids) + config.segment_start_idx
                    segment_end = len(context_input_ids) + config.segment_end_idx
                else:
                    segment_start = config.segment_start_idx
                    segment_end = config.segment_end_idx

                context_positions_rel = list(range(segment_start, segment_end))
                context_positions_abs = [left_pad + p for p in context_positions_rel]
                acts_BD = acts_dict[config.active_layer][batch_idx, context_positions_abs]

                datapoints.append(
                    create_training_datapoint(
                        datapoint_type="taboo_pairwise",
                        prompt=example["verbalizer_prompt"],
                        target_response=example["correct_candidate"],
                        layer=config.active_layer,
                        num_positions=len(context_positions_rel),
                        tokenizer=tokenizer,
                        acts_BD=acts_BD,
                        feature_idx=-1,
                        context_input_ids=context_input_ids,
                        context_positions=context_positions_rel,
                        ds_label=target_secret,
                        meta_info={
                            "target_secret": target_secret,
                            "negative_source": example.get("negative_source", "unknown"),
                        },
                    )
                )

        if sanitized_target_name in model.peft_config:
            model.delete_adapter(sanitized_target_name)

    return datapoints


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    tokenizer = load_tokenizer(args.model_name)
    activation_model = load_model(
        args.model_name,
        dtype,
        device_map={"": f"cuda:{local_rank}"},
    )
    activation_model.eval()
    activation_model.add_adapter(LoraConfig(), adapter_name="default")

    train_examples = load_jsonl(args.train_jsonl)
    eval_examples = load_jsonl(args.eval_jsonl) if args.eval_jsonl else []

    train_data = build_training_datapoints(
        examples=train_examples,
        model_name=args.model_name,
        tokenizer=tokenizer,
        model=activation_model,
        device=device,
        eval_batch_size=args.eval_batch_size,
    )
    eval_data = (
        build_training_datapoints(
            examples=eval_examples,
            model_name=args.model_name,
            tokenizer=tokenizer,
            model=activation_model,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        if eval_examples
        else []
    )

    activation_model = None
    gc.collect()
    torch.cuda.empty_cache()

    cfg = sft.SelfInterpTrainingConfig(
        model_name=args.model_name,
        hook_onto_layer=1,
        layer_percents=[25, 50, 75],
        generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 5},
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_dir=str(args.save_dir),
        wandb_project="activation_oracle_pairwise",
        wandb_run_name=f"taboo_pairwise_{args.model_name.split('/')[-1]}",
        seed=args.seed,
    )

    eval_datasets = {"taboo_pairwise_eval": eval_data} if eval_data else {}
    model_kwargs = {}

    sft.train_model(
        cfg=cfg,
        training_data=train_data,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()
