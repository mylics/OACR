from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.io import load_yaml_config, read_jsonl, set_random_seed, write_jsonl
from src.privacy import sanitize_huggingface_metadata
from src.prompts import KGQA_SYSTEM_PROMPT, format_question_prompt
from src.rl.engine import PolicyTrainer
from src.rl.rollout import AgenticKGQAPolicy
from src.rl.training import disable_policy_dropout
from src.state_model.buffer import RolloutBuffer
from src.state_model.model import KGStateOutcomeScorer
from src.state_model.refresh import gather_rollouts, refresh_state_model
from src.state_model.training import StateModelTrainer


def validate_config(config, world_size=1):
    for name in (
        "train_batch_size",
        "mini_batch_size",
        "micro_batch_size",
        "rollout_group_size",
        "max_prompt_length",
        "max_response_length",
        "max_turns",
        "epochs",
        "updates_per_rollout_group",
    ):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    batch = int(config["train_batch_size"])
    mini = int(config["mini_batch_size"])
    if batch % mini or mini % world_size:
        raise ValueError("Training batch must divide into mini-batches evenly across ranks")
    if int(config["rollout_group_size"]) < 2:
        raise ValueError("GRPO needs at least two rollouts per question")
    if bool(config.get("credit_reallocation", True)):
        for name in ("rollout_buffer_capacity", "state_model_refresh_interval", "state_model_refresh_epochs"):
            if int(config[name]) < 1:
                raise ValueError(f"{name} must be positive")
    if not 0 < float(config["policy_clip_epsilon"]) < 1:
        raise ValueError("Policy clipping epsilon must lie between zero and one")
    if min(float(config["reallocation_strength"]), float(config["invalid_operation_credit"])) < 0:
        raise ValueError("Operation-credit coefficients must be nonnegative")


def eligible_questions(records, tokenizer, max_prompt_length):
    result = []
    for index, record in enumerate(records):
        question = dict(record)
        question["question_id"] = str(record.get("question_id") or record.get("id") or index)
        messages = [
            {"role": "system", "content": KGQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_question_prompt(str(record["question"]), record.get("start_entities", ())),
            },
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) <= max_prompt_length:
            result.append(question)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train an OACR or outcome-only GRPO policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    if args.outcome_only:
        config["credit_reallocation"] = False
        config["output_dir"] = "checkpoints/oacr_grpo"
    if args.output_dir:
        config["output_dir"] = args.output_dir
    validate_config(config)
    if args.check_config:
        print(json.dumps(config, indent=2))
        return
    mixed_precision = "bf16" if config.get("bf16") and torch.cuda.is_available() else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    validate_config(config, accelerator.num_processes)
    set_random_seed(int(config["seed"]) + accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(config["sft_policy_model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if mixed_precision == "bf16" else None
    policy = AutoModelForCausalLM.from_pretrained(config["sft_policy_model"], torch_dtype=dtype)
    reference = AutoModelForCausalLM.from_pretrained(config["reference_policy_model"], torch_dtype=dtype).to(
        accelerator.device
    )
    disable_policy_dropout(policy)
    disable_policy_dropout(reference)
    reference.eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    policy, optimizer = accelerator.prepare(policy, optimizer)
    unwrapped = accelerator.unwrap_model(policy)
    rollout_policy = AgenticKGQAPolicy(
        unwrapped,
        tokenizer,
        config["kg_service_url"],
        int(config["kg_request_timeout"]),
        int(config["max_turns"]),
        int(config["max_new_tokens_per_turn"]),
        float(config["temperature"]),
        float(config["top_p"]),
        max_prompt_length=int(config["max_prompt_length"]),
        max_response_length=int(config["max_response_length"]),
    )
    output = Path(config["output_dir"])
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    accelerator.wait_for_everyone()
    scorer = state_trainer = buffer = None
    if config.get("credit_reallocation", True):
        state_config = load_yaml_config(config["state_model_config"])
        if float(state_config["outcome_reward_mixing_coefficient"]) != float(
            config["outcome_reward_mixing_coefficient"]
        ):
            raise ValueError("State-model targets and policy rewards must use the same mixing coefficient")
        state_device = (
            accelerator.device if config["state_model_device"] == "auto" else config["state_model_device"]
        )
        scorer = KGStateOutcomeScorer(
            config["kg_state_outcome_model"],
            device=state_device,
            max_length=int(config["state_model_max_length"]),
            batch_size=int(config["state_model_batch_size"]),
        )
        if accelerator.is_main_process:
            state_trainer = StateModelTrainer(scorer.model, scorer.tokenizer, state_config)
        buffer = RolloutBuffer(
            int(config["rollout_buffer_capacity"]), read_jsonl(config["buffered_rollouts_file"])
        )
    trainer = PolicyTrainer(
        policy, reference, tokenizer, rollout_policy, optimizer, accelerator, config, scorer
    )
    questions = eligible_questions(
        read_jsonl(config["question_file"]), tokenizer, int(config["max_prompt_length"])
    )
    batch_size = int(config["train_batch_size"])
    if len(questions) < batch_size:
        raise ValueError("Training requires at least one complete global question batch")
    iteration = 0
    for epoch in range(int(config["epochs"])):
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        order = torch.randperm(len(questions), generator=generator).tolist()
        for start in range(0, len(order) - batch_size + 1, batch_size):
            selected = [questions[i] for i in order[start : start + batch_size]]
            local = selected[accelerator.process_index :: accelerator.num_processes]
            rollouts, metrics = trainer.step(local)
            iteration += 1
            state_metrics = None
            if buffer is not None:
                buffer.extend(gather_rollouts(rollouts))
                if iteration % int(config["state_model_refresh_interval"]) == 0:
                    state_metrics = refresh_state_model(
                        scorer,
                        state_trainer,
                        buffer,
                        int(config["state_model_refresh_epochs"]),
                        accelerator,
                        output / "state_model" / f"iteration_{iteration}",
                    )
            if accelerator.is_main_process:
                record = dict(iteration=iteration, epoch=epoch + 1, **metrics)
                if state_metrics is not None:
                    record["state_model_refresh"] = state_metrics
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(record, flush=True)
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(policy)
        if accelerator.is_main_process:
            checkpoint = output / f"epoch_{epoch + 1}"
            sanitize_huggingface_metadata(unwrapped, tokenizer)
            unwrapped.save_pretrained(checkpoint, state_dict=state_dict, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            if buffer is not None:
                write_jsonl(output / "rollout_buffer.jsonl", buffer.records())
                scorer.model.save_pretrained(output / "state_model" / "latest", tokenizer=scorer.tokenizer)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
