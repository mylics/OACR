from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.io import load_yaml_config, set_random_seed
from src.privacy import sanitize_huggingface_metadata
from src.sft.dataset import SFTBatchCollator, SFTTrajectoryDataset
from src.rl.objective import masked_sequence_mean
from src.rl.training import compute_causal_token_log_probabilities


def compute_sft_loss(model, batch):
    probabilities = compute_causal_token_log_probabilities(model, batch["input_ids"], batch["attention_mask"])
    return -masked_sequence_mean(probabilities, batch["labels"][:, 1:] != -100)


@torch.inference_mode()
def evaluate(model, validation_loader, accelerator: Accelerator) -> float:
    model.eval()
    losses = []
    for batch in validation_loader:
        loss = compute_sft_loss(model, batch)
        losses.append(accelerator.gather_for_metrics(loss.detach()).float())
    model.train()
    if not losses:
        return float("nan")
    return torch.cat([loss.reshape(-1) for loss in losses]).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SFT-initialized KGQA policy π_θ0.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    set_random_seed(int(config["seed"]))

    mixed_precision = "bf16" if bool(config.get("bf16", False)) else "no"
    processes = int(os.environ.get("WORLD_SIZE", "1"))
    per_step_batch = int(config["train_batch_size_per_device"]) * processes
    if int(config["global_batch_size"]) % per_step_batch:
        raise ValueError("Global SFT batch must divide evenly across devices")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["global_batch_size"]) // per_step_batch,
        mixed_precision=mixed_precision,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else None,
    )
    if bool(config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    train_dataset = SFTTrajectoryDataset(
        config["train_file"],
        tokenizer,
        int(config["max_length"]),
    )
    validation_dataset = SFTTrajectoryDataset(
        config["validation_file"],
        tokenizer,
        int(config["max_length"]),
    )
    if len(train_dataset) == 0:
        raise ValueError("No SFT trajectories were found")
    collator = SFTBatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size_per_device"]),
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["train_batch_size_per_device"]),
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(config["gradient_accumulation_steps"]))
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * float(config["warmup_ratio"])),
        num_training_steps=total_updates,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )

    global_update = 0
    model.train()
    for epoch in range(int(config["epochs"])):
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = compute_sft_loss(model, batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(config["max_grad_norm"]),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_update += 1
                if accelerator.is_main_process and global_update % 50 == 0:
                    print(
                        f"epoch={epoch + 1} update={global_update} L_sft={loss.detach().float().item():.6f}"
                    )

        validation_loss = evaluate(model, validation_loader, accelerator)
        if accelerator.is_main_process:
            print(f"epoch={epoch + 1} validation_L_sft={validation_loss:.6f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        sanitize_huggingface_metadata(unwrapped, tokenizer)
        unwrapped.save_pretrained(
            output_dir,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)
        print(f"Saved π_θ0 after {global_update} optimizer updates.")


if __name__ == "__main__":
    main()
