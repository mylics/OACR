from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.io import load_yaml_config, read_jsonl, set_random_seed
from src.state_model.model import KGStateOutcomeModel, KGStateOutcomeModelConfig
from src.state_model.training import StateModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train the KG-state outcome model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    set_random_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_model"], use_fast=True)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[QUESTION]", "[KG_SEARCH_STATE]", "[EMPTY_STATE]"]}
    )
    model = KGStateOutcomeModel(
        KGStateOutcomeModelConfig(
            encoder_model=str(config["encoder_model"]), dropout=float(config["dropout"])
        )
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = StateModelTrainer(model, tokenizer, config)
    metrics = trainer.fit_rows(
        list(read_jsonl(config["question_state_file"])),
        list(read_jsonl(config["temporal_ordering_file"])),
    )
    model.save_pretrained(Path(config["output_dir"]), tokenizer=tokenizer)
    print(metrics)


if __name__ == "__main__":
    main()
