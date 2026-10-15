import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.io import load_yaml_config, read_jsonl, set_random_seed, write_jsonl
from src.metrics import compute_answer_set_f1, compute_hit_at_one
from src.rl.rollout import AgenticKGQAPolicy


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agentic KGQA policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--questions")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    set_random_seed(int(config["seed"]))
    model_path = args.model or config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.get("bf16") and device.type == "cuda" else None,
    ).to(device)
    policy = AgenticKGQAPolicy(
        model,
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
    rows = []
    for index, question in enumerate(read_jsonl(args.questions or config["question_file"])):
        started = time.perf_counter()
        rollout = policy.sample_rollout(
            str(question.get("question_id") or question.get("id") or index),
            question["question"],
            question.get("start_entities", ()),
            question.get("gold_answer_set", ()),
            float(config["outcome_reward_mixing_coefficient"]),
        )
        rows.append(
            dict(
                rollout.to_dict(),
                f1=compute_answer_set_f1(rollout.predicted_answer_list, rollout.gold_answer_set),
                hit_at_one=compute_hit_at_one(rollout.predicted_answer_list, rollout.gold_answer_set),
                llm_calls=sum(message["role"] == "assistant" for message in rollout.messages),
                output_tokens=policy.max_response_length - policy.remaining_response_tokens,
                latency_seconds=time.perf_counter() - started,
            )
        )
    if not rows:
        raise ValueError("No evaluation questions were found")
    output = Path(args.output or config["output_file"])
    write_jsonl(output, rows)
    summary = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("f1", "hit_at_one", "llm_calls", "output_tokens", "latency_seconds")
    }
    summary["questions"] = len(rows)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
