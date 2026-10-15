from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.io import load_yaml_config, read_jsonl, set_random_seed, write_jsonl
from src.rl.rollout import AgenticKGQAPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the initial buffer from the outcome-only GRPO policy."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    collector = config["collector"]
    set_random_seed(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        collector["policy_model"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        collector["policy_model"],
        torch_dtype=(torch.bfloat16 if bool(collector.get("bf16", False)) else None),
    )
    device = torch.device(collector.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    policy = AgenticKGQAPolicy(
        model=model,
        tokenizer=tokenizer,
        kg_service_url=collector["kg_service_url"],
        kg_request_timeout=int(collector["kg_request_timeout"]),
        max_turns=int(collector["max_turns"]),
        max_new_tokens_per_turn=int(collector["max_new_tokens_per_turn"]),
        temperature=float(collector["temperature"]),
        top_p=float(collector["top_p"]),
        max_prompt_length=int(collector["max_prompt_length"]),
        max_response_length=int(collector["max_response_length"]),
    )

    def sampled_rollouts():
        for question_record in read_jsonl(collector["question_file"]):
            for _ in range(int(collector["rollouts_per_question"])):
                yield policy.sample_rollout(
                    question_id=str(question_record.get("question_id") or question_record.get("id") or ""),
                    question=str(question_record["question"]),
                    start_entities=question_record.get("start_entities") or (),
                    gold_answer_set=question_record.get("gold_answer_set") or (),
                    outcome_reward_mixing_coefficient=float(config["outcome_reward_mixing_coefficient"]),
                ).to_dict()

    count = write_jsonl(config["buffered_rollouts_file"], sampled_rollouts())
    print(f"Collected {count} initial GRPO rollouts.")


if __name__ == "__main__":
    main()
