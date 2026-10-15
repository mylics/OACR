from __future__ import annotations

import argparse

from src.io import load_yaml_config, read_jsonl, write_jsonl
from src.state_model.data import aggregate_question_state_targets
from src.state_model.buffer import RolloutBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate y(q,S) targets and temporal ordering pairs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    buffer = RolloutBuffer(
        int(config["rollout_buffer_capacity"]), read_jsonl(config["buffered_rollouts_file"])
    )
    question_states, temporal_ordering_pairs = aggregate_question_state_targets(
        buffer.records(),
        outcome_reward_mixing_coefficient=float(config["outcome_reward_mixing_coefficient"]),
    )
    state_count = write_jsonl(config["question_state_file"], question_states)
    pair_count = write_jsonl(
        config["temporal_ordering_file"],
        temporal_ordering_pairs,
    )
    print(
        f"Prepared {state_count} canonical question-state targets and {pair_count} temporal ordering pairs."
    )


if __name__ == "__main__":
    main()
