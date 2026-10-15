from __future__ import annotations

import argparse
import random
from collections import Counter

from src.io import load_yaml_config, read_jsonl, set_random_seed, write_jsonl
from src.kg.client import HTTPKnowledgeGraphTool
from src.sft.trajectory_builder import InvalidSFTTrajectory, SFTTrajectoryBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute gold KG paths and build SFT trajectories.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    seed = int(config["seed"])
    set_random_seed(seed)

    def kg_tool_factory(question, start_entities):
        return HTTPKnowledgeGraphTool(
            service_url=config["kg_service_url"],
            question=question,
            start_entities=start_entities,
            request_timeout=int(config["kg_request_timeout"]),
        )

    builder = SFTTrajectoryBuilder(kg_tool_factory)
    trajectories = []
    rejected = Counter()
    for record in read_jsonl(config["gold_path_file"]):
        try:
            trajectories.append(builder.build(record).to_dict())
        except InvalidSFTTrajectory as exc:
            rejected[str(exc)] += 1

    random.Random(seed).shuffle(trajectories)
    validation_size = round(len(trajectories) * float(config["validation_ratio"]))
    validation_rows = trajectories[:validation_size]
    train_rows = trajectories[validation_size:]
    write_jsonl(config["train_file"], train_rows)
    write_jsonl(config["validation_file"], validation_rows)
    print(
        f"Saved train={len(train_rows)} validation={len(validation_rows)} "
        f"rejected={sum(rejected.values())} reasons={dict(rejected)}"
    )


if __name__ == "__main__":
    main()
