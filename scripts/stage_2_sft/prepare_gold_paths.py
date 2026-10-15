from __future__ import annotations

import argparse

from src.io import load_yaml_config, read_jsonl, write_jsonl
from src.kg.client import HTTPKnowledgeGraphTool
from src.sft.gold_path import build_gold_path_record_with_tool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct shortest gold paths that reach at least one answer."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    records = []
    skipped = 0
    for question_record in read_jsonl(config["question_file"]):
        knowledge_graph_tool = HTTPKnowledgeGraphTool(
            service_url=config["kg_service_url"],
            question=str(question_record["question"]),
            start_entities=question_record.get("start_entities") or (),
            request_timeout=int(config["kg_request_timeout"]),
        )
        try:
            gold_path_record = build_gold_path_record_with_tool(
                knowledge_graph_tool,
                question_record,
                max_path_hops=int(config["max_path_hops"]),
            )
        finally:
            knowledge_graph_tool.reset()
        if gold_path_record is None:
            skipped += 1
        else:
            records.append(gold_path_record)
    count = write_jsonl(config["gold_path_file"], records)
    print(f"Saved {count} gold-path records; skipped_without_path={skipped}.")


if __name__ == "__main__":
    main()
