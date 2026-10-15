from __future__ import annotations

import argparse

from src.data import (
    normalize_canonical_questions,
    normalize_cwq_questions,
    normalize_grailqa_questions,
    normalize_webqsp_questions,
)
from src.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert KGQA questions to the canonical RL data contract.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format",
        required=True,
        choices=["canonical", "cwq", "webqsp", "grailqa"],
    )
    args = parser.parse_args()

    normalizers = {
        "canonical": normalize_canonical_questions,
        "cwq": normalize_cwq_questions,
        "webqsp": normalize_webqsp_questions,
        "grailqa": normalize_grailqa_questions,
    }
    rows = normalizers[args.format](args.input)
    count = write_jsonl(args.output, rows)
    print(f"Saved {count} canonical RL questions.")


if __name__ == "__main__":
    main()
