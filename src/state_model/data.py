from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.metrics import compute_outcome_reward
from src.schema import Rollout
from src.serialization import (
    canonicalize_question_state,
    question_state_key,
    serialize_question_state_for_model,
)


def aggregate_question_state_targets(
    buffered_rollouts: Iterable[Mapping[str, Any]],
    outcome_reward_mixing_coefficient: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visits: dict[str, list[float]] = defaultdict(list)
    state_records: dict[str, dict[str, Any]] = {}
    temporal_ordering_pairs: list[dict[str, Any]] = []

    for raw_rollout in buffered_rollouts:
        rollout = Rollout.from_dict(raw_rollout)
        rollout_outcome_reward = (
            rollout.outcome_reward
            if rollout.outcome_reward is not None
            else compute_outcome_reward(
                rollout.predicted_answer_list,
                rollout.gold_answer_set,
                outcome_reward_mixing_coefficient,
            )
        )
        state_prefixes = rollout.kg_search_state.prefixes(include_empty=True)
        serialized_states = [
            serialize_question_state_for_model(rollout.question, state) for state in state_prefixes
        ]

        for state, serialized_state in zip(state_prefixes[1:], serialized_states[1:]):
            key = question_state_key(rollout.question, state)
            visits[key].append(float(rollout_outcome_reward))
            state_records.setdefault(
                key,
                {
                    "question_state_key": key,
                    "canonical_question_state": canonicalize_question_state(
                        rollout.question,
                        state,
                    ),
                    "serialized_question_state": serialized_state,
                },
            )

        for earlier_index in range(len(state_prefixes)):
            for later_index in range(earlier_index + 1, len(state_prefixes)):
                temporal_ordering_pairs.append(
                    {
                        "question_id": rollout.question_id,
                        "earlier_turn": earlier_index,
                        "later_turn": later_index,
                        "earlier_serialized_question_state": serialized_states[earlier_index],
                        "later_serialized_question_state": serialized_states[later_index],
                        "rollout_outcome_reward": float(rollout_outcome_reward),
                    }
                )

    question_states: list[dict[str, Any]] = []
    for key in sorted(state_records):
        rewards = visits[key]
        record = dict(state_records[key])

        record.update(
            {
                "target_outcome_reward": sum(rewards) / len(rewards),
                "visit_count": len(rewards),
            }
        )
        question_states.append(record)
    return question_states, temporal_ordering_pairs


class QuestionStateDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "serialized_question_state": row["serialized_question_state"],
            "target_outcome_reward": float(row["target_outcome_reward"]),
        }


class TemporalOrderingDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "earlier_serialized_question_state": row["earlier_serialized_question_state"],
            "later_serialized_question_state": row["later_serialized_question_state"],
            "rollout_outcome_reward": float(row["rollout_outcome_reward"]),
        }


@dataclass
class QuestionStateCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [row["serialized_question_state"] for row in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["target_outcome_reward"] = torch.tensor(
            [row["target_outcome_reward"] for row in batch],
            dtype=torch.float32,
        )
        return encoded


@dataclass
class TemporalOrderingCollator:
    tokenizer: Any
    max_length: int

    def _encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "earlier": self._encode([row["earlier_serialized_question_state"] for row in batch]),
            "later": self._encode([row["later_serialized_question_state"] for row in batch]),
            "rollout_outcome_reward": torch.tensor(
                [row["rollout_outcome_reward"] for row in batch],
                dtype=torch.float32,
            ),
        }
