from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Hashable, Sequence

import torch


def normalize_answer_entity(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalized_answer_set(values: Sequence[str]) -> set[str]:
    return {normalized for value in values if (normalized := normalize_answer_entity(value))}


def compute_answer_set_f1(
    predicted_answer_set: Sequence[str],
    gold_answer_set: Sequence[str],
) -> float:
    predicted = normalized_answer_set(predicted_answer_set)
    gold = normalized_answer_set(gold_answer_set)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    return 2.0 * len(predicted & gold) / (len(predicted) + len(gold))


def compute_hit_at_one(
    predicted_answer_list: Sequence[str],
    gold_answer_set: Sequence[str],
) -> float:
    if not predicted_answer_list:
        return 0.0
    first = normalize_answer_entity(predicted_answer_list[0])
    return float(bool(first) and first in normalized_answer_set(gold_answer_set))


def compute_outcome_reward(
    predicted_answer_list: Sequence[str],
    gold_answer_set: Sequence[str],
    outcome_reward_mixing_coefficient: float,
) -> float:
    mixing = float(outcome_reward_mixing_coefficient)
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("outcome_reward_mixing_coefficient must be in [0, 1]")
    answer_set_f1 = compute_answer_set_f1(predicted_answer_list, gold_answer_set)
    hit_at_one = compute_hit_at_one(predicted_answer_list, gold_answer_set)

    return mixing * answer_set_f1 + (1.0 - mixing) * hit_at_one


def compute_group_outcome_advantage(
    outcome_rewards: torch.Tensor,
    question_group_ids: Sequence[Hashable],
    advantage_epsilon: float = 1e-6,
) -> torch.Tensor:
    if outcome_rewards.ndim != 1:
        raise ValueError("outcome_rewards must have shape [num_rollouts]")
    if len(question_group_ids) != outcome_rewards.shape[0]:
        raise ValueError("question_group_ids must align with outcome_rewards")

    grouped_indices: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(question_group_ids):
        grouped_indices[group_id].append(index)

    outcome_level_advantage = torch.zeros_like(outcome_rewards, dtype=torch.float32)
    for indices in grouped_indices.values():
        index_tensor = torch.tensor(indices, device=outcome_rewards.device)
        group_rewards = outcome_rewards[index_tensor].float()

        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False)
        outcome_level_advantage[index_tensor] = (group_rewards - group_mean) / (
            group_std + float(advantage_epsilon)
        )
    return outcome_level_advantage
