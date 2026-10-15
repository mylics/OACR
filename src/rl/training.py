from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from src.rl.credit import (
    compute_estimated_outcome_reward_change,
    compute_operation_credit,
    compute_outcome_anchored_advantage,
    compute_reallocated_token_weights,
)
from src.schema import Rollout
from src.sft.dataset import tokenize_policy_trajectory
from src.state_model.model import KGStateOutcomeScorer


def disable_policy_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def compute_causal_token_log_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    next_token_logits = output.logits[:, :-1, :]
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    log_probabilities = torch.log_softmax(
        next_token_logits.float(),
        dim=-1,
    )
    return log_probabilities.gather(-1, next_token_ids).squeeze(-1)


def build_outcome_anchored_training_example(
    rollout: Rollout,
    outcome_level_advantage: float,
    tokenizer: Any,
    max_length: int,
    kg_state_outcome_scorer: KGStateOutcomeScorer,
    outcome_reward_change_bound: float,
    invalid_operation_credit: float,
    reallocation_strength: float,
) -> dict[str, torch.Tensor]:
    tokenized = tokenize_policy_trajectory(
        rollout.messages,
        tokenizer,
        max_length,
    )
    num_operations = rollout.kg_search_state.num_turns
    operation_token_masks = tokenized["operation_token_mask"]
    if operation_token_masks.shape[0] != num_operations:
        raise ValueError(
            "Serialized operation spans do not align with KG search-state turns: "
            f"{operation_token_masks.shape[0]} != {num_operations}"
        )

    state_prefixes = rollout.kg_search_state.prefixes(include_empty=True)

    kg_state_outcome_estimates = torch.tensor(
        kg_state_outcome_scorer.score(rollout.question, state_prefixes),
        dtype=torch.float32,
    )
    estimated_outcome_reward_change = compute_estimated_outcome_reward_change(
        kg_state_outcome_estimates,
        outcome_reward_change_bound,
    )
    invalid_operation_mask = torch.tensor(
        [turn.is_invalid for turn in rollout.kg_search_state.turns],
        dtype=torch.bool,
    )
    operation_credit = compute_operation_credit(
        estimated_outcome_reward_change,
        outcome_level_advantage,
        invalid_operation_mask,
        invalid_operation_credit,
    )
    normalized_token_weight = compute_reallocated_token_weights(
        tokenized["policy_token_mask"],
        operation_token_masks,
        operation_credit,
        reallocation_strength,
    )
    outcome_anchored_advantage = compute_outcome_anchored_advantage(
        outcome_level_advantage,
        normalized_token_weight,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "policy_token_mask": tokenized["policy_token_mask"],
        "outcome_anchored_advantage": outcome_anchored_advantage,
        "estimated_outcome_reward_change": estimated_outcome_reward_change,
        "operation_credit": operation_credit,
    }


def build_outcome_only_training_example(rollout, advantage, tokenizer, max_length):
    tokenized = tokenize_policy_trajectory(rollout.messages, tokenizer, max_length)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "policy_token_mask": tokenized["policy_token_mask"],
        "outcome_anchored_advantage": tokenized["policy_token_mask"].float() * float(advantage),
    }


@dataclass
class OACRTrainingBatchCollator:
    pad_token_id: int

    def __call__(
        self,
        examples: Sequence[Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(example["input_ids"].shape[0] for example in examples)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        policy_token_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
        )
        outcome_anchored_advantage = torch.zeros(
            (batch_size, max_length),
            dtype=torch.float32,
        )
        for index, example in enumerate(examples):
            length = example["input_ids"].shape[0]
            input_ids[index, :length] = example["input_ids"]
            attention_mask[index, :length] = example["attention_mask"]
            policy_token_mask[index, :length] = example["policy_token_mask"]
            outcome_anchored_advantage[index, :length] = example["outcome_anchored_advantage"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "policy_token_mask": policy_token_mask[:, 1:],
            "outcome_anchored_advantage": outcome_anchored_advantage[:, 1:],
        }
