from __future__ import annotations

import torch


def compute_estimated_outcome_reward_change(
    kg_state_outcome_estimates: torch.Tensor,
    outcome_reward_change_bound: float,
) -> torch.Tensor:
    if kg_state_outcome_estimates.ndim != 1:
        raise ValueError("kg_state_outcome_estimates must have shape [L + 1]")
    if kg_state_outcome_estimates.numel() < 1:
        raise ValueError("At least the empty-state estimate is required")
    bound = float(outcome_reward_change_bound)
    if bound <= 0:
        raise ValueError("outcome_reward_change_bound must be positive")
    adjacent_change = kg_state_outcome_estimates[1:] - kg_state_outcome_estimates[:-1]

    return adjacent_change.clamp(min=-bound, max=bound)


def compute_operation_credit(
    estimated_outcome_reward_change: torch.Tensor,
    outcome_level_advantage: torch.Tensor | float,
    invalid_operation_mask: torch.Tensor,
    invalid_operation_credit: float,
) -> torch.Tensor:
    if estimated_outcome_reward_change.shape != invalid_operation_mask.shape:
        raise ValueError("Change and invalid-operation tensors must align")
    if invalid_operation_credit < 0:
        raise ValueError("invalid_operation_credit must be non-negative")
    advantage = torch.as_tensor(
        outcome_level_advantage,
        device=estimated_outcome_reward_change.device,
        dtype=estimated_outcome_reward_change.dtype,
    )
    positive_credit = estimated_outcome_reward_change.clamp_min(0.0)
    negative_credit = (-estimated_outcome_reward_change).clamp_min(0.0) + float(
        invalid_operation_credit
    ) * invalid_operation_mask.to(estimated_outcome_reward_change.dtype)

    return torch.where(
        advantage > 0,
        positive_credit,
        torch.where(
            advantage < 0,
            negative_credit,
            torch.zeros_like(estimated_outcome_reward_change),
        ),
    )


def compute_reallocated_token_weights(
    policy_token_mask: torch.Tensor,
    operation_token_masks: torch.Tensor,
    operation_credit: torch.Tensor,
    reallocation_strength: float,
) -> torch.Tensor:
    if policy_token_mask.ndim != 1:
        raise ValueError("policy_token_mask must have shape [sequence_length]")
    if operation_token_masks.ndim != 2:
        raise ValueError("operation_token_masks must have shape [num_operations, sequence_length]")
    if operation_token_masks.shape[1] != policy_token_mask.shape[0]:
        raise ValueError("Operation masks must align with the policy-token mask")
    if operation_token_masks.shape[0] != operation_credit.shape[0]:
        raise ValueError("Each operation mask must have one operation credit")
    if reallocation_strength < 0:
        raise ValueError("reallocation_strength must be non-negative")
    if not policy_token_mask.any():
        raise ValueError("At least one policy-generated token is required")

    policy_mask = policy_token_mask.bool()
    overlap_count = operation_token_masks.bool().sum(dim=0)
    if torch.any(overlap_count > 1):
        raise ValueError("Operation token spans must not overlap")
    if torch.any(operation_token_masks.bool() & ~policy_mask.unsqueeze(0)):
        raise ValueError("Operation token spans must be within I(x)")

    unnormalized_token_weight = torch.ones(
        policy_token_mask.shape[0],
        device=operation_credit.device,
        dtype=operation_credit.dtype,
    )
    for operation_index, credit in enumerate(operation_credit):
        operation_mask = operation_token_masks[operation_index].bool()

        unnormalized_token_weight[operation_mask] = 1.0 + float(reallocation_strength) * credit

    mean_weight = unnormalized_token_weight[policy_mask].mean()
    normalized_token_weight = unnormalized_token_weight / mean_weight
    return normalized_token_weight * policy_mask.to(normalized_token_weight.dtype)


def compute_outcome_anchored_advantage(
    outcome_level_advantage: torch.Tensor | float,
    normalized_token_weight: torch.Tensor,
) -> torch.Tensor:
    advantage = torch.as_tensor(
        outcome_level_advantage,
        device=normalized_token_weight.device,
        dtype=normalized_token_weight.dtype,
    )
    if torch.any(normalized_token_weight < 0):
        raise ValueError("OACR token weights must remain non-negative")

    outcome_anchored_advantage = advantage * normalized_token_weight
    nonzero = normalized_token_weight > 0
    if advantage > 0 and torch.any(outcome_anchored_advantage[nonzero] <= 0):
        raise AssertionError("Positive outcome advantage changed sign")
    if advantage < 0 and torch.any(outcome_anchored_advantage[nonzero] >= 0):
        raise AssertionError("Negative outcome advantage changed sign")
    return outcome_anchored_advantage
