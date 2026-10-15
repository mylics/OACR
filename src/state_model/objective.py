from __future__ import annotations

import torch
import torch.nn.functional as functional


def compute_state_fitting_loss(
    estimated_outcome_reward: torch.Tensor,
    target_outcome_reward: torch.Tensor,
) -> torch.Tensor:
    if estimated_outcome_reward.shape != target_outcome_reward.shape:
        raise ValueError("State estimates and targets must have the same shape")

    return functional.mse_loss(estimated_outcome_reward, target_outcome_reward)


def compute_temporal_ordering_loss(
    earlier_outcome_estimate: torch.Tensor,
    later_outcome_estimate: torch.Tensor,
    rollout_outcome_reward: torch.Tensor,
) -> torch.Tensor:
    if not (earlier_outcome_estimate.shape == later_outcome_estimate.shape == rollout_outcome_reward.shape):
        raise ValueError("Temporal ordering tensors must have the same shape")
    negative_log_sigmoid = functional.softplus(-(later_outcome_estimate - earlier_outcome_estimate))

    return (rollout_outcome_reward * negative_log_sigmoid).mean()


def compute_kg_state_outcome_model_loss(
    state_fitting_loss: torch.Tensor,
    temporal_ordering_loss: torch.Tensor,
    temporal_ordering_strength: float,
) -> torch.Tensor:
    if temporal_ordering_strength < 0:
        raise ValueError("temporal_ordering_strength must be non-negative")

    return state_fitting_loss + float(temporal_ordering_strength) * temporal_ordering_loss
