from __future__ import annotations

import torch


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_value = mask.to(value.dtype)
    denominator = mask_value.sum().clamp_min(1.0)
    return (value * mask_value).sum() / denominator


def masked_sequence_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or mask.shape != value.shape:
        raise ValueError("Expected aligned [batch, sequence] tensors")
    mask_value = mask.to(value.dtype)
    token_count = mask_value.sum(dim=-1)
    valid_sequence = token_count > 0
    if not torch.any(valid_sequence):
        return value.sum() * 0.0
    sequence_mean = (value * mask_value).sum(dim=-1) / token_count.clamp_min(1.0)
    return sequence_mean[valid_sequence].mean()


def compute_clipped_surrogate(
    importance_ratio: torch.Tensor,
    outcome_anchored_advantage: torch.Tensor,
    policy_clip_epsilon: float,
) -> torch.Tensor:
    epsilon = float(policy_clip_epsilon)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("policy_clip_epsilon must be in (0, 1)")
    unclipped = importance_ratio * outcome_anchored_advantage
    clipped = importance_ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped * outcome_anchored_advantage
    return torch.minimum(unclipped, clipped)


def compute_sampled_kl_penalty(
    policy_log_probability: torch.Tensor,
    reference_log_probability: torch.Tensor,
) -> torch.Tensor:
    log_reference_over_policy = reference_log_probability - policy_log_probability
    return torch.exp(log_reference_over_policy) - log_reference_over_policy - 1.0


def compute_oacr_objective(
    policy_log_probability: torch.Tensor,
    old_policy_log_probability: torch.Tensor,
    reference_log_probability: torch.Tensor,
    outcome_anchored_advantage: torch.Tensor,
    policy_token_mask: torch.Tensor,
    policy_clip_epsilon: float,
    kl_penalty_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected_shape = policy_log_probability.shape
    for name, tensor in {
        "old_policy_log_probability": old_policy_log_probability,
        "reference_log_probability": reference_log_probability,
        "outcome_anchored_advantage": outcome_anchored_advantage,
        "policy_token_mask": policy_token_mask,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if kl_penalty_coefficient < 0:
        raise ValueError("kl_penalty_coefficient must be non-negative")

    importance_ratio = torch.exp(policy_log_probability - old_policy_log_probability)
    clipped_surrogate = compute_clipped_surrogate(
        importance_ratio,
        outcome_anchored_advantage,
        policy_clip_epsilon,
    )
    policy_loss = -masked_sequence_mean(clipped_surrogate, policy_token_mask)
    token_kl = compute_sampled_kl_penalty(
        policy_log_probability,
        reference_log_probability,
    )
    kl_penalty = masked_sequence_mean(token_kl, policy_token_mask)

    oacr_loss = policy_loss + float(kl_penalty_coefficient) * kl_penalty
    clip_fraction = masked_mean(
        (
            (importance_ratio < 1.0 - float(policy_clip_epsilon))
            | (importance_ratio > 1.0 + float(policy_clip_epsilon))
        ).to(policy_log_probability.dtype),
        policy_token_mask,
    )
    return oacr_loss, {
        "policy_loss": policy_loss.detach(),
        "kl_penalty": kl_penalty.detach(),
        "clip_fraction": clip_fraction.detach(),
    }
