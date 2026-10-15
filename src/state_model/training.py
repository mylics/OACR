from collections.abc import Mapping

import torch

from src.state_model.data import (
    QuestionStateCollator,
    TemporalOrderingCollator,
    aggregate_question_state_targets,
)
from src.state_model.objective import (
    compute_kg_state_outcome_model_loss,
    compute_state_fitting_loss,
    compute_temporal_ordering_loss,
)


def move_encoded(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


class StateModelTrainer:
    def __init__(self, model, tokenizer, config: Mapping):
        self.model = model
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        self.fit_count = 0

    def fit(self, rollouts, epochs=None):
        states, pairs = aggregate_question_state_targets(
            rollouts, float(self.config["outcome_reward_mixing_coefficient"])
        )
        return self.fit_rows(states, pairs, epochs=epochs)

    def fit_rows(self, states, pairs, epochs=None):
        if not states:
            raise ValueError("State-model fitting requires nonempty visited states")
        config = self.config
        epochs = int(config["epochs"] if epochs is None else epochs)
        batch_size = int(config["batch_size"])
        micro_batch_size = int(config.get("micro_batch_size", batch_size))
        if min(epochs, batch_size, micro_batch_size) < 1:
            raise ValueError("State-model epochs and batch sizes must be positive")
        device = next(self.model.parameters()).device
        state_collator = QuestionStateCollator(self.tokenizer, int(config["max_length"]))
        pair_collator = TemporalOrderingCollator(self.tokenizer, int(config["max_length"]))
        generator = torch.Generator().manual_seed(int(config["seed"]) + self.fit_count)
        self.fit_count += 1
        self.model.requires_grad_(True)
        self.model.train()
        updates = 0
        losses = []
        use_amp = bool(config.get("bf16", False)) and device.type == "cuda"
        try:
            for _ in range(epochs):
                indices = torch.randperm(len(states), generator=generator).tolist()
                for start in range(0, len(indices), batch_size):
                    selected = indices[start : start + batch_size]
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for offset in range(0, len(selected), micro_batch_size):
                        micro = selected[offset : offset + micro_batch_size]
                        batch = move_encoded(state_collator([states[i] for i in micro]), device)
                        target = batch.pop("target_outcome_reward")
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                            estimate = self.model(**batch)
                            fitting = compute_state_fitting_loss(estimate.float(), target)
                            ordering = fitting.new_zeros(())
                            if pairs:
                                pair_indices = torch.randint(
                                    len(pairs), (len(micro),), generator=generator
                                ).tolist()
                                pair_batch = pair_collator([pairs[i] for i in pair_indices])
                                earlier = self.model(**move_encoded(pair_batch["earlier"], device))
                                later = self.model(**move_encoded(pair_batch["later"], device))
                                ordering = compute_temporal_ordering_loss(
                                    earlier.float(),
                                    later.float(),
                                    pair_batch["rollout_outcome_reward"].to(device),
                                )
                            loss = compute_kg_state_outcome_model_loss(
                                fitting, ordering, float(config["temporal_ordering_strength"])
                            ) * (len(micro) / len(selected))
                        loss.backward()
                        total_loss += float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(config.get("max_grad_norm", 1.0)),
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    updates += 1
                    losses.append(total_loss)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            self.model.requires_grad_(False)
        return {
            "state_pairs": len(states),
            "ordering_pairs": len(pairs),
            "epochs": epochs,
            "optimizer_updates": updates,
            "loss": sum(losses) / len(losses),
        }
