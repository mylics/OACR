import torch

from src.metrics import compute_group_outcome_advantage
from src.rl.objective import compute_oacr_objective
from src.rl.training import (
    OACRTrainingBatchCollator,
    build_outcome_anchored_training_example,
    build_outcome_only_training_example,
    compute_causal_token_log_probabilities,
)


class PolicyTrainer:
    def __init__(
        self, policy, reference, tokenizer, rollout_policy, optimizer, accelerator, config, state_scorer=None
    ):
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.rollout_policy = rollout_policy
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.config = config
        self.state_scorer = state_scorer
        self.collator = OACRTrainingBatchCollator(tokenizer.pad_token_id)
        self.optimizer_updates = 0

    def _batch(self, examples):
        return {key: value.to(self.accelerator.device) for key, value in self.collator(examples).items()}

    def step(self, questions):
        config = self.config
        examples, rollouts, snapshots = [], [], []
        group_size = int(config["rollout_group_size"])
        for question in questions:
            group = [
                self.rollout_policy.sample_rollout(
                    question_id=str(question["question_id"]),
                    question=str(question["question"]),
                    start_entities=question.get("start_entities", ()),
                    gold_answer_set=question.get("gold_answer_set", ()),
                    outcome_reward_mixing_coefficient=float(config["outcome_reward_mixing_coefficient"]),
                )
                for _ in range(group_size)
            ]
            rewards = torch.tensor([rollout.outcome_reward for rollout in group])
            advantages = compute_group_outcome_advantage(
                rewards, [0] * group_size, float(config["advantage_epsilon"])
            )
            for rollout, advantage in zip(group, advantages, strict=True):
                if self.state_scorer is None:
                    example = build_outcome_only_training_example(
                        rollout, advantage, self.tokenizer, int(config["max_length"])
                    )
                else:
                    example = build_outcome_anchored_training_example(
                        rollout,
                        float(advantage),
                        self.tokenizer,
                        int(config["max_length"]),
                        self.state_scorer,
                        float(config["outcome_reward_change_bound"]),
                        float(config["invalid_operation_credit"]),
                        float(config["reallocation_strength"]),
                    )
                examples.append(example)
            rollouts.extend(group)
        for example in examples:
            batch = self._batch([example])
            with torch.no_grad():
                old = (
                    compute_causal_token_log_probabilities(
                        self.policy, batch["input_ids"], batch["attention_mask"]
                    )
                    .detach()
                    .cpu()
                )
                reference = (
                    compute_causal_token_log_probabilities(
                        self.reference, batch["input_ids"], batch["attention_mask"]
                    )
                    .detach()
                    .cpu()
                )
            snapshots.append((old, reference))
        local_mini_questions = int(config["mini_batch_size"]) // self.accelerator.num_processes
        mini_size = local_mini_questions * group_size
        micro_size = int(config["micro_batch_size"])
        losses, divergences = [], []
        for _ in range(int(config["updates_per_rollout_group"])):
            for start in range(0, len(examples), mini_size):
                indices = list(range(start, min(start + mini_size, len(examples))))
                self.optimizer.zero_grad(set_to_none=True)
                mini_loss = mini_kl = 0.0
                for offset in range(0, len(indices), micro_size):
                    micro = indices[offset : offset + micro_size]
                    batch = self._batch([examples[i] for i in micro])
                    old = torch.zeros_like(batch["outcome_anchored_advantage"])
                    reference = torch.zeros_like(old)
                    for row, index in enumerate(micro):
                        old_values, reference_values = snapshots[index]
                        length = old_values.shape[1]
                        old[row, :length] = old_values[0].to(old.device)
                        reference[row, :length] = reference_values[0].to(reference.device)
                    probability = compute_causal_token_log_probabilities(
                        self.policy, batch["input_ids"], batch["attention_mask"]
                    )
                    loss, metrics = compute_oacr_objective(
                        probability,
                        old,
                        reference,
                        batch["outcome_anchored_advantage"],
                        batch["policy_token_mask"],
                        float(config["policy_clip_epsilon"]),
                        float(config["kl_penalty_coefficient"]),
                    )
                    weight = len(micro) / len(indices)
                    self.accelerator.backward(loss * weight)
                    mini_loss += float(loss.detach()) * weight
                    mini_kl += float(metrics["kl_penalty"]) * weight
                self.accelerator.clip_grad_norm_(self.policy.parameters(), float(config["max_grad_norm"]))
                self.optimizer.step()
                self.optimizer_updates += 1
                losses.append(mini_loss)
                divergences.append(mini_kl)
        self.optimizer.zero_grad(set_to_none=True)
        return rollouts, {
            "loss": sum(losses) / len(losses),
            "kl": sum(divergences) / len(divergences),
            "reward": sum(float(r.outcome_reward) for r in rollouts) / len(rollouts),
            "optimizer_updates": self.optimizer_updates,
        }
