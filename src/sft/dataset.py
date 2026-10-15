from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.io import read_jsonl
from src.serialization import block_token_spans


POLICY_BLOCK_TAGS = (
    ("<think>", "</think>"),
    ("<search>", "</search>"),
    ("<answer>", "</answer>"),
)


def _coerce_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.squeeze(0).tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def tokenize_policy_trajectory(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    full_ids = _coerce_token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if len(full_ids) > max_length:
        raise ValueError(f"Serialized trajectory length {len(full_ids)} exceeds {max_length}")

    policy_token_mask = torch.zeros(len(full_ids), dtype=torch.bool)
    operation_token_masks: list[torch.Tensor] = []
    previous_length = 0

    for message_index, message in enumerate(messages):
        prefix_ids = _coerce_token_ids(
            tokenizer.apply_chat_template(
                list(messages[: message_index + 1]),
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        message_end = len(prefix_ids)
        if message["role"] == "assistant":
            for start_tag, end_tag in POLICY_BLOCK_TAGS:
                for start, end in block_token_spans(
                    full_ids,
                    tokenizer,
                    start_tag,
                    end_tag,
                    start=previous_length,
                    stop=message_end,
                ):
                    policy_token_mask[start:end] = True
            search_spans = block_token_spans(
                full_ids,
                tokenizer,
                "<search>",
                "</search>",
                start=previous_length,
                stop=message_end,
            )
            if search_spans:
                operation_mask = torch.zeros(len(full_ids), dtype=torch.bool)
                for start, end in search_spans:
                    operation_mask[start:end] = True
                operation_token_masks.append(operation_mask)
            else:
                answer_spans = block_token_spans(
                    full_ids,
                    tokenizer,
                    "<answer>",
                    "</answer>",
                    start=previous_length,
                    stop=message_end,
                )
                if not answer_spans:
                    operation_token_masks.append(torch.zeros(len(full_ids), dtype=torch.bool))
        previous_length = message_end

    if not policy_token_mask.any():
        raise ValueError("Trajectory contains no trainable policy blocks")
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[~policy_token_mask] = -100
    attention_mask = torch.ones_like(input_ids)
    operation_token_mask = (
        torch.stack(operation_token_masks)
        if operation_token_masks
        else torch.zeros((0, len(full_ids)), dtype=torch.bool)
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "policy_token_mask": policy_token_mask,
        "operation_token_mask": operation_token_mask,
    }


class SFTTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.rows[index].get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {index} has no message list")
        return tokenize_policy_trajectory(
            messages,
            self.tokenizer,
            self.max_length,
        )


@dataclass
class SFTBatchCollator:
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
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        for index, example in enumerate(examples):
            length = example["input_ids"].shape[0]
            input_ids[index, :length] = example["input_ids"]
            attention_mask[index, :length] = example["attention_mask"]
            labels[index, :length] = example["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
