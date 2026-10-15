from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.privacy import sanitize_huggingface_metadata
from src.schema import KGSearchState
from src.serialization import serialize_question_state_for_model


@dataclass(frozen=True)
class KGStateOutcomeModelConfig:
    encoder_model: str
    dropout: float = 0.1
    pooling: str = "mean"


class KGStateOutcomeModel(nn.Module):
    CONFIG_FILE = "kg_state_outcome_model_config.json"
    WEIGHTS_FILE = "kg_state_outcome_model.pt"

    def __init__(self, config: KGStateOutcomeModelConfig) -> None:
        super().__init__()
        if config.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.model_config = config
        encoder_config = AutoConfig.from_pretrained(config.encoder_model)
        self.encoder = AutoModel.from_pretrained(
            config.encoder_model,
            config=encoder_config,
        )
        hidden_size = int(encoder_config.hidden_size)
        self.outcome_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, 1),
        )

    def resize_token_embeddings(self, vocabulary_size: int) -> None:
        self.encoder.resize_token_embeddings(vocabulary_size)

    def _pool(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_config.pooling == "cls":
            return hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_state * mask).sum(dim=1) / denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **encoder_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )
        pooled = self._pool(encoded.last_hidden_state, attention_mask)
        outcome_logit = self.outcome_head(pooled).squeeze(-1)
        return torch.sigmoid(outcome_logit)

    def save_pretrained(
        self,
        output_dir: str | Path,
        tokenizer: Any | None = None,
    ) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        sanitize_huggingface_metadata(self, tokenizer)
        encoder_dir = destination / "encoder"
        self.encoder.save_pretrained(encoder_dir, safe_serialization=True)
        torch.save(self.outcome_head.state_dict(), destination / self.WEIGHTS_FILE)
        with (destination / self.CONFIG_FILE).open("w", encoding="utf-8") as handle:
            portable_config = asdict(self.model_config)
            portable_config["encoder_model"] = "encoder"
            json.dump(portable_config, handle, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(destination)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "KGStateOutcomeModel":
        source = Path(model_dir)
        with (source / cls.CONFIG_FILE).open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        encoder_reference = Path(str(raw_config["encoder_model"]))
        if not encoder_reference.is_absolute():
            raw_config["encoder_model"] = str(source / encoder_reference)
        config = KGStateOutcomeModelConfig(**raw_config)
        model = cls(config)
        outcome_head_state_dict = torch.load(
            source / cls.WEIGHTS_FILE,
            map_location=map_location,
            weights_only=True,
        )
        model.outcome_head.load_state_dict(outcome_head_state_dict)
        return model


class KGStateOutcomeScorer:
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cpu",
        max_length: int = 512,
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = KGStateOutcomeModel.from_pretrained(
            model_dir,
            map_location=self.device,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def score_serialized_question_states(
        self,
        serialized_question_states: Sequence[str],
    ) -> list[float]:
        outcome_estimates: list[float] = []
        for start in range(0, len(serialized_question_states), self.batch_size):
            batch_text = list(serialized_question_states[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            estimates = self.model(**encoded)
            outcome_estimates.extend(float(value) for value in estimates.float().cpu().tolist())
        return outcome_estimates

    def score(
        self,
        question: str,
        kg_search_states: Sequence[KGSearchState],
    ) -> list[float]:
        serialized = [
            serialize_question_state_for_model(question, kg_search_state)
            for kg_search_state in kg_search_states
        ]
        return self.score_serialized_question_states(serialized)
