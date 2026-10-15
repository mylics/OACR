from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

import torch

from src.kg.client import HTTPKnowledgeGraphTool
from src.metrics import compute_outcome_reward
from src.prompts import KGQA_SYSTEM_PROMPT, format_question_prompt
from src.schema import KGObservation, KGOperation, KGSearchState, Rollout
from src.serialization import (
    format_observation_block,
    parse_answer_block,
    parse_operation_block,
)


@contextmanager
def preserve_training_mode(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class AgenticKGQAPolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        kg_service_url: str,
        kg_request_timeout: int,
        max_turns: int,
        max_new_tokens_per_turn: int,
        temperature: float,
        top_p: float,
        max_prompt_length: int = 2048,
        max_response_length: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.kg_service_url = kg_service_url
        self.kg_request_timeout = int(kg_request_timeout)
        self.max_turns = int(max_turns)
        self.max_new_tokens_per_turn = int(max_new_tokens_per_turn)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)
        self.remaining_response_tokens = self.max_response_length

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def _generate_assistant_turn(
        self,
        messages: Sequence[dict[str, str]],
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        if len(messages) == 2 and input_ids.shape[1] > self.max_prompt_length:
            raise ValueError("Question prompt exceeds the configured token limit")
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": min(self.max_new_tokens_per_turn, self.remaining_response_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        generated = self.model.generate(
            **generation_kwargs,
        )
        new_token_ids = generated[0, input_ids.shape[1] :].tolist()
        self.remaining_response_tokens -= len(new_token_ids)
        special_ids = set(self.tokenizer.all_special_ids)
        while new_token_ids and new_token_ids[-1] in special_ids:
            new_token_ids.pop()
        return self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def sample_rollout(
        self,
        question_id: str,
        question: str,
        start_entities: Sequence[Any],
        gold_answer_set: Sequence[str],
        outcome_reward_mixing_coefficient: float,
    ) -> Rollout:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": KGQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_question_prompt(question, start_entities),
            },
        ]
        kg_search_state = KGSearchState()
        self.remaining_response_tokens = self.max_response_length
        predicted_answer_list: tuple[str, ...] = ()
        kg_tool = HTTPKnowledgeGraphTool(
            service_url=self.kg_service_url,
            question=question,
            start_entities=start_entities,
            request_timeout=self.kg_request_timeout,
        )

        with preserve_training_mode(self.model):
            for _ in range(self.max_turns):
                if self.remaining_response_tokens <= 0:
                    break
                assistant_content = self._generate_assistant_turn(messages)
                messages.append({"role": "assistant", "content": assistant_content})
                parsed_answer = parse_answer_block(assistant_content)
                operation = parse_operation_block(assistant_content)
                if parsed_answer is not None and operation is None:
                    predicted_answer_list = parsed_answer
                    break

                if parsed_answer is not None and operation is not None:
                    operation = KGOperation(
                        action="",
                        raw_payload={"error": "answer and KG operation emitted together"},
                    )
                    observation = KGObservation(
                        payload={"error": "answer and KG operation emitted together"},
                        execution_succeeded=False,
                        error="answer and KG operation emitted together",
                    )
                elif operation is None:
                    operation = KGOperation(
                        action="",
                        raw_payload={"unparsed_assistant_turn": assistant_content},
                    )
                    observation = KGObservation(
                        payload={"error": "missing complete <search> operation"},
                        execution_succeeded=False,
                        error="missing complete <search> operation",
                    )
                else:
                    observation = kg_tool.execute(operation)
                kg_search_state = kg_search_state.append(operation, observation)
                messages.append(
                    {
                        "role": "user",
                        "content": format_observation_block(observation),
                    }
                )

        outcome_reward = compute_outcome_reward(
            predicted_answer_list,
            gold_answer_set,
            outcome_reward_mixing_coefficient,
        )
        return Rollout(
            question_id=question_id,
            question=question,
            start_entities=tuple(start_entities),
            gold_answer_set=tuple(gold_answer_set),
            kg_search_state=kg_search_state,
            predicted_answer_list=predicted_answer_list,
            messages=tuple(messages),
            outcome_reward=outcome_reward,
        )
