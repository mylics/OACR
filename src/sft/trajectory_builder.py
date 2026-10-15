from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.kg.interface import KnowledgeGraphToolInterface
from src.metrics import normalize_answer_entity
from src.prompts import KGQA_SYSTEM_PROMPT, format_question_prompt
from src.schema import KGOperation, KGSearchState, Trajectory
from src.serialization import (
    entity_values_observed_in_state,
    format_answer_block,
    format_observation_block,
    format_operation_block,
)


class InvalidSFTTrajectory(ValueError):
    __slots__ = ()


def default_operation_reasoning(operation: KGOperation) -> str:
    if operation.action.endswith("_relations"):
        direction = "outgoing" if "tail" in operation.action else "incoming"
        return f"Inspect the available {direction} relations for {operation.entity}."
    direction = "outgoing" if "tail" in operation.action else "incoming"
    return f"Follow {direction} relation {operation.relation} from {operation.entity}."


@dataclass
class SFTTrajectoryBuilder:
    kg_tool_factory: Callable[[str, Sequence[Any]], KnowledgeGraphToolInterface]

    def build(self, record: Mapping[str, Any]) -> Trajectory:
        question_id = str(record.get("question_id") or record.get("id") or "")
        question = str(record.get("question") or "").strip()
        start_entities = tuple(record.get("start_entities") or ())
        gold_answer_set = tuple(str(answer) for answer in record.get("gold_answer_set") or ())
        raw_operations = list(record.get("gold_operations") or ())
        if not question:
            raise InvalidSFTTrajectory("missing question")
        if not raw_operations:
            raise InvalidSFTTrajectory("missing gold operations")
        if not gold_answer_set:
            raise InvalidSFTTrajectory("missing gold answer set")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": KGQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_question_prompt(question, start_entities),
            },
        ]
        kg_search_state = KGSearchState()
        kg_tool = self.kg_tool_factory(question, start_entities)

        for raw_operation in raw_operations:
            operation = KGOperation.from_dict(raw_operation)
            if not operation.is_schema_valid:
                raise InvalidSFTTrajectory("schema-invalid operation")
            observation = kg_tool.execute(operation)
            if observation.is_error:
                raise InvalidSFTTrajectory(f"operation execution failed: {observation.error}")
            kg_search_state = kg_search_state.append(operation, observation)
            reasoning = str(raw_operation.get("thought") or default_operation_reasoning(operation))
            messages.append(
                {
                    "role": "assistant",
                    "content": (f"<think>{reasoning}</think>\n{format_operation_block(operation)}"),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": format_observation_block(observation),
                }
            )

        observed_values = {
            normalize_answer_entity(value) for value in entity_values_observed_in_state(kg_search_state)
        }

        observed_gold_answers = [
            answer for answer in gold_answer_set if normalize_answer_entity(answer) in observed_values
        ]
        if not observed_gold_answers:
            raise InvalidSFTTrajectory("gold path did not return any gold answer entity")

        messages.append(
            {
                "role": "assistant",
                "content": (
                    "<think>The retrieved entities satisfy the question.</think>\n"
                    f"{format_answer_block(observed_gold_answers)}"
                ),
            }
        )
        return Trajectory(
            question_id=question_id,
            question=question,
            start_entities=start_entities,
            gold_answer_set=gold_answer_set,
            kg_search_state=kg_search_state,
            predicted_answer_list=tuple(observed_gold_answers),
            messages=tuple(messages),
        )
