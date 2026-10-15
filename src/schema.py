from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class KGOperationName(str, Enum):
    GET_TAIL_RELATIONS = "get_tail_relations"
    GET_HEAD_RELATIONS = "get_head_relations"
    GET_TAIL_ENTITIES = "get_tail_entities"
    GET_HEAD_ENTITIES = "get_head_entities"


RELATION_LIST_OPERATIONS = {
    KGOperationName.GET_TAIL_RELATIONS.value,
    KGOperationName.GET_HEAD_RELATIONS.value,
}
ENTITY_RETURN_OPERATIONS = {
    KGOperationName.GET_TAIL_ENTITIES.value,
    KGOperationName.GET_HEAD_ENTITIES.value,
}
SCHEMA_VALID_OPERATIONS = RELATION_LIST_OPERATIONS | ENTITY_RETURN_OPERATIONS


@dataclass(frozen=True)
class KGOperation:
    action: str
    entity: str = ""
    relation: str | None = None
    raw_payload: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def is_schema_valid(self) -> bool:
        if self.action not in SCHEMA_VALID_OPERATIONS:
            return False
        if not isinstance(self.entity, str) or not self.entity.strip():
            return False
        if self.action in ENTITY_RETURN_OPERATIONS:
            return isinstance(self.relation, str) and bool(self.relation.strip())
        return self.relation in {None, ""}

    @property
    def returns_entities(self) -> bool:
        return self.action in ENTITY_RETURN_OPERATIONS

    def to_dict(self) -> dict[str, str]:
        payload = {"action": self.action, "entity": self.entity}
        if self.relation:
            payload["relation"] = self.relation
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KGOperation":
        return cls(
            action=str(payload.get("action") or ""),
            entity=str(payload.get("entity") or ""),
            relation=(str(payload["relation"]) if payload.get("relation") is not None else None),
            raw_payload=dict(payload),
        )


@dataclass(frozen=True)
class KGObservation:
    payload: Mapping[str, Any]
    execution_succeeded: bool = True
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return not self.execution_succeeded or bool(self.error) or "error" in self.payload

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if self.error and "error" not in payload:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KGObservation":
        value = dict(payload)
        error = value.get("error")
        return cls(
            payload=value,
            execution_succeeded=not bool(error),
            error=str(error) if error else None,
        )


@dataclass(frozen=True)
class KGInteractionTurn:
    operation: KGOperation
    observation: KGObservation

    @property
    def is_invalid(self) -> bool:
        return not self.operation.is_schema_valid or self.observation.is_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.to_dict(),
            "observation": self.observation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KGInteractionTurn":
        return cls(
            operation=KGOperation.from_dict(payload.get("operation") or {}),
            observation=KGObservation.from_dict(payload.get("observation") or {}),
        )


@dataclass(frozen=True)
class KGSearchState:
    turns: tuple[KGInteractionTurn, ...] = ()

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def append(
        self,
        operation: KGOperation,
        observation: KGObservation,
    ) -> "KGSearchState":
        return KGSearchState(self.turns + (KGInteractionTurn(operation, observation),))

    def prefix(self, num_turns: int) -> "KGSearchState":
        if num_turns < 0 or num_turns > self.num_turns:
            raise IndexError(f"Invalid state prefix length: {num_turns}")
        return KGSearchState(self.turns[:num_turns])

    def prefixes(self, include_empty: bool = True) -> list["KGSearchState"]:
        start = 0 if include_empty else 1
        return [self.prefix(index) for index in range(start, self.num_turns + 1)]

    def to_list(self) -> list[dict[str, Any]]:
        return [turn.to_dict() for turn in self.turns]

    @classmethod
    def from_turns(cls, turns: Iterable[Mapping[str, Any]]) -> "KGSearchState":
        return cls(tuple(KGInteractionTurn.from_dict(turn) for turn in turns))


@dataclass(frozen=True)
class Trajectory:
    question_id: str
    question: str
    start_entities: tuple[Any, ...]
    gold_answer_set: tuple[str, ...]
    kg_search_state: KGSearchState
    predicted_answer_list: tuple[str, ...]
    messages: tuple[Mapping[str, str], ...] = ()

    @property
    def predicted_answer_set(self) -> set[str]:
        return set(self.predicted_answer_list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "start_entities": list(self.start_entities),
            "gold_answer_set": list(self.gold_answer_set),
            "predicted_answer_list": list(self.predicted_answer_list),
            "turns": self.kg_search_state.to_list(),
            "messages": [dict(message) for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Trajectory":
        return cls(
            question_id=str(payload.get("question_id") or payload.get("id") or ""),
            question=str(payload.get("question") or ""),
            start_entities=tuple(payload.get("start_entities") or ()),
            gold_answer_set=tuple(str(x) for x in payload.get("gold_answer_set") or ()),
            kg_search_state=KGSearchState.from_turns(payload.get("turns") or ()),
            predicted_answer_list=tuple(str(x) for x in payload.get("predicted_answer_list") or ()),
            messages=tuple(payload.get("messages") or ()),
        )


@dataclass(frozen=True)
class Rollout(Trajectory):
    outcome_reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.outcome_reward is not None:
            payload["outcome_reward"] = float(self.outcome_reward)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Rollout":
        trajectory = Trajectory.from_dict(payload)
        reward = payload.get("outcome_reward")
        return cls(
            question_id=trajectory.question_id,
            question=trajectory.question,
            start_entities=trajectory.start_entities,
            gold_answer_set=trajectory.gold_answer_set,
            kg_search_state=trajectory.kg_search_state,
            predicted_answer_list=trajectory.predicted_answer_list,
            messages=trajectory.messages,
            outcome_reward=float(reward) if reward is not None else None,
        )


def coerce_answer_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        for key in ("answers", "answer", "name", "text"):
            if key in value:
                return coerce_answer_sequence(value[key])
        return ()
    if isinstance(value, Sequence):
        answers: list[str] = []
        for item in value:
            answers.extend(coerce_answer_sequence(item))
        return tuple(dict.fromkeys(answer for answer in answers if answer))
    return (str(value),)
