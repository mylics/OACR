from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from src.schema import KGObservation, KGOperation, KGSearchState


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
SEARCH_BLOCK_PATTERN = re.compile(r"<search>.*?</search>", flags=re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>.*?</answer>", flags=re.DOTALL)


def normalize_serialized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            normalize_serialized_text(key): _canonicalize_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        normalized = [_canonicalize_value(item) for item in value]
        unique = {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in normalized}
        return [unique[key] for key in sorted(unique)]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return normalize_serialized_text(value)


def canonicalize_operation(operation: KGOperation) -> dict[str, Any]:
    return _canonicalize_value(operation.to_dict())


def canonicalize_observation(observation: KGObservation) -> dict[str, Any]:
    return _canonicalize_value(observation.to_dict())


def canonicalize_kg_search_state(kg_search_state: KGSearchState) -> str:
    canonical_turns = [
        {
            "operation": canonicalize_operation(turn.operation),
            "observation": canonicalize_observation(turn.observation),
        }
        for turn in kg_search_state.turns
    ]
    return json.dumps(
        canonical_turns,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonicalize_question_state(question: str, kg_search_state: KGSearchState) -> str:
    return json.dumps(
        {
            "question": normalize_serialized_text(question),
            "state": json.loads(canonicalize_kg_search_state(kg_search_state)),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def question_state_key(question: str, kg_search_state: KGSearchState) -> str:
    return canonicalize_question_state(question, kg_search_state)


def serialize_question_state_for_model(
    question: str,
    kg_search_state: KGSearchState,
) -> str:
    state_text = canonicalize_kg_search_state(kg_search_state)
    if not kg_search_state.turns:
        state_text = "[EMPTY_STATE]"
    return f"[QUESTION]\n{normalize_serialized_text(question)}\n[KG_SEARCH_STATE]\n{state_text}"


def format_operation_block(operation: KGOperation) -> str:
    payload = json.dumps(operation.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"<search>{payload}</search>"


def format_observation_block(observation: KGObservation) -> str:
    payload = json.dumps(observation.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"<observation>{payload}</observation>"


def format_answer_block(predicted_answer_list: Sequence[str]) -> str:
    payload = json.dumps(
        {"answers": list(predicted_answer_list)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<answer>{payload}</answer>"


def parse_operation_block(text: str) -> KGOperation | None:
    matches = list(SEARCH_BLOCK_PATTERN.finditer(text or ""))
    if not matches:
        return None
    if len(matches) != 1:
        return KGOperation(
            action="",
            raw_payload={"error": "multiple KG operations in one assistant turn"},
        )
    match = matches[0]
    block = match.group(0)
    payload_text = block[len("<search>") : -len("</search>")].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return KGOperation(action="", raw_payload={"unparsed": payload_text})
    if not isinstance(payload, Mapping):
        return KGOperation(action="", raw_payload={"unparsed": payload})
    return KGOperation.from_dict(payload)


def parse_answer_block(text: str) -> tuple[str, ...] | None:
    matches = list(ANSWER_BLOCK_PATTERN.finditer(text or ""))
    if not matches:
        return None
    block = matches[-1].group(0)
    payload_text = block[len("<answer>") : -len("</answer>")].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return ()
    answers = payload.get("answers") if isinstance(payload, Mapping) else None
    if isinstance(answers, str):
        return (answers,)
    if isinstance(answers, Sequence):
        return tuple(str(answer) for answer in answers if answer is not None)
    return ()


def extract_policy_block_character_spans(text: str) -> list[tuple[int, int]]:
    spans = [
        match.span()
        for pattern in (THINK_BLOCK_PATTERN, SEARCH_BLOCK_PATTERN, ANSWER_BLOCK_PATTERN)
        for match in pattern.finditer(text or "")
    ]
    return sorted(spans)


def extract_search_block_character_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in SEARCH_BLOCK_PATTERN.finditer(text or "")]


def find_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
    start: int = 0,
    stop: int | None = None,
) -> int:
    if not subsequence:
        return -1
    stop = len(sequence) if stop is None else min(stop, len(sequence))
    last = stop - len(subsequence)
    for index in range(max(start, 0), last + 1):
        if list(sequence[index : index + len(subsequence)]) == list(subsequence):
            return index
    return -1


def block_token_spans(
    token_ids: Sequence[int],
    tokenizer: Any,
    start_tag: str,
    end_tag: str,
    start: int = 0,
    stop: int | None = None,
) -> list[tuple[int, int]]:
    start_ids = tokenizer.encode(start_tag, add_special_tokens=False)
    end_ids = tokenizer.encode(end_tag, add_special_tokens=False)
    spans: list[tuple[int, int]] = []
    cursor = start
    stop = len(token_ids) if stop is None else stop
    while cursor < stop:
        block_start = find_subsequence(token_ids, start_ids, cursor, stop)
        if block_start < 0:
            break
        content_start = block_start + len(start_ids)
        end_start = find_subsequence(token_ids, end_ids, content_start, stop)
        if end_start < 0:
            break
        block_end = end_start + len(end_ids)
        spans.append((block_start, block_end))
        cursor = block_end
    return spans


def values_observed_in_state(kg_search_state: KGSearchState) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)
        elif value is not None:
            text = normalize_serialized_text(value)
            if text:
                values.add(text)

    for turn in kg_search_state.turns:
        visit(turn.observation.to_dict())
    return values


def entity_values_observed_in_state(
    kg_search_state: KGSearchState,
) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)
        elif value is not None:
            normalized = normalize_serialized_text(value)
            if normalized:
                values.add(normalized)

    for turn in kg_search_state.turns:
        if turn.operation.returns_entities and not turn.observation.is_error:
            visit(turn.observation.to_dict())
    return values
