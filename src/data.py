from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.schema import coerce_answer_sequence


def _deduplicate(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _load_json_or_jsonl(path: str | Path) -> Any:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_cwq_questions(path: str | Path) -> list[dict[str, Any]]:
    raw_items = _load_json_or_jsonl(path)
    if not isinstance(raw_items, list):
        raise ValueError("CWQ input must be a JSON array or JSONL")
    output = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        question = str(
            item.get("question") or item.get("machine_question") or item.get("webqsp_question") or ""
        ).strip()
        start_entities = [
            {
                "name": str(entity.get("name") or "").strip(),
                "mid": str(entity.get("mid") or "").strip(),
            }
            for entity in item.get("entities") or ()
            if isinstance(entity, Mapping) and entity.get("name")
        ]
        answers = []
        for answer in item.get("answers") or ():
            if isinstance(answer, Mapping):
                value = answer.get("answer") or answer.get("name") or answer.get("EntityName")
                if value is not None:
                    answers.append(value)
            else:
                answers.append(answer)
        if question and answers:
            output.append(
                {
                    "question_id": str(item.get("ID") or item.get("id") or f"cwq-{index}"),
                    "question": question,
                    "start_entities": start_entities,
                    "gold_answer_set": _deduplicate(answers),
                }
            )
    return output


def normalize_webqsp_questions(path: str | Path) -> list[dict[str, Any]]:
    raw = _load_json_or_jsonl(path)
    if isinstance(raw, Mapping):
        raw_items = raw.get("Questions") or raw.get("questions") or ()
    else:
        raw_items = raw
    if not isinstance(raw_items, list):
        raise ValueError("WebQSP input must contain a Questions array")

    output = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        question = str(
            item.get("RawQuestion") or item.get("ProcessedQuestion") or item.get("question") or ""
        ).strip()
        start_entities = []
        answers = []
        for parse in item.get("Parses") or item.get("parses") or ():
            if not isinstance(parse, Mapping):
                continue
            topic_name = parse.get("TopicEntityName")
            topic_mid = parse.get("TopicEntityMid")
            if topic_name:
                start_entities.append(
                    {
                        "name": str(topic_name).strip(),
                        "mid": str(topic_mid or "").strip(),
                    }
                )
            for answer in parse.get("Answers") or ():
                if isinstance(answer, Mapping):
                    value = answer.get("EntityName") or answer.get("answer") or answer.get("AnswerArgument")
                    if value is not None:
                        answers.append(value)
                else:
                    answers.extend(coerce_answer_sequence(answer))
        if question and answers:
            entity_keys = set()
            unique_entities = []
            for entity in start_entities:
                key = (entity["name"], entity["mid"])
                if key not in entity_keys:
                    entity_keys.add(key)
                    unique_entities.append(entity)
            output.append(
                {
                    "question_id": str(item.get("QuestionId") or item.get("id") or f"webqsp-{index}"),
                    "question": question,
                    "start_entities": unique_entities,
                    "gold_answer_set": _deduplicate(answers),
                }
            )
    return output


def normalize_grailqa_questions(path: str | Path) -> list[dict[str, Any]]:
    raw_items = _load_json_or_jsonl(path)
    if not isinstance(raw_items, list):
        raise ValueError("GrailQA input must be a JSON array or JSONL")
    output = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        question = str(item.get("question") or "").strip()
        answers = []
        for answer in item.get("answer") or item.get("answers") or ():
            if isinstance(answer, Mapping):
                value = answer.get("entity_name") or answer.get("answer") or answer.get("answer_argument")
                if value is not None:
                    answers.append(value)
            else:
                answers.append(answer)

        graph_query = item.get("graph_query") or {}
        nodes = graph_query.get("nodes") if isinstance(graph_query, Mapping) else ()
        start_entities = []
        for node in nodes or ():
            if not isinstance(node, Mapping):
                continue
            if str(node.get("node_type") or "").lower() != "entity":
                continue
            if node.get("question_node") in {1, "1", True}:
                continue
            name = node.get("friendly_name") or node.get("name") or node.get("entity_name")
            if name:
                start_entities.append(
                    {
                        "name": str(name).strip(),
                        "mid": str(node.get("id") or "").strip(),
                    }
                )
        if question and answers:
            output.append(
                {
                    "question_id": str(
                        item.get("qid") or item.get("question_id") or item.get("id") or f"grailqa-{index}"
                    ),
                    "question": question,
                    "start_entities": start_entities,
                    "gold_answer_set": _deduplicate(answers),
                }
            )
    return output


def normalize_canonical_questions(path: str | Path) -> list[dict[str, Any]]:
    raw_items = _load_json_or_jsonl(path)
    if not isinstance(raw_items, list):
        raise ValueError("Canonical input must be a JSON array or JSONL")
    output = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        question = str(item.get("question") or "").strip()
        answers = coerce_answer_sequence(item.get("gold_answer_set"))
        if question and answers:
            output.append(
                {
                    "question_id": str(item.get("question_id") or item.get("id") or f"question-{index}"),
                    "question": question,
                    "start_entities": list(item.get("start_entities") or ()),
                    "gold_answer_set": list(answers),
                }
            )
    return output
