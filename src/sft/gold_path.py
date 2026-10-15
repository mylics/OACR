from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.kg.interface import KnowledgeGraphToolInterface
from src.kg.local import LocalKnowledgeGraph
from src.metrics import normalize_answer_entity
from src.prompts import visible_start_entity_names
from src.schema import KGOperation


@dataclass(frozen=True)
class GoldPathEdge:
    current_entity: str
    relation: str
    next_entity: str
    direction: str


def find_shortest_gold_path(
    knowledge_graph: LocalKnowledgeGraph,
    start_entities: Sequence[Any],
    gold_answer_set: Sequence[str],
    max_path_hops: int,
) -> list[GoldPathEdge] | None:
    if max_path_hops < 1:
        raise ValueError("max_path_hops must be positive")
    gold_keys = {normalize_answer_entity(answer) for answer in gold_answer_set}
    queue = deque((entity, [], {entity}) for entity in visible_start_entity_names(start_entities))
    while queue:
        current_entity, path, visited = queue.popleft()
        if path and normalize_answer_entity(current_entity) in gold_keys:
            return path
        if len(path) >= max_path_hops:
            continue

        for relation, next_entities in sorted(knowledge_graph.outgoing.get(current_entity, {}).items()):
            for next_entity in next_entities:
                if next_entity in visited:
                    continue
                edge = GoldPathEdge(
                    current_entity=current_entity,
                    relation=relation,
                    next_entity=next_entity,
                    direction="tail",
                )
                queue.append((next_entity, path + [edge], visited | {next_entity}))
        for relation, next_entities in sorted(knowledge_graph.incoming.get(current_entity, {}).items()):
            for next_entity in next_entities:
                if next_entity in visited:
                    continue
                edge = GoldPathEdge(
                    current_entity=current_entity,
                    relation=relation,
                    next_entity=next_entity,
                    direction="head",
                )
                queue.append((next_entity, path + [edge], visited | {next_entity}))
    return None


def gold_path_to_operations(
    gold_path: Sequence[GoldPathEdge],
) -> list[dict[str, str]]:
    operations: list[dict[str, str]] = []
    for edge in gold_path:
        operations.append(
            KGOperation(
                action=f"get_{edge.direction}_relations",
                entity=edge.current_entity,
            ).to_dict()
        )
        operations.append(
            KGOperation(
                action=f"get_{edge.direction}_entities",
                entity=edge.current_entity,
                relation=edge.relation,
            ).to_dict()
        )
    return operations


def find_shortest_gold_path_with_tool(
    knowledge_graph_tool: KnowledgeGraphToolInterface,
    start_entities: Sequence[Any],
    gold_answer_set: Sequence[str],
    max_path_hops: int,
) -> list[GoldPathEdge] | None:
    if max_path_hops < 1:
        raise ValueError("max_path_hops must be positive")
    gold_keys = {normalize_answer_entity(answer) for answer in gold_answer_set}
    queue = deque(
        (entity, [], {normalize_answer_entity(entity)})
        for entity in visible_start_entity_names(start_entities)
    )
    while queue:
        current_entity, path, visited = queue.popleft()
        if path and normalize_answer_entity(current_entity) in gold_keys:
            return path
        if len(path) >= max_path_hops:
            continue

        for direction in ("tail", "head"):
            relation_observation = knowledge_graph_tool.execute(
                KGOperation(
                    action=f"get_{direction}_relations",
                    entity=current_entity,
                )
            )
            if relation_observation.is_error:
                continue
            relations = relation_observation.payload.get("relations") or ()
            for relation in relations:
                relation_text = str(relation)
                entity_observation = knowledge_graph_tool.execute(
                    KGOperation(
                        action=f"get_{direction}_entities",
                        entity=current_entity,
                        relation=relation_text,
                    )
                )
                if entity_observation.is_error:
                    continue
                for next_entity_value in entity_observation.payload.get("entities") or ():
                    next_entity = str(next_entity_value).strip()
                    next_key = normalize_answer_entity(next_entity)
                    if not next_entity or next_key in visited:
                        continue
                    edge = GoldPathEdge(
                        current_entity=current_entity,
                        relation=relation_text,
                        next_entity=next_entity,
                        direction=direction,
                    )
                    next_path = path + [edge]
                    if next_key in gold_keys:
                        return next_path
                    queue.append(
                        (
                            next_entity,
                            next_path,
                            visited | {next_key},
                        )
                    )
    return None


def build_gold_path_record(
    knowledge_graph: LocalKnowledgeGraph,
    question_record: Mapping[str, Any],
    max_path_hops: int,
) -> dict[str, Any] | None:
    gold_path = find_shortest_gold_path(
        knowledge_graph=knowledge_graph,
        start_entities=question_record.get("start_entities") or (),
        gold_answer_set=question_record.get("gold_answer_set") or (),
        max_path_hops=max_path_hops,
    )
    if not gold_path:
        return None
    return {
        "question_id": str(question_record.get("question_id") or question_record.get("id") or ""),
        "question": str(question_record["question"]),
        "start_entities": list(question_record.get("start_entities") or ()),
        "gold_answer_set": list(question_record.get("gold_answer_set") or ()),
        "gold_operations": gold_path_to_operations(gold_path),
    }


def build_gold_path_record_with_tool(
    knowledge_graph_tool: KnowledgeGraphToolInterface,
    question_record: Mapping[str, Any],
    max_path_hops: int,
) -> dict[str, Any] | None:
    gold_path = find_shortest_gold_path_with_tool(
        knowledge_graph_tool=knowledge_graph_tool,
        start_entities=question_record.get("start_entities") or (),
        gold_answer_set=question_record.get("gold_answer_set") or (),
        max_path_hops=max_path_hops,
    )
    if not gold_path:
        return None
    return {
        "question_id": str(question_record.get("question_id") or question_record.get("id") or ""),
        "question": str(question_record["question"]),
        "start_entities": list(question_record.get("start_entities") or ()),
        "gold_answer_set": list(question_record.get("gold_answer_set") or ()),
        "gold_operations": gold_path_to_operations(gold_path),
    }
