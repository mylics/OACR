from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.schema import KGObservation, KGOperation


class LocalKnowledgeGraph:
    def __init__(self, triples: Iterable[tuple[str, str, str]]) -> None:
        self.outgoing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.incoming: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for head_entity, relation, tail_entity in triples:
            head = str(head_entity).strip()
            edge = str(relation).strip()
            tail = str(tail_entity).strip()
            if not head or not edge or not tail:
                continue
            if tail not in self.outgoing[head][edge]:
                self.outgoing[head][edge].append(tail)
            if head not in self.incoming[tail][edge]:
                self.incoming[tail][edge].append(head)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LocalKnowledgeGraph":
        def triples():
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise ValueError(f"Expected an object at line {line_number}")
                    head = record.get("head") or record.get("head_entity")
                    relation = record.get("relation")
                    tail = record.get("tail") or record.get("tail_entity")
                    if head is None or relation is None or tail is None:
                        raise ValueError(f"Missing head, relation, or tail at line {line_number}")
                    yield (str(head), str(relation), str(tail))

        return cls(triples())


class LocalKnowledgeGraphTool:
    def __init__(
        self,
        knowledge_graph: LocalKnowledgeGraph,
        start_entities: Sequence[Any],
        top_k: int | None = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.start_entities = list(start_entities)
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self.visible_entities = {
            str(entity.get("name") if isinstance(entity, Mapping) else entity).strip()
            for entity in self.start_entities
        }
        self.visible_entities.discard("")
        self.relation_cache: dict[tuple[str, str], set[str]] = {}

    def _error(self, message: str) -> KGObservation:
        return KGObservation(
            payload={"error": message},
            execution_succeeded=False,
            error=message,
        )

    def execute(self, operation: KGOperation) -> KGObservation:
        if not operation.is_schema_valid:
            return self._error("schema-invalid KG operation")
        if operation.entity not in self.visible_entities:
            return self._error("entity was not exposed by the question or KG")

        action = operation.action
        direction = "tail" if "tail" in action else "head"
        adjacency = self.knowledge_graph.outgoing if direction == "tail" else self.knowledge_graph.incoming
        if action.endswith("_relations"):
            relations = sorted(adjacency.get(operation.entity, {}))
            if self.top_k is not None and self.top_k > 0:
                relations = relations[: self.top_k]
            self.relation_cache[(operation.entity, direction)] = set(relations)
            return KGObservation(payload={"relations": relations})

        relation = str(operation.relation)
        exposed_relations = self.relation_cache.get((operation.entity, direction))
        if exposed_relations is None or relation not in exposed_relations:
            return self._error("relation was not returned by the preceding relation operation")
        entities = list(adjacency.get(operation.entity, {}).get(relation, ()))
        self.visible_entities.update(entities)
        return KGObservation(payload={"entities": entities})
