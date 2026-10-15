from __future__ import annotations

from typing import Any, Mapping, Sequence


KGQA_SYSTEM_PROMPT = """Answer the question through multi-turn knowledge graph interaction.

Protocol:
1. Put concise reasoning inside <think>...</think>.
2. Issue at most one KG operation inside <search>...</search>.
3. Wait for the structured <observation> before the next operation.
4. When ready, emit the ordered answer list inside <answer>...</answer>.

The only valid KG operations are:
{"action":"get_tail_relations","entity":"<entity_name>"}
{"action":"get_head_relations","entity":"<entity_name>"}
{"action":"get_tail_entities","entity":"<entity_name>","relation":"<relation_id>"}
{"action":"get_head_entities","entity":"<entity_name>","relation":"<relation_id>"}

Use entity names exposed by the question or observations. Copy relation IDs
exactly from observations. Never invent hidden entity identifiers.

Final answer format:
<answer>{"answers":["<name1>","<name2>"]}</answer>
"""


def visible_start_entity_names(start_entities: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for entity in start_entities:
        if isinstance(entity, Mapping):
            value = entity.get("name") or entity.get("entity")
        else:
            value = entity
        name = str(value or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def format_question_prompt(question: str, start_entities: Sequence[Any]) -> str:
    names = visible_start_entity_names(start_entities)
    return f"Question: {question}\nStart entities: {'; '.join(names)}"
