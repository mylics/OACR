from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import requests

from src.schema import KGObservation, KGOperation

FREEBASE_NAMESPACE = "http://rdf.freebase.com/ns/"

_FREEBASE_MID_PATTERN = re.compile(r"^[mg]\.[A-Za-z0-9_.\-/]+$")
_FREEBASE_RELATION_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")
_FILTERED_RELATION_PREFIXES = (
    "base.",
    "common.",
    "dataworld.",
    "freebase.",
    "kg.",
    "media_common.",
    "owl.",
    "rdf.",
    "rdfs.",
    "skos.",
    "type.",
    "user.",
)
_FILTERED_RELATION_MARKERS = (
    ".alias",
    ".article",
    ".attribution",
    ".catalog_id",
    ".creator",
    ".database_id",
    ".description",
    ".external_id",
    ".external_links",
    ".guid",
    ".homepage",
    ".identifier",
    ".image",
    ".key",
    ".keys",
    ".license",
    ".official_website",
    ".permission",
    ".provider",
    ".registry_id",
    ".rights",
    ".social_media",
    ".source_uri",
    ".thumbnail",
    ".timestamp",
    ".uri",
    ".url",
    ".webpage",
    ".website",
)


def normalize_freebase_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith(FREEBASE_NAMESPACE):
        return text[len(FREEBASE_NAMESPACE) :]
    if text.startswith("ns:"):
        return text[3:]
    return text


def is_freebase_mid(value: Any) -> bool:
    return bool(_FREEBASE_MID_PATTERN.fullmatch(normalize_freebase_id(value)))


def normalize_entity_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _entity_name_key(value: Any) -> str:
    return normalize_entity_name(value).casefold()


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_entity_name(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _is_allowed_relation(value: Any) -> bool:
    relation = normalize_freebase_id(value)
    if not _FREEBASE_RELATION_PATTERN.fullmatch(relation):
        return False
    if relation.startswith(_FILTERED_RELATION_PREFIXES):
        return False
    return not any(marker in relation for marker in _FILTERED_RELATION_MARKERS)


class FreebaseSPARQLClient:
    def __init__(
        self,
        sparql_url: str,
        *,
        default_graph: str | None = None,
        request_timeout: int = 30,
        relation_limit: int = 1000,
        username_env: str | None = None,
        password_env: str | None = None,
        bearer_token_env: str | None = None,
    ) -> None:
        self.sparql_url = str(sparql_url).strip()
        if not self.sparql_url:
            raise ValueError("sparql_url must not be empty")
        self.default_graph = str(default_graph or "").strip() or None
        self.request_timeout = int(request_timeout)
        self.relation_limit = int(relation_limit)
        self.username_env = str(username_env or "").strip() or None
        self.password_env = str(password_env or "").strip() or None
        self.bearer_token_env = str(bearer_token_env or "").strip() or None
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            return session
        session = requests.Session()
        session.trust_env = False
        if self.username_env:
            username = os.environ.get(self.username_env)
            password = os.environ.get(self.password_env) if self.password_env else None
            if username:
                session.auth = (username, password or "")
        if self.bearer_token_env:
            token = os.environ.get(self.bearer_token_env)
            if token:
                session.headers["Authorization"] = f"Bearer {token}"
        self._thread_local.session = session
        return session

    def query(self, sparql: str) -> list[dict[str, Any]]:
        payload: dict[str, str] = {
            "query": sparql,
            "format": "application/sparql-results+json",
        }
        if self.default_graph:
            payload["default-graph-uri"] = self.default_graph
        response = self._session().post(
            self.sparql_url,
            data=payload,
            headers={"Accept": "application/sparql-results+json"},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        rows = result.get("results", {}).get("bindings", [])
        if not isinstance(rows, list):
            raise ValueError("SPARQL endpoint returned invalid result bindings")
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def ping(self) -> bool:
        try:
            self.query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
            return True
        except Exception:
            return False

    @lru_cache(maxsize=200_000)
    def lookup_mids_by_name(self, name: str) -> tuple[str, ...]:
        name_literal = json.dumps(normalize_entity_name(name), ensure_ascii=False)
        query = f"""
PREFIX ns: <{FREEBASE_NAMESPACE}>
SELECT DISTINCT ?entity WHERE {{
  ?entity ns:type.object.name ?name .
  FILTER(STRSTARTS(STR(?entity), "{FREEBASE_NAMESPACE}"))
  FILTER(LCASE(STR(?name)) = LCASE({name_literal}))
  FILTER(LANG(?name) = "en" || LANG(?name) = "")
}}
LIMIT 20
"""
        mids: list[str] = []
        for row in self.query(query):
            value = row.get("entity", {}).get("value")
            mid = normalize_freebase_id(value)
            if is_freebase_mid(mid) and mid not in mids:
                mids.append(mid)
        return tuple(mids)

    @lru_cache(maxsize=200_000)
    def get_name(self, mid: str) -> str | None:
        normalized_mid = normalize_freebase_id(mid)
        if not is_freebase_mid(normalized_mid):
            return None
        query = f"""
PREFIX ns: <{FREEBASE_NAMESPACE}>
SELECT ?name WHERE {{
  ns:{normalized_mid} ns:type.object.name ?name .
  FILTER(LANG(?name) = "en" || LANG(?name) = "")
}}
LIMIT 1
"""
        rows = self.query(query)
        if not rows:
            return None
        return normalize_entity_name(rows[0].get("name", {}).get("value"))

    @lru_cache(maxsize=200_000)
    def is_cvt(self, mid: str) -> bool:
        return is_freebase_mid(mid) and self.get_name(mid) is None

    @lru_cache(maxsize=200_000)
    def get_relations(self, mid: str, direction: str) -> tuple[str, ...]:

        normalized_mid = normalize_freebase_id(mid)
        if not is_freebase_mid(normalized_mid):
            return ()
        if direction == "tail":
            pattern = f"ns:{normalized_mid} ?relation ?value"
        elif direction == "head":
            pattern = f"?value ?relation ns:{normalized_mid}"
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        query = f"""
PREFIX ns: <{FREEBASE_NAMESPACE}>
SELECT DISTINCT ?relation WHERE {{
  {pattern} .
  FILTER(STRSTARTS(STR(?relation), "{FREEBASE_NAMESPACE}"))
}}
LIMIT {self.relation_limit}
"""
        relations = [normalize_freebase_id(row.get("relation", {}).get("value")) for row in self.query(query)]
        return tuple(sorted({relation for relation in relations if _is_allowed_relation(relation)}))

    @lru_cache(maxsize=500_000)
    def get_values(
        self,
        mid: str,
        relation: str,
        direction: str,
        limit: int,
    ) -> tuple[tuple[str, bool], ...]:

        normalized_mid = normalize_freebase_id(mid)
        normalized_relation = normalize_freebase_id(relation)
        if not is_freebase_mid(normalized_mid):
            return ()
        if not _FREEBASE_RELATION_PATTERN.fullmatch(normalized_relation):
            return ()
        if direction == "tail":
            pattern = f"ns:{normalized_mid} ns:{normalized_relation} ?value"
        elif direction == "head":
            pattern = f"?value ns:{normalized_relation} ns:{normalized_mid}"
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        query = f"""
PREFIX ns: <{FREEBASE_NAMESPACE}>
SELECT ?value WHERE {{
  {pattern} .
}}
LIMIT {max(1, int(limit))}
"""
        output: list[tuple[str, bool]] = []
        for row in self.query(query):
            cell = row.get("value", {})
            value = str(cell.get("value") or "")
            is_entity = cell.get("type") == "uri" and value.startswith(FREEBASE_NAMESPACE)
            output.append(
                (
                    normalize_freebase_id(value) if is_entity else value,
                    is_entity,
                )
            )
        return tuple(output)

    @lru_cache(maxsize=500_000)
    def get_cvt_out_values(
        self,
        cvt_mid: str,
        limit: int,
    ) -> tuple[tuple[str, bool], ...]:
        normalized_mid = normalize_freebase_id(cvt_mid)
        if not is_freebase_mid(normalized_mid):
            return ()
        query = f"""
PREFIX ns: <{FREEBASE_NAMESPACE}>
SELECT ?relation ?value WHERE {{
  ns:{normalized_mid} ?relation ?value .
  FILTER(STRSTARTS(STR(?relation), "{FREEBASE_NAMESPACE}"))
}}
LIMIT {max(1, int(limit))}
"""
        output: list[tuple[str, bool]] = []
        for row in self.query(query):
            relation = normalize_freebase_id(row.get("relation", {}).get("value"))
            if not _is_allowed_relation(relation):
                continue
            cell = row.get("value", {})
            value = str(cell.get("value") or "")
            is_entity = cell.get("type") == "uri" and value.startswith(FREEBASE_NAMESPACE)
            output.append(
                (
                    normalize_freebase_id(value) if is_entity else value,
                    is_entity,
                )
            )
        return tuple(output)


@dataclass
class _FreebaseSession:
    name_to_mids: dict[str, list[str]] = field(default_factory=dict)
    relation_cache: dict[tuple[str, str], set[str]] = field(default_factory=dict)


class FreebaseKnowledgeGraphService:
    def __init__(
        self,
        client: FreebaseSPARQLClient,
        *,
        top_k: int = 0,
        per_relation_limit: int = 0,
        entity_fetch_limit: int = 300,
        cvt_expand_limit: int = 300,
    ) -> None:
        self.client = client
        self.top_k = int(top_k)
        self.per_relation_limit = int(per_relation_limit)
        self.entity_fetch_limit = int(entity_fetch_limit)
        self.cvt_expand_limit = int(cvt_expand_limit)
        self.sessions: dict[str, _FreebaseSession] = {}
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.client.ping(),
            "backend": "freebase-sparql",
            "sessions": len(self.sessions),
        }

    def _session(
        self,
        session_id: str,
        topic_entities: Sequence[Any],
    ) -> _FreebaseSession:
        with self.lock:
            state = self.sessions.setdefault(session_id, _FreebaseSession())
            for entity in topic_entities:
                if not isinstance(entity, Mapping):
                    continue
                name = normalize_entity_name(entity.get("name"))
                mid = normalize_freebase_id(entity.get("mid") or entity.get("id"))
                if name and is_freebase_mid(mid):
                    mids = state.name_to_mids.setdefault(_entity_name_key(name), [])
                    if mid not in mids:
                        mids.append(mid)
            return state

    def _resolve_mids(
        self,
        state: _FreebaseSession,
        entity_name: str,
    ) -> list[str]:
        key = _entity_name_key(entity_name)
        mids = state.name_to_mids.get(key)
        if mids:
            return list(mids)
        discovered = list(self.client.lookup_mids_by_name(entity_name))
        if not discovered:
            raise ValueError(f"Unknown entity name: {entity_name}")
        state.name_to_mids[key] = discovered
        return discovered

    def _remember_entity(
        self,
        state: _FreebaseSession,
        name: str,
        mid: str,
    ) -> None:
        key = _entity_name_key(name)
        values = state.name_to_mids.setdefault(key, [])
        if mid not in values:
            values.append(mid)

    def _visible_value(
        self,
        state: _FreebaseSession,
        value: str,
        is_entity: bool,
    ) -> str | None:
        if not is_entity:
            text = normalize_entity_name(value)
            return text or None
        mid = normalize_freebase_id(value)
        name = self.client.get_name(mid)
        if not name:
            return None
        self._remember_entity(state, name, mid)
        return name

    def _relation_observation(
        self,
        state: _FreebaseSession,
        operation: KGOperation,
        direction: str,
    ) -> KGObservation:
        mids = self._resolve_mids(state, operation.entity)
        relations: set[str] = set()
        for mid in mids:
            relations.update(self.client.get_relations(mid, direction))
        selected = sorted(relations)
        if self.top_k > 0:
            selected = selected[: self.top_k]
        state.relation_cache[(_entity_name_key(operation.entity), direction)] = set(selected)
        return KGObservation(payload={"relations": selected})

    def _entity_observation(
        self,
        state: _FreebaseSession,
        operation: KGOperation,
        direction: str,
    ) -> KGObservation:
        relation = normalize_freebase_id(operation.relation)
        allowed = state.relation_cache.get((_entity_name_key(operation.entity), direction))
        if allowed is None or relation not in allowed:
            raise ValueError("relation was not returned by the preceding relation operation")
        values: list[str] = []
        for mid in self._resolve_mids(state, operation.entity):
            rows = self.client.get_values(
                mid,
                relation,
                direction,
                self.entity_fetch_limit,
            )
            for value, is_entity in rows:
                if is_entity and self.client.is_cvt(value):
                    for cvt_value, cvt_is_entity in self.client.get_cvt_out_values(
                        value,
                        self.cvt_expand_limit,
                    ):
                        if cvt_is_entity and normalize_freebase_id(cvt_value) == mid:
                            continue
                        visible = self._visible_value(state, cvt_value, cvt_is_entity)
                        if visible:
                            values.append(visible)
                    continue
                visible = self._visible_value(state, value, is_entity)
                if visible:
                    values.append(visible)
        entities = _deduplicate(values)
        if self.per_relation_limit > 0:
            entities = entities[: self.per_relation_limit]
        return KGObservation(payload={"entities": entities})

    def execute(
        self,
        action: str,
        request: Any,
    ) -> dict[str, Any]:
        operation = KGOperation(
            action=action,
            entity=request.entity,
            relation=request.relation,
        )
        if not operation.is_schema_valid:
            message = "schema-invalid KG operation"
            return {
                "ok": False,
                "observation": {"error": message},
                "error": message,
            }
        state = self._session(request.session_id, request.topic_entities)
        try:
            direction = "tail" if "tail" in action else "head"
            if action.endswith("_relations"):
                observation = self._relation_observation(state, operation, direction)
            else:
                observation = self._entity_observation(state, operation, direction)
            return {
                "ok": True,
                "observation": observation.to_dict(),
                "error": None,
            }
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return {
                "ok": False,
                "observation": {"error": message},
                "error": message,
            }

    def reset(self, session_id: str) -> bool:
        with self.lock:
            return self.sessions.pop(session_id, None) is not None
