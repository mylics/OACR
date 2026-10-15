from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

import requests

from src.schema import KGObservation, KGOperation


class HTTPKnowledgeGraphTool:
    def __init__(
        self,
        service_url: str,
        question: str,
        start_entities: Sequence[Any],
        request_timeout: int = 60,
        session_id: str | None = None,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.question = question
        self.start_entities = list(start_entities)
        self.request_timeout = int(request_timeout)
        self.session_id = session_id or uuid.uuid4().hex
        self._session = requests.Session()
        self._session.trust_env = False

    def _topic_entities(self) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for entity in self.start_entities:
            if isinstance(entity, Mapping):
                entities.append(dict(entity))
            else:
                entities.append({"name": str(entity)})
        return entities

    def execute(self, operation: KGOperation) -> KGObservation:
        if not operation.is_schema_valid:
            return KGObservation(
                payload={"error": "schema-invalid KG operation"},
                execution_succeeded=False,
                error="schema-invalid KG operation",
            )

        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "question": self.question,
            "topic_entities": self._topic_entities(),
            "entity": operation.entity,
        }
        if operation.relation:
            payload["relation"] = operation.relation

        try:
            response = self._session.post(
                f"{self.service_url}/{operation.action}",
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, Mapping):
                raise ValueError("KG service returned a non-object response")
            observation = result.get("observation", result)
            if not isinstance(observation, Mapping):
                observation = {"result": observation}
            error = result.get("error") or observation.get("error")
            return KGObservation(
                payload=dict(observation),
                execution_succeeded=bool(result.get("ok", not error)) and not bool(error),
                error=str(error) if error else None,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return KGObservation(
                payload={"error": message},
                execution_succeeded=False,
                error=message,
            )

    def reset(self) -> None:
        try:
            self._session.post(
                f"{self.service_url}/reset_session",
                json={"session_id": self.session_id},
                timeout=self.request_timeout,
            )
        finally:
            self.session_id = uuid.uuid4().hex
