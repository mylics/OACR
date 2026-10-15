from __future__ import annotations

import threading
from typing import Any, Mapping

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.kg.local import LocalKnowledgeGraph, LocalKnowledgeGraphTool
from src.schema import KGOperation


class KGOperationRequest(BaseModel):
    session_id: str
    question: str = ""
    topic_entities: list[Any] = Field(default_factory=list)
    entity: str
    relation: str | None = None


class LocalKnowledgeGraphService:
    def __init__(
        self,
        knowledge_graph: LocalKnowledgeGraph,
        top_k: int | None = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.top_k = top_k
        self.sessions: dict[str, LocalKnowledgeGraphTool] = {}
        self.lock = threading.Lock()

    def _session(self, request: KGOperationRequest) -> LocalKnowledgeGraphTool:
        with self.lock:
            if request.session_id not in self.sessions:
                self.sessions[request.session_id] = LocalKnowledgeGraphTool(
                    self.knowledge_graph,
                    request.topic_entities,
                    top_k=self.top_k,
                )
            return self.sessions[request.session_id]

    def execute(
        self,
        action: str,
        request: KGOperationRequest,
    ) -> dict[str, Any]:
        operation = KGOperation(
            action=action,
            entity=request.entity,
            relation=request.relation,
        )
        observation = self._session(request).execute(operation)
        return {
            "ok": not observation.is_error,
            "observation": observation.to_dict(),
            "error": observation.error,
        }

    def reset(self, session_id: str) -> bool:
        with self.lock:
            return self.sessions.pop(session_id, None) is not None


def create_app(service: Any) -> FastAPI:
    app = FastAPI(title="OACR KG Tool Interface", version="0.1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        if hasattr(service, "health"):
            return dict(service.health())
        return {"ok": True, "sessions": len(service.sessions)}

    @app.post("/get_tail_relations")
    def get_tail_relations(request: KGOperationRequest) -> dict[str, Any]:
        return service.execute("get_tail_relations", request)

    @app.post("/get_head_relations")
    def get_head_relations(request: KGOperationRequest) -> dict[str, Any]:
        return service.execute("get_head_relations", request)

    @app.post("/get_tail_entities")
    def get_tail_entities(request: KGOperationRequest) -> dict[str, Any]:
        return service.execute("get_tail_entities", request)

    @app.post("/get_head_entities")
    def get_head_entities(request: KGOperationRequest) -> dict[str, Any]:
        return service.execute("get_head_entities", request)

    @app.post("/reset_session")
    def reset_session(payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return {"ok": False, "error": "missing session_id"}
        return {"ok": True, "removed": service.reset(session_id)}

    return app
