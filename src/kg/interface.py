from __future__ import annotations

from typing import Protocol

from src.schema import KGObservation, KGOperation


class KnowledgeGraphToolInterface(Protocol):
    def execute(self, operation: KGOperation) -> KGObservation: ...

    def reset(self) -> None: ...
