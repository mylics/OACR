from collections import deque
from copy import deepcopy
from typing import Iterable, Mapping

from src.schema import Rollout


class RolloutBuffer:
    def __init__(self, capacity: int, rollouts: Iterable = ()):
        if capacity < 1:
            raise ValueError("Rollout buffer capacity must be positive")
        self.capacity = int(capacity)
        self._rows = deque(maxlen=self.capacity)
        self.extend(rollouts)

    def extend(self, rollouts: Iterable):
        for rollout in rollouts:
            row = rollout.to_dict() if isinstance(rollout, Rollout) else dict(rollout)
            if not isinstance(row, Mapping):
                raise ValueError("Buffered rollouts must be records")
            self._rows.append(deepcopy(row))

    def records(self):
        return deepcopy(list(self._rows))

    def __len__(self):
        return len(self._rows)
