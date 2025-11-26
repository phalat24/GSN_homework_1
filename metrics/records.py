from __future__ import annotations

from typing import Dict, Iterable, Mapping


class TrainingRecord:
    def __init__(self) -> None:
        self._history: Dict[str, list[dict[str, float]]] = {"train": [], "val": []}

    def add(self, split: str, stats: Mapping[str, float]) -> None:
        if split not in self._history:
            raise ValueError("split must be either 'train' or 'val'")
        if not stats:
            return
        record = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}
        if record:
            self._history[split].append(record)

    def get_history(self) -> dict[str, list[dict[str, float]]]:
        return {split: list(records) for split, records in self._history.items()}

