from __future__ import annotations

from typing import Iterable, Tuple

from torch import Tensor


class PairLabelEncoder:

    SHAPE_NAMES = {
        0: "square",
        1: "circle",
        2: "triangle up",
        3: "triangle right",
        4: "triangle down",
        5: "triangle left",
    }

    def __init__(self):
        self.pair_combinations = list(self._generate_pairs())
        self.pair_to_index = {pair: idx for idx, pair in enumerate(self.pair_combinations)}

    @staticmethod
    def _generate_pairs() -> Iterable[Tuple[int, int]]:
        for i in range(6):
            for j in range(i + 1, 6):
                yield (i, j)

    def encode(self, counts: Tensor) -> int:
        positives = (counts > 0.5).nonzero(as_tuple=True)[0]
        if positives.numel() != 2:
            raise ValueError("Expected exactly two non-zero shapes per sample")
        a, b = sorted(int(idx.item()) for idx in positives)
        pair = (a, b)
        if pair not in self.pair_to_index:
            raise ValueError(f"Unexpected shape pair: {pair}")
        base = self.pair_to_index[pair]
        primary_count = int(round(counts[a].item()))
        if not 1 <= primary_count <= 9:
            raise ValueError("Primary count must lie between 1 and 9")
        return base * 9 + (primary_count - 1)

    def decode(self, label: int) -> Tuple[Tuple[int, int], int]:
        if not 0 <= label < 135:
            raise ValueError("label must be in range [0, 135)")
        pair_index = label // 9
        count = (label % 9) + 1
        return self.pair_combinations[pair_index], count

    def decode_counts(self, label: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        (a, b), primary_count = self.decode(label)
        count_a = primary_count
        count_b = 10 - primary_count
        return (a, b), (count_a, count_b)

    def decode_human(self, label: int) -> str:
        (a, b), (count_a, count_b) = self.decode_counts(label)
        name_a = self.SHAPE_NAMES.get(a, f"shape_{a}")
        name_b = self.SHAPE_NAMES.get(b, f"shape_{b}")

        return f"{count_a} {name_a} + {count_b} {name_b}"
