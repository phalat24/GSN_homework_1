from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


class ConfusionMatrix:

    def __init__(self, num_classes: int, device: torch.device | str = "cpu"):
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=self.device,
        )

    def reset(self) -> None:
        self.matrix.zero_()

    @torch.no_grad()
    def update(self, preds: Tensor, targets: Tensor) -> None:

        if preds.ndim == 2:
            preds = preds.argmax(dim=1)

        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have same shape")

        preds = preds.to(self.device)
        targets = targets.to(self.device)

        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        if preds.numel() == 0:
            return

        indices = targets * self.num_classes + preds
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2)
        cm_batch = bincount.view(self.num_classes, self.num_classes)

        self.matrix += cm_batch

    def to_tensor(self) -> Tensor:
        return self.matrix.clone()

    def to_numpy(self) -> np.ndarray:
        return self.matrix.detach().cpu().numpy()

    def overall_accuracy(self) -> float:
        correct = torch.trace(self.matrix).item()
        total = self.matrix.sum().item()
        if total == 0:
            return 0.0
        return float(correct / total)

    def per_class_accuracy(self) -> torch.Tensor:
        tp = torch.diag(self.matrix)
        total = self.matrix.sum(dim=1)
        acc = torch.where(
            total > 0,
            tp.float() / total.float(),
            torch.zeros_like(total, dtype=torch.float),
        )
        return acc  # tensor [C]

    def top_k_best_worst(self, k: int = 5):
        acc = self.per_class_accuracy()
        C = acc.numel()
        k = min(k, C)

        # najlepsze
        best_vals, best_idx = torch.topk(acc, k=k, largest=True)
        # najgorsze
        worst_vals, worst_idx = torch.topk(acc, k=k, largest=False)

        best = list(zip(best_idx.tolist(), best_vals.tolist()))
        worst = list(zip(worst_idx.tolist(), worst_vals.tolist()))
        return best, worst
