from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt


class LossHistoryPlotter:
    def __init__(self, history: Mapping[str, Iterable[Mapping[str, float]]]):
        self.history = history

    def plot(self, loss_key: str = "loss", title: str | None = None, save_path: Path | None = None) -> None:
        plt.figure()
        for split, records in self.history.items():
            values = [record[loss_key] for record in records if loss_key in record]
            plt.plot(values, label=split)
        plt.xlabel("Epoch")
        plt.ylabel(loss_key)
        plt.title(title or "Loss history")
        plt.legend()
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.show()
        plt.close()


class ValidationMetricPlotter:
    """Plot metrics (accuracy/rmse/mae) recorded on validation split."""

    def __init__(self, history: Mapping[str, Iterable[Mapping[str, float]]]):
        self.history = history

    def plot(
        self,
        metrics: Sequence[str] = ("accuracy", "rmse", "mae"),
        title: str | None = None,
        save_path: Path | None = None,
    ) -> None:
        records = list(self.history.get("val", []))
        plt.figure()
        for metric in metrics:
            values = [record[metric] for record in records if metric in record]
            if not values:
                continue
            plt.plot(values, label=metric)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title or "Validation metrics")
        plt.legend()
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.show()
        plt.close()

class ConfusionMatrixPlotter:

    def __init__(self, cm):
        if hasattr(cm, "to_numpy"):
            self.cm = cm.to_numpy()
        else:
            self.cm = np.asarray(cm)

    def plot(
        self,
        class_indices: Sequence[int] | None = None,
        class_names: Sequence[str] | None = None,
        normalize: bool = False,
        title: str | None = None,
        save_path: Path | None = None,
        figsize: tuple[float, float] = (6.0, 5.0),
    ) -> None:
        cm = self.cm

        if class_indices is not None:
            class_indices = list(class_indices)
            cm = cm[np.ix_(class_indices, class_indices)]
        else:
            class_indices = list(range(cm.shape[0]))

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm = cm / row_sums

        if class_names is None:
            class_names = [str(i) for i in class_indices]
        else:
            if len(class_names) != len(class_indices):
                raise ValueError("len(class_names) must match len(class_indices)")

        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation="nearest", aspect="auto")
        plt.title(title or "Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_indices))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.show()
        plt.close()