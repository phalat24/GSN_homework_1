"""Dataset implementations moved into the `dataset` package.

This module contains the `ShapesDataset` class which was previously in
`dataset.py` at project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch import Tensor

from .augmentation import apply_augmentations


class ShapesDataset:
    _FILENAME_ALIASES = {"filename", "name"}
    _LABEL_ALIASES = {
        "shape_0": ["shape_0", "squares"],
        "shape_1": ["shape_1", "circles"],
        "shape_2": ["shape_2", "up"],
        "shape_3": ["shape_3", "right"],
        "shape_4": ["shape_4", "down"],
        "shape_5": ["shape_5", "left"],
    }

    def __init__(
        self,
        data_dir: str | Path = "../data/",
        transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"data_dir does not exist: {self.data_dir}")

        self.labels_path = self.data_dir / "labels.csv"
        if not self.labels_path.exists():
            raise ValueError(f"labels.csv not found in {self.data_dir}")

        self._df = pd.read_csv(self.labels_path)

        filename_cols = set(self._df.columns) & self._FILENAME_ALIASES
        if not filename_cols:
            raise ValueError(
                f"Expected one of {sorted(self._FILENAME_ALIASES)} in labels.csv, got columns: {list(self._df.columns)}"
            )
        self._filename_column = sorted(filename_cols)[0]

        self._label_columns = {}
        for canonical, candidates in self._LABEL_ALIASES.items():
            found = next((col for col in candidates if col in self._df.columns), None)
            if not found:
                raise ValueError(
                    f"Expected one of {candidates} for '{canonical}' in labels.csv, got columns: {list(self._df.columns)}"
                )
            self._label_columns[canonical] = found

        # Optional augmentation / transform that takes (image, counts)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if idx < 0 or idx >= len(self._df):
            raise IndexError("index out of range")

        row = self._df.iloc[idx]
        filename = row[self._filename_column]
        img_path = self.data_dir / str(filename)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = self._pil_to_tensor(image)

        labels = [row[self._label_columns[key]] for key in sorted(self._label_columns)]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Apply optional augmentation that can modify both image and labels
        image_tensor, labels_tensor = apply_augmentations(
            image_tensor, labels_tensor, self.transform
        )

        return image_tensor, labels_tensor

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> Tensor:
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        h, w = image.size[1], image.size[0]
        c = 3
        arr = arr.view(h, w, c).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        return arr

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ShapesDataset(data_dir={self.data_dir!r}, n_samples={len(self)})"
