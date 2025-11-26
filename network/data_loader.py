from typing import Any, Callable, Iterable, Tuple
import torch
from torch.utils.data import DataLoader, Subset

class NeuralDataLoader:
    def __init__(self, dataset_cls: Callable[..., Any], config: dict[str, Any] | None = None):
        """Wrap a dataset class and build train/validation DataLoaders.

        Args:
            dataset_cls: A callable that returns a dataset instance when called with ``data_dir``.
            config: Dictionary with optional keys:
                - data_dir (str|Path): location of the dataset (default: 'data')
                - batch_size (int): batch size for loaders (default: 32)
                - val_batch_size (int): batch size for validation (default: same as train)
                - num_workers (int): DataLoader num_workers (default: 0)
                - val_split (float): fraction of data used for validation (default: 0.2)
                - shuffle (bool): whether to shuffle before splitting (default: True)
                - seed (int): RNG seed for reproducible splits (default: 42)
                - train_range (tuple[int, int]): optional [start, end) training slice
                - val_range (tuple[int, int]): optional [start, end) validation slice
        """
        self.dataset_cls = dataset_cls
        self.config = dict(config or {})
        self.train_loader: Any | None = None
        self.val_loader: Any | None = None

    def build(self) -> Tuple[Any, Any]:
        """Instantiate the dataset and create train/validation DataLoaders.

        Returns:
            (train_loader, val_loader)

        Raises:
            TypeError: if ``dataset_cls`` is not callable.
            ValueError: if the dataset is empty or split parameters are invalid.
        """

        if not callable(self.dataset_cls):
            raise TypeError("dataset_cls must be callable (a Dataset class or a factory function)")

        data_dir = self.config.get("data_dir", "data")
        batch_size = int(self.config.get("batch_size", 32))
        val_batch_size = int(self.config.get("val_batch_size", batch_size))
        num_workers = int(self.config.get("num_workers", 0))
        val_split = float(self.config.get("val_split", 0.2))
        shuffle = bool(self.config.get("shuffle", True))
        seed = int(self.config.get("seed", 42))

        train_range = self.config.get("train_range")
        val_range = self.config.get("val_range")

        dataset = self.dataset_cls(data_dir)

        n = len(dataset)
        if n == 0:
            raise ValueError("Cannot build DataLoaders from an empty dataset")

        def _range_to_indices(value):
            if value is None:
                return None
            if not hasattr(value, "__iter__"):
                raise ValueError("range must be an iterable of two integers")
            start, end = value
            start, end = int(start), int(end)
            start = max(0, min(n, start))
            end = max(0, min(n, end))
            if start > end:
                raise ValueError("range start must be <= end")
            return list(range(start, end))

        train_indices = _range_to_indices(train_range)
        val_indices = _range_to_indices(val_range)

        if train_indices is None and val_indices is None:
            val_count = int(n * val_split)
            if n > 1:
                val_count = max(1, min(n - 1, val_count))
            else:
                val_count = 0
            train_count = n - val_count

            if shuffle:
                gen = torch.Generator()
                gen.manual_seed(seed)
                indices = torch.randperm(n, generator=gen).tolist()
                train_indices = indices[:train_count]
                val_indices = indices[train_count:]
            else:
                train_indices = list(range(train_count))
                val_indices = list(range(train_count, n))
        else:
            train_set = set(train_indices) if train_indices is not None else set()
            if val_indices is None and train_indices is not None:
                val_indices = [idx for idx in range(n) if idx not in train_set]
            val_set = set(val_indices) if val_indices is not None else set()
            if train_indices is None and val_indices is not None:
                train_indices = [idx for idx in range(n) if idx not in val_set]
            if train_set & val_set:
                raise ValueError("train_range and val_range overlap")
            if not train_indices:
                raise ValueError("train_range/reserved indices must cover at least one sample")
            # keep order defined by ranges, do not apply dataset-level shuffling

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        return self.train_loader, self.val_loader

    def train_iter(self) -> Iterable[Any]:
        """Return the training DataLoader (iterable)."""
        if self.train_loader is None:
            raise RuntimeError("train_loader is not built yet. Call build() first.")
        return self.train_loader

    def val_iter(self) -> Iterable[Any]:
        """Return the validation DataLoader (iterable)."""
        if self.val_loader is None:
            raise RuntimeError("val_loader is not built yet. Call build() first.")
        return self.val_loader
