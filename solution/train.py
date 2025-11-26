from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Sequence

from dataset import ShapesDataset
from network.neural_network import NeuralNetworkModel
from solution.label_encoder import PairLabelEncoder
from metrics.records import TrainingRecord
from metrics.confusion_matrix import ConfusionMatrix
from visualization.visualization import LossHistoryPlotter, ValidationMetricPlotter, ConfusionMatrixPlotter

LABEL_ENCODER = PairLabelEncoder()


class MultiTaskShapesDataset(Dataset):
    def __init__(self, data_dir: str | Path, transform=None):
        self.base = ShapesDataset(data_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, counts = self.base[idx]
        if image.shape[0] != 1:
            image = image.mean(dim=0, keepdim=True)
        cls_label = LABEL_ENCODER.encode(counts)
        return image, torch.tensor(cls_label, dtype=torch.long), counts


def build_loaders(
    data_dir: Path,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    max_train: int | None,
    max_val: int | None,
    train_shuffle: bool = True,
    train_transform=None,
) -> tuple[DataLoader, DataLoader]:
    train_end = min(9000, max_train) if max_train is not None else 9000
    if train_end <= 0:
        raise ValueError("train sample count must be > 0")
    val_start = 9000
    val_length = min(1000, max_val) if max_val is not None else 1000
    val_end = val_start + val_length

    dataset_train = MultiTaskShapesDataset(data_dir, transform=train_transform)
    dataset_val = MultiTaskShapesDataset(data_dir, transform=None)
    n = len(dataset_train)

    def clamp_range(span):
        start, end = span
        start = max(0, min(n, int(start)))
        end = max(0, min(n, int(end)))
        if start > end:
            raise ValueError("range start must be <= end")
        return list(range(start, end))

    train_indices = clamp_range((0, train_end))
    val_indices = clamp_range((val_start, val_end))

    train_subset = Subset(dataset_train, train_indices)
    val_subset = Subset(dataset_val, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def move_model_to_device(model: NeuralNetworkModel, device: torch.device) -> None:
    model.backbone.to(device)
    model.head_cls.to(device)
    model.head_cnt.to(device)


def train_one_epoch(model: NeuralNetworkModel, loader: DataLoader, device: torch.device):
    stats = {"loss": 0.0, "loss_cls": 0.0, "loss_cnt": 0.0}
    batches = 0
    for inputs, cls_targets, counts_targets in loader:
        inputs = inputs.to(device)
        cls_targets = cls_targets.to(device)
        counts_targets = counts_targets.to(device)
        result = model.train_step((inputs, cls_targets, counts_targets))
        for key in stats:
            stats[key] += result.get(key, 0.0)
        batches += 1
    if batches == 0:
        return {}
    return {key: value / batches for key, value in stats.items()}

def train_one_epoch_regression_only(model: NeuralNetworkModel, loader: DataLoader, device: torch.device):
    """One training epoch in regression-only mode (ignore classification loss)."""
    stats = {"loss": 0.0, "loss_cls": 0.0, "loss_cnt": 0.0}
    batches = 0

    # parametry do ewentualnego clipowania gradientów
    all_params = (
        list(model.backbone.parameters())
        + list(model.head_cls.parameters())
        + list(model.head_cnt.parameters())
    )
    max_grad_norm = getattr(model, "max_grad_norm", 0.0)

    for inputs, _, counts_targets in loader:
        inputs = inputs.to(device)
        counts_targets = counts_targets.to(device)

        model.backbone.train()
        model.head_cls.train()
        model.head_cnt.train()
        model.optimizer.zero_grad()

        # forward: nadal liczymy log_probs, ale ich nie używamy w lossie
        log_probs, counts_pred = model.forward(inputs)

        # regression-only loss
        loss_cnt = model.loss_cnt(counts_pred, counts_targets)
        loss = loss_cnt  # brak składowej klasyfikacyjnej

        loss.backward()

        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)

        model.optimizer.step()

        stats["loss"] += float(loss.item())
        stats["loss_cnt"] += float(loss_cnt.item())
        # loss_cls zostaje 0.0 (ignorujemy klasyfikację)
        batches += 1

    if batches == 0:
        return {}

    return {key: value / batches for key, value in stats.items()}


def compute_metrics(eval_out: dict):
    if not eval_out:
        return {}
    preds_cls = torch.cat([tensor.detach().cpu() for tensor in eval_out["preds_cls"]], dim=0)
    preds_cnt = torch.cat([tensor.detach().cpu() for tensor in eval_out["preds_cnt"]], dim=0)
    targets_cls = torch.cat([tensor.detach().cpu() for tensor in eval_out["targets_cls"]], dim=0)
    targets_cnt = torch.cat([tensor.detach().cpu() for tensor in eval_out["targets_cnt"]], dim=0)

    accuracy = (preds_cls.argmax(dim=1) == targets_cls).float().mean().item()
    diffs = preds_cnt - targets_cnt
    rmse = diffs.pow(2).mean().sqrt().item()
    mae = diffs.abs().mean().item()
    return {"accuracy": accuracy, "rmse": rmse, "mae": mae}


def evaluate_model(model: NeuralNetworkModel, loader: DataLoader, device: torch.device):
    eval_out = model.evaluate(loader, device=device)
    metrics = compute_metrics(eval_out)
    metrics.update({k: eval_out[k] for k in ("loss", "loss_cls", "loss_cnt") if k in eval_out})
    return metrics


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        model: NeuralNetworkModel,
        device: torch.device | str = "cuda",
        batch_size: int = 64,
        val_batch_size: int = 1000,
        num_workers: int = 0,
        max_train: int | None = 9000,
        max_val: int | None = 1000,
        train_shuffle: bool = True,
        train_transform=None,
    ):
        self.device = torch.device(device)
        self.train_loader, self.val_loader = build_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            max_train=max_train,
            max_val=max_val,
            train_shuffle=train_shuffle,
            train_transform=train_transform,
        )
        self.model = model
        move_model_to_device(self.model, self.device)
        self.records = TrainingRecord()

    def train(
            self,
            epochs: int = 1,
            verbose: bool = True,
            target_accuracy: float = 1.0,
            regression_only: bool = False,
    ) -> TrainingRecord:
        """Train the model for a given number of epochs.

        Early stops when validation accuracy reaches `target_accuracy`.

        If `regression_only` is True, only the regression loss is used for updates.
        """
        if not (0.0 <= target_accuracy <= 1.0):
            raise ValueError("target_accuracy must be in [0.0, 1.0]")

        for epoch in range(1, epochs + 1):
            # 1) standard vs regression-only training step
            if regression_only:
                train_stats = train_one_epoch_regression_only(self.model, self.train_loader, self.device)
            else:
                train_stats = train_one_epoch(self.model, self.train_loader, self.device)

            # 2) ewaluacja na walidacji (zostaje wspólna – można nadal oglądać accuracy i RMSE)
            val_stats = evaluate_model(self.model, self.val_loader, self.device)

            # zapis historii
            self.records.add("train", train_stats)
            self.records.add("val", val_stats)

            if verbose:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={train_stats.get('loss'):.4f}"
                    f" cls={train_stats.get('loss_cls', 0.0):.4f}"
                    f" cnt={train_stats.get('loss_cnt', 0.0):.4f}"
                )
                print(
                    f"          val_loss={val_stats.get('loss'):.4f}"
                    f" acc={val_stats.get('accuracy', 0):.4f}"
                    f" rmse={val_stats.get('rmse', 0):.4f}"
                    f" mae={val_stats.get('mae', 0):.4f}"
                )

            # 3) early stopping po accuracy na walidacji
            val_acc = float(val_stats.get("accuracy", 0.0))
            if val_acc >= target_accuracy:
                if verbose:
                    print(
                        f"Target accuracy {target_accuracy:.4f} reached at epoch {epoch}, "
                        f"stopping training."
                    )
                break

        return self.records

    def plot_losses(self, loss_key: str = "loss", title: str | None = None, save_path: Path | None = None) -> None:
        LossHistoryPlotter(self.records.get_history()).plot(
            loss_key=loss_key,
            title=title,
            save_path=save_path,
        )

    def plot_validation_metrics(self, metrics: Sequence[str] = ("accuracy", "rmse", "mae"), title: str | None = None, save_path: Path | None = None) -> None:
        ValidationMetricPlotter(self.records.get_history()).plot(
            metrics=metrics,
            title=title,
            save_path=save_path,
        )

    def compute_confusion_matrix(
        self,
        split: str = "val",
        num_classes: int = 135,
    ) -> ConfusionMatrix:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        loader = self.train_loader if split == "train" else self.val_loader
        cm = ConfusionMatrix(num_classes=num_classes, device=self.device)

        eval_out = self.model.evaluate(loader, device=self.device)
        preds_batches = eval_out["preds_cls"]
        targets_batches = eval_out["targets_cls"]

        for preds, targets in zip(preds_batches, targets_batches):
            cm.update(preds.detach(), targets.detach())

        return cm

    def plot_confusion_matrix(
            self,
            split: str = "val",
            num_classes: int = 135,
            class_indices: Sequence[int] | None = None,
            class_names: Sequence[str] | None = None,
            normalize: bool = False,
            title: str | None = None,
            save_path: Path | None = None,
    ) -> None:
        cm = self.compute_confusion_matrix(split=split, num_classes=num_classes)
        plotter = ConfusionMatrixPlotter(cm)
        if title is None:
            title = f"{split.capitalize()} confusion matrix"
        plotter.plot(
            class_indices=class_indices,
            class_names=class_names,
            normalize=normalize,
            title=title,
            save_path=save_path,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.model.backbone.state_dict(),
                "head_cls": self.model.head_cls.state_dict(),
                "head_cnt": self.model.head_cnt.state_dict(),
            },
            path,
        )

    def evaluate(self) -> dict[str, float]:
        return evaluate_model(self.model, self.val_loader, self.device)

    def report_best_worst_classes(
            self,
            split: str = "val",
            num_classes: int = 135,
            k: int = 5,
            class_names: Sequence[str] | None = None,
    ) -> None:
        cm = self.compute_confusion_matrix(split=split, num_classes=num_classes)
        best, worst = cm.top_k_best_worst(k=k)

        def name_for(idx: int) -> str:
            if class_names is not None and 0 <= idx < len(class_names):
                return class_names[idx]
            try:
                return LABEL_ENCODER.decode_human(idx)
            except Exception:
                return f"class_{idx}"

        print(f"=== {split.upper()} – best-classified ===")
        for idx, acc in best:
            desc = name_for(idx)
            print(f"  category {idx:3d}: acc={acc:.3f} | {desc}")

        print(f"\n=== {split.upper()} – worst-classified ===")
        for idx, acc in worst:
            desc = name_for(idx)
            print(f"  category {idx:3d}: acc={acc:.3f} | {desc}")


__all__ = ["Trainer"]
