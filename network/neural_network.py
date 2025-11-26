import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LabelSmoothingNLLLoss(nn.Module):
    """Manual label smoothing that operates on log-probabilities."""

    def __init__(self, smoothing: float, num_classes: int):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("label smoothing must lie in [0, 1)")
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nll = F.nll_loss(log_probs, targets, reduction="none")
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()


class AssignmentBackbone(nn.Module):
    """Backbone defined in the assignment, exposing the shared feature extractor."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = 256
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, self.out_features),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class NeuralNetworkModel:
    def __init__(self, config, head_cls=None, head_cnt=None, backbone_cls=None):
        self.config = config or {}
        self.backbone = None
        self.head_cls = head_cls
        self.head_cnt = head_cnt
        self.optimizer = None
        self.loss_cls = None
        self.loss_cnt = None
        self.backbone_cls = backbone_cls or AssignmentBackbone
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.0))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))
        self.label_smoothing = float(self.config.get("label_smoothing", 0.0))
        self._build()

    def _build(self):
        """Build the exact backbone from the assignment plus two heads and losses."""
        self.num_classes = int(self.config.get("num_classes", 135))
        self.num_counts = int(self.config.get("num_counts", 6))
        self.dropout = float(self.config.get("dropout", 0.5))
        lr = float(self.config.get("learning_rate", 1e-3))
        self.lambda_cnt = float(self.config.get("lambda_cnt", 1.0))

        backbone_kwargs = self.config.get("backbone_kwargs", {})
        self.backbone = self.backbone_cls(**backbone_kwargs)
        feature_dim = getattr(self.backbone, "out_features", 256)

        # If heads were not provided explicitly, fall back to simple linear heads
        if self.head_cls is None:
            self.head_cls = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(feature_dim, self.num_classes))
        if self.head_cnt is None:
            self.head_cnt = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(feature_dim, self.num_counts))

        params = (
            list(self.backbone.parameters())
            + list(self.head_cls.parameters())
            + list(self.head_cnt.parameters())
        )
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        if self.label_smoothing > 0:
            self.loss_cls = LabelSmoothingNLLLoss(self.label_smoothing, self.num_classes)
        else:
            self.loss_cls = nn.NLLLoss()
        self.loss_cnt = nn.SmoothL1Loss()

    def forward(self, inputs):
        """Return log-probs and regression outputs for a batch."""
        features = self.backbone(inputs)
        logits = self.head_cls(features)
        log_probs = F.log_softmax(logits, dim=1)
        counts = self.head_cnt(features)
        return log_probs, counts

    def _joint_loss(self, log_probs, cls_targets, counts_pred, counts_targets):
        """Compute the multitask loss used during training."""
        loss_cls = self.loss_cls(log_probs, cls_targets)
        loss_cnt = self.loss_cnt(counts_pred, counts_targets)
        return loss_cls + self.lambda_cnt * loss_cnt, loss_cls, loss_cnt

    def train_step(self, batch):
        """Perform one optimizer step on (inputs, cls_targets, counts_targets)."""
        inputs, cls_targets, counts_targets = batch
        self.backbone.train()
        self.head_cls.train()
        self.head_cnt.train()
        self.optimizer.zero_grad()
        log_probs, counts = self.forward(inputs)
        joint_loss, loss_cls, loss_cnt = self._joint_loss(
            log_probs, cls_targets, counts, counts_targets
        )
        joint_loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters())
                + list(self.head_cls.parameters())
                + list(self.head_cnt.parameters()),
                max_norm=self.max_grad_norm,
            )
        self.optimizer.step()
        return {
            "loss": float(joint_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_cnt": float(loss_cnt.item()),
        }

    def evaluate(self, dataset, device: torch.device | str | None = None):
        """Evaluate on an iterable of (inputs, cls_targets, counts_targets)."""
        device = torch.device(device) if device is not None else None
        self.backbone.eval()
        self.head_cls.eval()
        self.head_cnt.eval()
        total_loss = total_cls = total_cnt = 0.0
        batches = 0
        preds_cls = []
        preds_cnt = []
        targets_cls = []
        targets_cnt = []
        with torch.no_grad():
            for inputs, cls_targets, counts_targets in dataset:
                if device is not None:
                    inputs = inputs.to(device)
                    cls_targets = cls_targets.to(device)
                    counts_targets = counts_targets.to(device)
                log_probs, counts = self.forward(inputs)
                joint_loss, loss_cls, loss_cnt = self._joint_loss(
                    log_probs, cls_targets, counts, counts_targets
                )
                total_loss += float(joint_loss.item())
                total_cls += float(loss_cls.item())
                total_cnt += float(loss_cnt.item())
                batches += 1
                preds_cls.append(log_probs)
                preds_cnt.append(counts)
                targets_cls.append(cls_targets)
                targets_cnt.append(counts_targets)
        if batches == 0:
            return {}
        return {
            "loss": total_loss / batches,
            "loss_cls": total_cls / batches,
            "loss_cnt": total_cnt / batches,
            "preds_cls": preds_cls,
            "preds_cnt": preds_cnt,
            "targets_cls": targets_cls,
            "targets_cnt": targets_cnt,
        }


def build_model(config=None, head_cls=None, head_cnt=None, backbone_cls=None):
    """Factory helper to assemble models with custom heads/backbone.

    The heads should already be instantiated ``nn.Sequential`` (or any ``nn.Module``)
    objects, so they can be created in the notebook in a descriptive way, e.g.::

        head_cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 135),
        )

        head_cnt = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

        model = build_model(config, head_cls=head_cls, head_cnt=head_cnt)
    """
    config = config or {}
    return NeuralNetworkModel(
        config=config,
        head_cls=head_cls,
        head_cnt=head_cnt,
        backbone_cls=backbone_cls,
    )


__all__ = ["AssignmentBackbone", "NeuralNetworkModel", "build_model"]
