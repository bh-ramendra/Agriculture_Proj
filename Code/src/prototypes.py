"""Logit-prototype memory bank (DG-PLDR LPMB) and feature-prototype helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses import batch_feature_prototypes, inter_class_separation


class LogitPrototypeMemory(nn.Module):
    """External memory of per-class logit prototypes.

    Update rule from DG-PLDR, with momentum m (paper default 0.99):

        p_i <- m * p_i + (1 - m) * mean(logits of class i)

    The bank is not a learned parameter. Sample logits are pulled toward the
    detached prototypes by the contrastive loss.
    """

    def __init__(self, num_classes: int, momentum: float = 0.99):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, num_classes))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.detach()
        for class_id in labels.unique().tolist():
            class_logits = logits[labels == class_id].mean(dim=0)
            if not bool(self.initialized[class_id]):
                self.prototypes[class_id] = class_logits
                self.initialized[class_id] = True
            else:
                self.prototypes[class_id] = (
                    self.momentum * self.prototypes[class_id]
                    + (1.0 - self.momentum) * class_logits
                )

    def active_prototypes(self) -> torch.Tensor:
        if not bool(self.initialized.any()):
            return self.prototypes
        # Unseen classes stay at zero until the first sample arrives. The
        # contrastive denominator still includes every class, matching the paper.
        return self.prototypes


def feature_prototype_snapshot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    prototypes, mask = batch_feature_prototypes(
        torch.nn.functional.normalize(embeddings, dim=-1),
        labels,
        num_classes,
    )
    return {
        "prototypes": prototypes.detach().cpu(),
        "mask": mask.detach().cpu(),
        "separation": inter_class_separation(prototypes, mask).detach().cpu(),
    }
