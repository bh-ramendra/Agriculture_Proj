"""Classification, DG-PLDR, counterfactual, and prototype losses.

The total training objective used by the full model is

    L = L_cls + lambda_lpcl * L_lpcl
        + lambda_cf * (L_cf_feat + L_cf_pred)
        + lambda_proto * L_proto_cf

L_lpcl is the DG-PLDR logit-prototype contrastive term. The counterfactual
terms follow the implementation guide: feature consistency is an L2 distance
between disease embeddings, and prediction consistency is KL(p(y|x) || p(y|x_cf)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def feature_consistency_loss(z: torch.Tensor, z_cf: torch.Tensor) -> torch.Tensor:
    """L_cf_feat = ||normalize(z) - normalize(z_cf)||^2.

    Embeddings are L2-normalized so the term stays on a comparable scale to
    cross-entropy. z_cf may be [B, D] or [B, K, D].
    """
    z = F.normalize(z, dim=-1)
    if z_cf.dim() == 2:
        z_cf = F.normalize(z_cf, dim=-1)
        return F.mse_loss(z, z_cf)
    losses = []
    for view in range(z_cf.shape[1]):
        view_z = F.normalize(z_cf[:, view], dim=-1)
        losses.append(F.mse_loss(z, view_z))
    return torch.stack(losses).mean()


def prediction_consistency_loss(logits: torch.Tensor, logits_cf: torch.Tensor) -> torch.Tensor:
    """KL(p(y|x) || p(y|x_cf)), averaged over counterfactual views."""
    log_p = F.log_softmax(logits, dim=-1)
    if logits_cf.dim() == 2:
        log_q = F.log_softmax(logits_cf, dim=-1)
        return F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    losses = []
    for view in range(logits_cf.shape[1]):
        log_q = F.log_softmax(logits_cf[:, view], dim=-1)
        losses.append(F.kl_div(log_q, log_p, log_target=True, reduction="batchmean"))
    return torch.stack(losses).mean()


def logit_prototype_contrastive_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """DG-PLDR LPMS loss.

    For a sample logit vector c and class prototypes p_j,

        L = -log softmax(p_j · c / tau)[y]

    Prototypes are detached. The memory bank is updated outside this function.
    """
    if prototypes.numel() == 0:
        return logits.sum() * 0.0
    sims = torch.matmul(logits, prototypes.detach().t()) / temperature
    return F.cross_entropy(sims, labels)


def batch_feature_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean embedding per class. Returns [C, D] prototypes and a boolean mask."""
    dim = embeddings.shape[-1]
    prototypes = embeddings.new_zeros(num_classes, dim)
    counts = embeddings.new_zeros(num_classes)
    prototypes.index_add_(0, labels, embeddings)
    ones = embeddings.new_ones(labels.shape[0])
    counts.index_add_(0, labels, ones)
    mask = counts > 0
    prototypes[mask] = prototypes[mask] / counts[mask].unsqueeze(1)
    return prototypes, mask


def prototype_stability_loss(
    z: torch.Tensor,
    z_cf: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """L_proto_cf = ||P_c - P_c_cf||^2 over classes present in the batch.

    z_cf is [B, D] or [B, K, D]. Prototypes are means of L2-normalized embeddings.
    """
    z_n = F.normalize(z, dim=-1)
    proto, mask = batch_feature_prototypes(z_n, labels, num_classes)
    if z_cf.dim() == 2:
        views = z_cf.unsqueeze(1)
    else:
        views = z_cf
    losses = []
    for view in range(views.shape[1]):
        z_view = F.normalize(views[:, view], dim=-1)
        proto_cf, mask_cf = batch_feature_prototypes(z_view, labels, num_classes)
        shared = mask & mask_cf
        if not bool(shared.any()):
            continue
        losses.append(F.mse_loss(proto[shared], proto_cf[shared]))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def _masked_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average [B, C, H, W] features inside a [B, 1, h, w] mask."""
    mask = F.adaptive_avg_pool2d(mask, feat.shape[-2:]).clamp(0, 1)
    weight = mask.sum(dim=(2, 3)).clamp_min(1e-4)
    pooled = (feat * mask).sum(dim=(2, 3)) / weight
    return F.normalize(pooled, dim=-1)


def lesion_asymmetric_loss(
    feat: torch.Tensor,
    feat_cf: torch.Tensor,
    preserve: torch.Tensor,
    preserve_cf: torch.Tensor,
    background: torch.Tensor,
    background_cf: torch.Tensor,
    move_eps: float = 0.02,
    similarity_margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep lesion features, and push background features apart.

    feat is [B, C, H, W]. Counterfactual tensors include a view dimension K.
    A view that leaves both regions unchanged is dropped, because that edit is
    already ignored and has no training signal.
    """
    if feat_cf.dim() == 4:
        feat_cf = feat_cf.unsqueeze(1)
        preserve_cf = preserve_cf.unsqueeze(1)
        background_cf = background_cf.unsqueeze(1)
    lesion = _masked_mean(feat, preserve)
    outside = _masked_mean(feat, background)
    keep_terms = []
    change_terms = []
    for view in range(feat_cf.shape[1]):
        lesion_cf = _masked_mean(feat_cf[:, view], preserve_cf[:, view])
        outside_cf = _masked_mean(feat_cf[:, view], background_cf[:, view])
        keep = ((lesion - lesion_cf) ** 2).sum(dim=-1)
        similarity = (outside * outside_cf).sum(dim=-1)
        change = (similarity - similarity_margin).clamp_min(0)
        moved = (keep.detach() > move_eps) | (similarity.detach() < 0.98)
        present = preserve_cf[:, view].flatten(1).sum(dim=-1) > 1.0
        use = moved & present
        if bool(use.any()):
            keep_terms.append(keep[use].mean())
            change_terms.append(change[use].mean())
    zero = feat.sum() * 0.0
    keep_loss = torch.stack(keep_terms).mean() if keep_terms else zero
    change_loss = torch.stack(change_terms).mean() if change_terms else zero
    return keep_loss, change_loss


def inter_class_separation(prototypes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pairwise L2 distance between class prototypes that are present."""
    present = prototypes[mask]
    count = present.shape[0]
    if count < 2:
        return prototypes.new_zeros(())
    distances = torch.cdist(present, present, p=2)
    # Exclude the zero diagonal.
    return distances.sum() / (count * (count - 1))
