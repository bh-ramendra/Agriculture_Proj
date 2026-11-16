"""Style-sensitive distillation from DG-PLDR (Zhan et al., 2025).

The public DG-PLDR repository (V1) trains a standard classifier on the source
domain. It does not ship the SSD / LPMB / LPMS modules. This file reimplements
those modules from Section 4 of the paper.

Deviations required to make the published equations runnable:

* SSD is applied to the last ResNet feature map only. Building an HW x HW
  relation matrix at every encoder stage is not feasible (early maps are
  56 x 56 and larger).
* Tokens are L2-normalized before the relation product so cosine relations stay
  in [-1, 1]. The paper writes an unnormalized product.
* Equation (12) is implemented as Z + (R_hat @ Z) / HW. R is HW x HW and Z is
  HW x C, so R @ Z is the product that is dimensionally valid. Dividing by HW
  keeps the residual from drowning the pretrained features.
* The hard threshold in Equation (9) is replaced by a steep sigmoid gate so the
  learnable threshold receives gradient. It is initialized at 0.5.
* Row-wise softmax is used in Equation (8). A single softmax over all HW*HW
  pairs makes D almost identically 1, so the mask would never remove anything.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleSensitiveDistillation(nn.Module):
    def __init__(self, init_threshold: float = 0.5, gate_sharpness: float = 20.0):
        super().__init__()
        # sigmoid(0) = 0.5, which is the paper's initial threshold.
        initial = torch.logit(torch.tensor(float(init_threshold)))
        self.threshold = nn.Parameter(initial)
        self.gate_sharpness = gate_sharpness

    def _tokens(self, feat: torch.Tensor) -> torch.Tensor:
        tokens = feat.flatten(2).transpose(1, 2)  # B, HW, C
        return F.normalize(tokens, dim=-1)

    def forward(self, feat: torch.Tensor, feat_aug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._tokens(feat)
        z_aug = self._tokens(feat_aug)
        relation = torch.matmul(z, z.transpose(1, 2))
        relation_aug = torch.matmul(z_aug, z_aug.transpose(1, 2))
        difference = (relation - relation_aug).abs()
        # Equation (8), with softmax along each token's relations.
        insensitive = 1.0 - torch.softmax(difference, dim=-1)
        threshold = torch.sigmoid(self.threshold)
        mask = torch.sigmoid(self.gate_sharpness * (insensitive - threshold))
        relation_hat = relation * mask
        relation_hat_aug = relation_aug * mask
        hw = z.shape[1]
        z_hat = z + torch.matmul(relation_hat, z) / hw
        z_hat_aug = z_aug + torch.matmul(relation_hat_aug, z_aug) / hw
        return self._to_map(z_hat, feat.shape), self._to_map(z_hat_aug, feat_aug.shape)

    @staticmethod
    def _to_map(tokens: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
        batch, channels, height, width = spatial_shape
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)
