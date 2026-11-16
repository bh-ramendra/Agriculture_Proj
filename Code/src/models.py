"""ResNet-50 and ViT classifiers, with optional style-sensitive distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from src.dgpldr import StyleSensitiveDistillation


def _resnet50(pretrained: bool) -> nn.Module:
    """Support both the torchvision 0.13 weights API and the 0.11 pretrained flag."""
    if hasattr(models, "ResNet50_Weights"):
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet50(weights=weights)
    return models.resnet50(pretrained=pretrained)


class PlantDiseaseModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_ssd: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.use_ssd = use_ssd and backbone == "resnet50"
        if use_ssd and backbone != "resnet50":
            raise ValueError("SSD is implemented for ResNet-50, which is the DG-PLDR reproduction backbone.")
        if backbone == "resnet50":
            net = _resnet50(pretrained)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.embedding_dim = 2048
        elif backbone == "vit_b_16":
            import timm

            self.features = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,
            )
            self.pool = nn.Identity()
            self.embedding_dim = self.features.num_features
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use resnet50 or vit_b_16.")
        self.ssd = StyleSensitiveDistillation() if self.use_ssd else None
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(images)
        if feat.dim() == 4:
            embedding = self.pool(feat).flatten(1)
        else:
            embedding = feat
            feat = None
        return feat, embedding

    def forward(
        self,
        images: torch.Tensor,
        style_images: torch.Tensor | None = None,
        return_map: bool = False,
    ):
        feat, embedding = self.encode(images)
        raw_map = feat
        if self.use_ssd and feat is not None:
            # Evaluation has no style view. Pairing the image with itself keeps
            # the classifier in the same normalized SSD feature space used in training.
            if style_images is None:
                feat_style = feat
            else:
                feat_style, _ = self.encode(style_images)
            feat, _ = self.ssd(feat, feat_style)
            embedding = self.pool(feat).flatten(1)
        logits = self.classifier(embedding)
        if return_map:
            return logits, embedding, raw_map
        return logits, embedding
