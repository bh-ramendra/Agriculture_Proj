"""DINOv2 ViT-L/14 encoder that runs on PyTorch 1.10.

The public DINOv2 code uses newer PyTorch features. This file keeps the same
block layout and weight names, and only the classifier heads are trained.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(tokens)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(mixed)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.ls1 * self.attn(self.norm1(tokens))
        tokens = tokens + self.ls2 * self.mlp(self.norm2(tokens))
        return tokens


class DinoV2Encoder(nn.Module):
    """Frozen ViT-L/14. Returns the class token after the final LayerNorm."""

    def __init__(self, embed_dim: int = 1024, depth: int = 24, num_heads: int = 16, patch_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        grid = 518 // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, embed_dim))
        self.blocks = nn.ModuleList(_Block(embed_dim, num_heads, mlp_ratio=4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, tokens: torch.Tensor, width: int, height: int) -> torch.Tensor:
        patch_count = tokens.shape[1] - 1
        stored = self.pos_embed.shape[1] - 1
        if patch_count == stored:
            return self.pos_embed
        class_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = tokens.shape[-1]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        side = int(math.sqrt(stored))
        patch_pos = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2).float()
        scale_y = float(grid_h + 0.1) / side
        scale_x = float(grid_w + 0.1) / side
        resized = F.interpolate(patch_pos, scale_factor=(scale_y, scale_x), mode="bicubic", align_corners=False)
        if resized.shape[-2:] != (grid_h, grid_w):
            resized = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        resized = resized.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos, resized), dim=1)

    def forward(self, images: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        batch, _, height, width = images.shape
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, width, height)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        if return_patches:
            return tokens
        return tokens[:, 0]


def pool_tokens(tokens: torch.Tensor, mode: str = "cls", topk: int = 8) -> torch.Tensor:
    """Build a frozen feature vector from DINOv2 tokens.

    cls: class token only (1024-d)
    cls_mean: DINOv2 linear-eval recipe, class token + mean of patches (2048-d)
    cls_mean_salient: above plus the mean of the top-k patches farthest from the
    patch mean, a source-only lesion proxy (3072-d)
    """
    cls = tokens[:, 0]
    patches = tokens[:, 1:]
    if mode == "cls":
        return cls
    patch_mean = patches.mean(dim=1)
    if mode == "cls_mean":
        return torch.cat((cls, patch_mean), dim=-1)
    if mode == "cls_mean_salient":
        deviation = (patches - patch_mean.unsqueeze(1)).pow(2).sum(dim=-1)
        keep = min(topk, patches.shape[1])
        index = deviation.topk(keep, dim=1).indices
        gather = index.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        salient = patches.gather(1, gather).mean(dim=1)
        return torch.cat((cls, patch_mean, salient), dim=-1)
    raise ValueError(f"unknown pool mode {mode}")


def load_dinov2(weights: str | Path) -> DinoV2Encoder:
    payload = torch.load(str(weights), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("model", payload.get("state_dict", payload.get("teacher", payload)))
    else:
        state = payload
    cleaned = {}
    for key, value in state.items():
        name = key.replace("module.", "").replace("backbone.", "")
        name = name.replace("patch_embed.proj.", "patch_embed.")
        if name.startswith("blocks.") and ".ls1.gamma" in name:
            name = name.replace(".ls1.gamma", ".ls1")
        if name.startswith("blocks.") and ".ls2.gamma" in name:
            name = name.replace(".ls2.gamma", ".ls2")
        cleaned[name] = value
    model = DinoV2Encoder()
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    real_missing = [key for key in missing if key != "mask_token"]
    if real_missing or any(not key.startswith("mask_token") for key in unexpected):
        unexpected = [key for key in unexpected if key != "mask_token"]
        if real_missing or unexpected:
            raise RuntimeError(f"DINOv2 weight mismatch. missing={real_missing[:8]} unexpected={unexpected[:8]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
