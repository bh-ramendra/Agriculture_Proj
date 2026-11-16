"""Disease-preserving counterfactual interventions.

Each intervention changes a nuisance factor and keeps the leaf (the region that
carries lesion evidence) as close to the original as possible.

* background: replace non-leaf pixels with a synthetic field-like background
* illumination: gamma and brightness change on the whole image
* style: Reinhard color transfer, applied strongly on the background and only
  lightly on the leaf

Inputs and outputs are float tensors in [0, 1], shaped [3, H, W] or [B, 3, H, W].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


INTERVENTIONS = ("background", "illumination", "style")


def excess_green_mask(images: torch.Tensor) -> torch.Tensor:
    """Soft leaf mask from the excess-green index. Returns [B, 1, H, W]."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    red, green, blue = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    excess = 2.0 * green - red - blue
    green_dom = (green > red) & (green > blue * 0.9)
    mask = ((excess > 0.05) & green_dom).float()
    mask = _close(mask, kernel=7)
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2).clamp(0, 1)
    return mask


def _close(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - dilated, kernel, stride=1, padding=pad)
    return eroded


def _synthetic_background(batch: int, height: int, width: int, device, generator) -> torch.Tensor:
    choice = int(torch.randint(0, 3, (1,), generator=generator).item())
    if choice == 0:
        # Soil: brown base plus spatial noise.
        base = torch.tensor([0.45, 0.32, 0.18], device=device).view(1, 3, 1, 1)
        noise = torch.rand(batch, 1, height, width, device=device, generator=generator) * 0.25
        background = (base + noise).clamp(0, 1)
    elif choice == 1:
        # Low-frequency field color, upsampled so it looks like a blurred scene.
        coarse_h = max(4, height // 16)
        coarse_w = max(4, width // 16)
        kwargs = {"device": device}
        if generator is not None:
            kwargs["generator"] = generator
        coarse = torch.rand(batch, 3, coarse_h, coarse_w, **kwargs)
        # Keep the field in soil and foliage colors. Full-range noise looks artificial.
        earth = torch.tensor([0.42, 0.38, 0.22], device=device).view(1, 3, 1, 1)
        coarse = (earth + (coarse - 0.5) * 0.18).clamp(0, 1)
        background = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    else:
        gray = 0.35 + 0.3 * torch.rand(batch, 1, 1, 1, device=device, generator=generator)
        noise = 0.05 * torch.randn(batch, 1, height, width, device=device, generator=generator)
        background = (gray + noise).expand(batch, 3, height, width).clamp(0, 1)
    return background


def replace_background(images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    single = images.dim() == 3
    if single:
        images = images.unsqueeze(0)
    mask = excess_green_mask(images)
    background = _synthetic_background(
        images.shape[0], images.shape[2], images.shape[3], images.device, generator
    )
    counterfactual = images * mask + background * (1.0 - mask)
    return counterfactual[0] if single else counterfactual


def _uniform(shape, low: float, high: float, device, generator: torch.Generator | None) -> torch.Tensor:
    kwargs = {"device": device}
    if generator is not None:
        kwargs["generator"] = generator
    return low + (high - low) * torch.rand(shape, **kwargs)


def change_illumination(
    images: torch.Tensor,
    generator: torch.Generator | None = None,
    gamma_range: tuple[float, float] = (0.65, 1.45),
    brightness_range: tuple[float, float] = (0.75, 1.25),
) -> torch.Tensor:
    single = images.dim() == 3
    if single:
        images = images.unsqueeze(0)
    batch = images.shape[0]
    gamma = _uniform((batch, 1, 1, 1), gamma_range[0], gamma_range[1], images.device, generator)
    brightness = _uniform(
        (batch, 1, 1, 1), brightness_range[0], brightness_range[1], images.device, generator
    )
    counterfactual = (images.clamp(1e-4, 1.0) ** gamma) * brightness
    counterfactual = counterfactual.clamp(0, 1)
    return counterfactual[0] if single else counterfactual


def _reinhard(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    src_mean = source.mean(dim=(2, 3), keepdim=True)
    src_std = source.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    ref_mean = reference.mean(dim=(2, 3), keepdim=True)
    ref_std = reference.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    return ((source - src_mean) / src_std * ref_std + ref_mean).clamp(0, 1)


def change_style(
    images: torch.Tensor,
    generator: torch.Generator | None = None,
    leaf_strength: float = 0.15,
    background_strength: float = 0.9,
) -> torch.Tensor:
    """Match color statistics to a rolled batch image, mostly off the leaf."""
    single = images.dim() == 3
    if single:
        images = images.unsqueeze(0)
    if images.shape[0] == 1:
        # A single image has no partner. Shift its own channel means instead.
        shift = _uniform((1, 3, 1, 1), -0.12, 0.12, images.device, generator)
        reference = (images + shift).clamp(0, 1)
    else:
        shift = int(torch.randint(1, images.shape[0], (1,), generator=generator).item())
        reference = images.roll(shift, dims=0)
    styled = _reinhard(images, reference)
    mask = excess_green_mask(images)
    weight = mask * leaf_strength + (1.0 - mask) * background_strength
    counterfactual = images * (1.0 - weight) + styled * weight
    return counterfactual[0] if single else counterfactual


def apply_intervention(
    images: torch.Tensor,
    intervention: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if intervention == "background":
        return replace_background(images, generator)
    if intervention == "illumination":
        return change_illumination(images, generator)
    if intervention == "style":
        return change_style(images, generator)
    raise ValueError(f"Unknown intervention '{intervention}'. Expected one of {INTERVENTIONS}.")


def make_counterfactuals(
    images: torch.Tensor,
    num_views: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Build K counterfactual views, cycling through the three interventions.

    Returns [B, K, 3, H, W] and the intervention name used for each view.
    """
    single = images.dim() == 3
    if single:
        images = images.unsqueeze(0)
    views = []
    names = []
    for index in range(num_views):
        name = INTERVENTIONS[index % len(INTERVENTIONS)]
        views.append(apply_intervention(images, name, generator))
        names.append(name)
    stacked = torch.stack(views, dim=1)
    if single:
        return stacked[0], names
    return stacked, names


# Training backgrounds and the checkpoint stress set use different color families
# so model selection is not done on the same pictures the loss trains on.
# These are synthesized textures. They are not PlantDoc photographs.
_TRAIN_PALETTES = (
    (0.55, 0.72, 0.88),
    (0.36, 0.26, 0.16),
    (0.22, 0.38, 0.16),
    (0.48, 0.45, 0.40),
    (0.42, 0.27, 0.14),
    (0.62, 0.55, 0.32),
)
_STRESS_PALETTES = (
    (0.25, 0.28, 0.38),
    (0.55, 0.55, 0.56),
    (0.15, 0.32, 0.38),
    (0.33, 0.18, 0.12),
)


def _hard_green(images: torch.Tensor) -> torch.Tensor:
    red, green, blue = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    excess = 2.0 * green - red - blue
    return ((excess > 0.05) & (green > red) & (green > blue * 0.9)).float()


def _erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    return 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=pad)


def leaf_and_preserve(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaf mask and the region the edit must keep.

    A close fills lesion holes inside the leaf. The lesion is the interior that
    is not green, so the morphological halo around a healthy leaf is not treated
    as disease. Healthy leaves keep the whole leaf.
    """
    hard = _hard_green(images)
    leaf = _close(hard, kernel=17)
    interior = _erode(leaf, kernel=7)
    lesion = _close(interior * (1.0 - hard), kernel=5)
    lesion_area = lesion.flatten(2).sum(dim=-1)
    leaf_area = (leaf > 0.5).flatten(2).float().sum(dim=-1).clamp_min(1.0)
    use_leaf = (lesion_area / leaf_area) < 0.02
    preserve = torch.where(use_leaf.view(-1, 1, 1, 1), (leaf > 0.5).float(), lesion)
    return leaf.clamp(0, 1), preserve.clamp(0, 1)


def _blur(images: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return images
    pad = kernel // 2
    return F.avg_pool2d(images, kernel_size=kernel, stride=1, padding=pad)


def _texture(
    batch: int,
    height: int,
    width: int,
    color: tuple[float, float, float],
    device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Multi-scale color texture. Harder than a flat soil fill, and not a field photo."""
    accumulator = torch.zeros(batch, 1, height, width, device=device)
    weight = 0.0
    amplitude = 1.0
    size = 4
    while size < max(height, width):
        kwargs = {"device": device}
        if generator is not None:
            kwargs["generator"] = generator
        noise = torch.rand(batch, 1, size, size, **kwargs)
        noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
        accumulator = accumulator + amplitude * noise
        weight += amplitude
        amplitude *= 0.55
        size *= 2
    shade = (accumulator / max(weight, 1e-4)).clamp(0, 1)
    base = torch.tensor(color, device=device).view(1, 3, 1, 1)
    return (base * (0.45 + shade)).clamp(0, 1)


def _sample_background(
    batch: int,
    height: int,
    width: int,
    device,
    generator: torch.Generator | None,
    stress: bool,
) -> torch.Tensor:
    palettes = _STRESS_PALETTES if stress else _TRAIN_PALETTES
    index = int(torch.randint(0, len(palettes), (1,), generator=generator).item())
    return _texture(batch, height, width, palettes[index], device, generator)


def _zoom_out(
    images: torch.Tensor,
    masks: list[torch.Tensor],
    scale: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    batch, _, height, width = images.shape
    new_h = min(height, max(8, int(round(height * scale))))
    new_w = min(width, max(8, int(round(width * scale))))
    scaled = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
    scaled_masks = [F.interpolate(mask, size=(new_h, new_w), mode="nearest") for mask in masks]
    canvas = images.new_zeros(images.shape)
    mask_canvas = [images.new_zeros(mask.shape) for mask in masks]
    for item in range(batch):
        top_span = height - new_h
        left_span = width - new_w
        top = int(torch.randint(0, top_span + 1, (1,), generator=generator).item()) if top_span else 0
        left = int(torch.randint(0, left_span + 1, (1,), generator=generator).item()) if left_span else 0
        canvas[item, :, top : top + new_h, left : left + new_w] = scaled[item]
        for destination, source in zip(mask_canvas, scaled_masks):
            destination[item, :, top : top + new_h, left : left + new_w] = source[item]
    return canvas, mask_canvas


def _healthy_holes(
    healthy: torch.Tensor,
    strength: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    holes = torch.zeros_like(healthy)
    batch, _, height, width = healthy.shape
    count = int(2 + 5 * strength)
    for _ in range(count):
        hole_h = max(2, int(height * (0.06 + 0.10 * strength)))
        hole_w = max(2, int(width * (0.06 + 0.10 * strength)))
        for item in range(batch):
            top_span = max(height - hole_h, 0)
            left_span = max(width - hole_w, 0)
            top = int(torch.randint(0, top_span + 1, (1,), generator=generator).item()) if top_span else 0
            left = int(torch.randint(0, left_span + 1, (1,), generator=generator).item()) if left_span else 0
            holes[item, :, top : top + hole_h, left : left + hole_w] = 1.0
    return holes * (healthy > 0.5).float()


def _mean_color(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.sum(dim=(2, 3)).clamp_min(1.0)
    return (images * mask).sum(dim=(2, 3)) / weight


def _compose_boundary(
    images: torch.Tensor,
    strength: float,
    generator: torch.Generator | None,
    stress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One hard edit. Returns image, preserve mask, background mask, and the original preserve mask."""
    leaf, preserve = leaf_and_preserve(images)
    scale = 0.92 - 0.38 * float(strength)
    scaled, (scaled_leaf, scaled_preserve) = _zoom_out(images, [leaf, preserve], scale, generator)
    healthy = (scaled_leaf > 0.5) & (scaled_preserve < 0.5)
    holes = _healthy_holes(healthy.float(), strength, generator)
    kept_leaf = (scaled_leaf * (1.0 - holes)).clamp(0, 1)
    kept_preserve = (scaled_preserve * (1.0 - holes)).clamp(0, 1)
    background = _sample_background(
        images.shape[0], images.shape[2], images.shape[3], images.device, generator, stress
    )
    blur_kernel = 9 if strength < 0.8 else 15
    background = _blur(background, blur_kernel)
    cast = _uniform((images.shape[0], 3, 1, 1), 1.0 - 0.55 * strength, 1.0 + 0.55 * strength, images.device, generator)
    healthy_leaf = (kept_leaf * (1.0 - kept_preserve)).clamp(0, 1)
    leaf_cast = 1.0 + (cast - 1.0) * 0.2
    tinted_leaf = (scaled * leaf_cast).clamp(0, 1)
    tinted_background = (background * cast).clamp(0, 1)
    # Lesion pixels are copied unchanged. Healthy leaf tissue is only lightly tinted.
    counterfactual = tinted_background * (1.0 - kept_leaf) + tinted_leaf * healthy_leaf + scaled * kept_preserve
    background_mask = (1.0 - kept_leaf).clamp(0, 1)
    return counterfactual.clamp(0, 1), kept_preserve, background_mask, preserve


def _accepted(images: torch.Tensor, counterfactual: torch.Tensor, preserve_cf: torch.Tensor, preserve: torch.Tensor) -> bool:
    original_area = float(preserve.sum())
    if original_area < 1.0:
        return False
    area_ratio = float(preserve_cf.sum()) / original_area
    if area_ratio < 0.22:
        return False
    color_shift = (_mean_color(counterfactual, preserve_cf) - _mean_color(images, preserve)).abs().mean()
    return float(color_shift) < 0.12


def make_boundary_counterfactuals(
    images: torch.Tensor,
    num_views: int = 1,
    generator: torch.Generator | None = None,
    stress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard nuisance edits that keep the lesion (or the whole leaf, if it is healthy).

    Each view pastes the leaf onto a natural texture, shrinks it, covers healthy
    tissue, blurs the background, and shifts background color. PlantDoc images
    are not used. The hardest strength whose preserved region still overlaps is kept.

    Returns counterfactuals [B, K, 3, H, W], the original preserve mask [B, 1, H, W],
    per-view preserve masks [B, K, 1, H, W], the original background mask, and
    per-view background masks.
    """
    single = images.dim() == 3
    if single:
        images = images.unsqueeze(0)
    strengths = (0.8,) if stress else (1.0, 0.75, 0.5, 0.3)
    leaf, preserve = leaf_and_preserve(images)
    original_background = (1.0 - (leaf > 0.5).float()).clamp(0, 1)
    view_images = []
    view_preserve = []
    view_background = []
    for _ in range(num_views):
        chosen = None
        for strength in strengths:
            candidate = _compose_boundary(images, strength, generator, stress)
            if _accepted(images, candidate[0], candidate[1], candidate[3]):
                chosen = candidate
                break
        if chosen is None:
            chosen = _compose_boundary(images, strengths[-1], generator, stress)
        view_images.append(chosen[0])
        view_preserve.append(chosen[1])
        view_background.append(chosen[2])
    counterfactuals = torch.stack(view_images, dim=1)
    preserve_cf = torch.stack(view_preserve, dim=1)
    background_cf = torch.stack(view_background, dim=1)
    if single:
        return counterfactuals[0], preserve[0], preserve_cf[0], original_background[0], background_cf[0]
    return counterfactuals, preserve, preserve_cf, original_background, background_cf
