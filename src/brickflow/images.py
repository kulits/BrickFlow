"""Render augmentation and DINOv3 features."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.io import decode_jpeg, encode_jpeg
from transformers import AutoModel

from .model import compiled, second_sample

_YIQ = ((0.299, 0.587, 0.114), (0.596, -0.274, -0.322), (0.211, -0.523, 0.312))  # RGB -> luma and chroma
_PHOTOMETRIC = (0.0125, 0.0125, 0.009375, 0.009375, 0.000625)  # brightness, contrast, saturation, gamma, hue (turns)
_PHOTOMETRIC_P = (0.5, 0.5, 0.5, 0.5, 0.3, 0.3)  # probabilities: those five, then white balance
_WHITE_BALANCE = 0.00375
# blur, motion blur, resolution loss, JPEG, noise, vignette, aberration
_SENSOR_P = (0.3, 0.1, 0.25, 0.2, 0.3, 0.15, 0.1)


def _where(selected, a, b):
    """Per image: a where selected (B,), else b."""
    return torch.where(selected[:, None, None, None], a, b)


def _project(points, h):
    """Points (..., P, 2) through homographies h (..., 3, 3): (points, depth)."""
    p = torch.einsum("...ij,...pj->...pi", h, F.pad(points, (0, 1), value=1.0))
    return p[..., :2] / p[..., 2:].clamp(min=1e-12), p[..., 2]


def _pixel_centers(height, width, device):
    """(height * width, 2) normalized (x, y) of the pixel centers, row-major."""
    x = (torch.arange(width, device=device) + 0.5) / width
    y = (torch.arange(height, device=device) + 0.5) / height
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), -1).view(-1, 2)


def _rotation(axis, angle):
    """Rotations by `angle` about coordinate axis 0, 1, or 2."""
    i, j = (axis + 1) % 3, (axis + 2) % 3
    rotation = torch.eye(3, device=angle.device).expand(*angle.shape, 3, 3).clone()
    rotation[..., i, i] = rotation[..., j, j] = angle.cos()
    rotation[..., j, i] = angle.sin()
    rotation[..., i, j] = -rotation[..., j, i]
    return rotation


def _ray(k, point):
    """Unit camera rays through image points, for intrinsics k."""
    return F.normalize(torch.linalg.solve(k, F.pad(point, (0, 1), value=1.0)), dim=-1)


def _framing(k, box_center, roll, scale, offset):
    """Homographies H = K' Q K^-1 of a camera roll, re-aim, and zoom."""
    target = k.clone()
    target[..., :2, :2] *= scale[..., None, None]
    target[..., :2, 2] = 0.5 * target[..., 2:3, 2]
    goal = _ray(target, 0.5 + offset)
    rz = _rotation(2, roll)
    center = (rz @ _ray(k, box_center)[..., None])[..., 0]
    height = goal[..., 1] / center[..., 1:].norm(dim=-1).clamp(min=1e-8)
    rx = _rotation(0, torch.asin(height.clamp(-1, 1)) - torch.atan2(center[..., 1], -center[..., 2]))
    center = (rx @ center[..., None])[..., 0]
    yaw = torch.atan2(center[..., 0], -center[..., 2]) - torch.atan2(goal[..., 0], -goal[..., 2])
    return target @ _rotation(1, yaw) @ rx @ rz @ torch.linalg.inv(k)


def _bounding_box(alpha):
    """Center and extent (B, 2), as (x, y), of the region where alpha > 1e-4."""
    n = alpha.shape[-1]
    coordinate = (torch.arange(n, device=alpha.device) + 0.5) / n
    present = alpha > 1e-4
    occupied = torch.stack([present.any(1), present.any(2)], 1)
    lo = torch.where(occupied, coordinate, 1).amin(-1)
    hi = torch.where(occupied, coordinate, 0).amax(-1)
    return (lo + hi) / 2, (hi - lo).clamp(min=1 / n)


def _camera_change(k, alpha, points, requested, candidates=8):
    """First homography of `candidates` random camera changes that keeps the assembly in view, else the identity."""
    b = len(k)
    device = k.device
    shape = (b, candidates)
    narrow_roll = (torch.randn(shape, device=device) * 12).clamp(-30, 30)
    wide_roll = torch.rand(shape, device=device) * 90 - 45
    roll = torch.where(torch.rand(shape, device=device) < 0.85, narrow_roll, wide_roll)
    scale = 0.8 * (1.15 / 0.8) ** torch.rand(shape, device=device)
    box_center, extent = _bounding_box(alpha)
    room = ((1 - scale[..., None] * extent[:, None]) / 2).clamp(min=0)
    offset = (torch.rand(*shape, 2, device=device) * 2 - 1) * room
    k = k[:, None].expand(-1, candidates, -1, -1)
    box_center = box_center[:, None].expand(-1, candidates, -1)
    h = _framing(k, box_center, roll * math.pi / 180, scale, offset)

    moved, depth = _project(points[:, None].nan_to_num(0.5), h)
    in_frame = ((moved >= 0.01) & (moved <= 0.99)).all(-1) & (depth > 1e-6)
    absent = ~points.isfinite().all(-1)[:, None]
    points_ok = (in_frame | absent).all(-1)
    mapped, depth = _project(_pixel_centers(128, 128, device), h)
    in_view = ((mapped >= 0) & (mapped <= 1)).all(-1) & (depth > 1e-6)
    mass = alpha.flatten(1)[:, None]
    alpha_ok = (in_view * mass).sum(-1) >= 0.95 * mass.sum(-1)
    corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    corners_ok = (_project(corners, h)[1] > 1e-6).all(-1)
    valid = points_ok & alpha_ok & corners_ok & (torch.linalg.det(h) > 0)

    chosen = h[torch.arange(b, device=device), valid.long().argmax(-1)]
    use = requested & valid.any(-1)
    return torch.where(use[:, None, None], chosen, torch.eye(3, device=device))


def _warp(layers, h):
    """Resample (B, C, H, W) through homographies h: bilinear, zero outside."""
    b, _, height, width = layers.shape
    source, depth = _project(_pixel_centers(height, width, layers.device), torch.linalg.inv(h))
    grid = torch.where(depth[..., None] > 1e-6, source * 2 - 1, 2.0).view(b, height, width, 2)
    return F.grid_sample(layers, grid, padding_mode="zeros", align_corners=False)


def _gaussian(x, sigma, radius, padding="replicate"):
    """Separable Gaussian blur with a per-image sigma (B,), in pixels."""
    taps = torch.arange(-radius, radius + 1, device=x.device)
    weight = torch.softmax(-0.5 * (taps / sigma[:, None].clamp(min=0.01)) ** 2, -1)

    def band(n):  # (B, 1, n, n): row i averages pixels i + taps
        index = torch.arange(n, device=x.device)[:, None] + taps
        inside = (index >= 0) & (index < n)
        weights = weight[:, None] * (inside | (padding == "replicate"))
        matrix = torch.zeros(len(x), n, n, device=x.device)
        matrix.scatter_add_(2, index.clamp(0, n - 1).expand_as(weights), weights)
        return matrix[:, None]

    rows_blurred = x @ band(x.shape[3]).mT
    return band(x.shape[2]) @ rows_blurred


def _erase(layers, selected):
    """Random erasing of 3-20% of the assembly's alpha with uniform noise."""
    b, _, height, width = layers.shape
    device = layers.device
    alpha = layers[:, 3]
    mass = alpha.sum((1, 2))
    center = torch.multinomial(F.adaptive_avg_pool2d(alpha[:, None], 128).flatten(1) + 1e-12, 8, replacement=True)
    center_x = ((center % 128) + 0.5) * width / 128
    center_y = ((center // 128) + 0.5) * height / 128
    aspect = 0.3 + 3 * torch.rand(b, 8, device=device)
    target = (0.03 + 0.17 * torch.rand(b, 8, device=device)) * mass[:, None]
    integral = F.pad(alpha.cumsum(-1).cumsum(-2), (1, 0, 1, 0)).flatten(1)  # summed-area table

    def rectangle(area):
        w = (area * aspect).sqrt().round().long().clamp(1, width)
        h = (area / aspect).sqrt().round().long().clamp(1, height)
        x = (center_x - w / 2).round().long().clamp(min=0).minimum(width - w)
        y = (center_y - h / 2).round().long().clamp(min=0).minimum(height - h)

        def corner(dx, dy):
            return integral.gather(1, (y + dy) * (width + 1) + x + dx)

        covered = corner(w, h) - corner(0, h) - corner(w, 0) + corner(0, 0)
        return x, y, w, h, covered

    lo = torch.zeros_like(target)
    hi = torch.minimum(width * width / aspect, height * height * aspect)
    for _ in range(16):
        middle = (lo + hi) / 2
        reached = rectangle(middle)[-1] >= target
        lo = torch.where(reached, lo, middle)
        hi = torch.where(reached, middle, hi)
    x, y, w, h, covered = rectangle((lo + hi) / 2)
    fraction = covered / mass[:, None].clamp(min=1e-8)
    valid = (fraction >= 0.03) & (fraction <= 0.2) & (w / h >= 0.3) & (w / h <= 3.3) & (mass[:, None] > 0)
    first = valid.long().argmax(-1, keepdim=True)
    x, y, w, h = (value.gather(1, first)[..., None] for value in (x, y, w, h))
    row = torch.arange(height, device=device)[:, None]
    column = torch.arange(width, device=device)
    inside = (column >= x) & (column < x + w) & (row >= y) & (row < y + h)
    erased = inside & (selected & valid.any(-1))[:, None, None]
    noise = torch.cat([torch.rand_like(layers[:, :3]), torch.ones_like(layers[:, 3:])], 1)
    return torch.where(erased[:, None], noise, layers)


def _photometric_op(op, y, value, balance):
    """Photometric operation `op` on images y: 0 brightness, 1 contrast, 2 saturation, 3 gamma, 4 hue (a rotation of
    the chroma plane by `value` turns), 5 white balance."""
    if op == 0:
        return y * (1 + value)
    if op == 1:
        mean = y.mean((1, 2, 3), keepdim=True)
        return (y - mean) * (1 + value) + mean
    yiq = torch.tensor(_YIQ, device=y.device)
    if op == 2:
        gray = torch.einsum("c,bchw->bhw", yiq[0], y)[:, None]
        return (y - gray) * (1 + value) + gray
    if op == 3:
        return y.clamp(min=0) ** value.exp()
    if op == 4:
        turn = torch.linalg.inv(yiq) @ _rotation(0, 2 * math.pi * value[:, 0, 0, 0]) @ yiq
        return torch.einsum("bij,bjhw->bihw", turn, y)
    return y * balance[:, :, None, None]


def _photometric(x, selected):
    """Brightness, contrast, saturation, gamma, hue, and white balance, each on with its probability, in random
    order."""
    b = len(x)
    device = x.device
    value = (torch.rand(b, 5, device=device) * 2 - 1) * torch.tensor(_PHOTOMETRIC, device=device)
    balance = 1 + (torch.rand(b, 3, device=device) * 2 - 1) * _WHITE_BALANCE
    balance = balance / (balance * torch.tensor(_YIQ[0], device=device)).sum(-1, keepdim=True)  # unit luma
    on = (torch.rand(b, 6, device=device) < torch.tensor(_PHOTOMETRIC_P, device=device)) & selected[:, None]
    order = torch.rand(b, 6, device=device).argsort(-1)
    on, order = torch.stack([on.long(), order]).cpu()  # which images each step changes: one transfer to the host
    for position in range(6):
        for op in range(6):
            i = (on[:, op].bool() & (order[:, position] == op)).nonzero()[:, 0].to(device)
            if len(i):
                x[i] = _photometric_op(op, x[i], value[i, min(op, 4), None, None, None], balance[i]).clamp(0, 1)
    return x


def _jpeg(x, quality):
    """JPEG round trip in place of each image (B, 3, H, W) at its quality (B,) on the host; none at quality 0."""
    for q in quality.unique().tolist():
        if q > 0:
            i = (quality == q).nonzero()[:, 0].to(x.device)
            encoded = encode_jpeg(list((x[i] * 255).round().byte()), quality=q)
            x[i] = torch.stack(decode_jpeg([data.cpu() for data in encoded], device=x.device)).float() / 255
    return x


def _motion_blur(x, length, angle):
    """Line kernels of `length` pixels at `angle`, within 7 x 7, per image."""
    b, _, height, width = x.shape
    taps = torch.arange(-3.0, 4.0, device=x.device)
    yy, xx = torch.meshgrid(taps, taps, indexing="ij")
    cos = angle.cos()[:, None, None]
    sin = angle.sin()[:, None, None]
    along = xx * cos + yy * sin
    across = yy * cos - xx * sin
    kernel = (1 - across.abs()).clamp(min=0) * (along.abs() <= (length[:, None, None] - 1) / 2)
    kernel = kernel / kernel.sum((1, 2), keepdim=True)
    padded = F.pad(x.reshape(1, 3 * b, height, width), (3, 3, 3, 3), mode="replicate")
    return F.conv2d(padded, kernel.repeat_interleave(3, 0)[:, None], groups=3 * b).view_as(x)


def _sensor(x, active, force_blur):
    """Blur, motion blur, resolution loss (area downsampling, JPEG, bicubic upsampling), JPEG, noise, vignetting, and
    aberration (red and blue rescaled about the center), each on with its probability."""
    b, _, height, width = x.shape
    device = x.device
    on = (torch.rand(b, 7, device=device) < torch.tensor(_SENSOR_P, device=device)) & active[:, None]
    on[:, 0] |= force_blur

    sigma = torch.where(force_blur, 0.3 + 0.5 * torch.rand(b, device=device), 0.2 + torch.rand(b, device=device))
    x = _where(on[:, 0], _gaussian(x, sigma, 4), x)
    length = torch.randint(3, 8, (b,), device=device)
    angle = torch.rand(b, device=device) * math.pi
    x = _where(on[:, 1], _motion_blur(x, length, angle), x)
    level = torch.randint(4, (b,), device=device)
    quality = torch.tensor([47, 61, 74, 88], device=device)[torch.randint(4, (b,), device=device)]
    final = torch.tensor([78, 82, 88, 92], device=device)[torch.randint(4, (b,), device=device)] * on[:, 3]
    level, quality, final = torch.stack([torch.where(on[:, 2], level, -1), quality, final]).cpu()
    for f, factor in enumerate((0.35, 0.5, 0.7, 0.85)):
        i = (level == f).nonzero()[:, 0]
        if len(i):
            rows = i.to(device)
            small = F.interpolate(x[rows], size=(round(height * factor), round(width * factor)), mode="area")
            x[rows] = F.interpolate(_jpeg(small, quality[i]), size=(height, width), mode="bicubic").clamp(0, 1)
    x = _jpeg(x, final)
    read_noise = 0.02 * torch.rand(b, 1, 1, 1, device=device)
    shot_noise = 0.01 * torch.rand(b, 1, 1, 1, device=device)
    noisy = x + torch.randn_like(x) * (read_noise**2 + shot_noise * x).sqrt()
    x = _where(on[:, 4], noisy.clamp(0, 1), x)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing="ij"
    )
    vignette = 1 - 0.3 * torch.rand(b, 1, 1, 1, device=device) * (xx**2 + yy**2) / 2
    x = _where(on[:, 5], x * vignette, x)
    stretch = (torch.rand(b, 2, device=device) * 2 - 1) / (math.sqrt(2) * (width - 1) / 2)
    grid = torch.stack([xx, yy], -1)
    aberrated = x.clone()
    for channel, j in ((0, 0), (2, 1)):
        stretched = grid * (1 + stretch[:, j, None, None, None])
        plane = x[:, channel : channel + 1]
        aberrated[:, channel : channel + 1] = F.grid_sample(plane, stretched, padding_mode="border", align_corners=True)
    return _where(on[:, 6], aberrated, x)


def _background(crops, flip, kind):
    """1024 x 1024 backgrounds: the crops resized and flipped; white for kind 2, black for 3."""
    if not crops:
        return torch.ones(len(kind), 3, 1024, 1024, device=kind.device)
    images = [crop.permute(2, 0, 1)[None].float() for crop in crops]
    bg = torch.cat([F.interpolate(image, (1024, 1024), mode="bilinear", antialias=True) for image in images]) / 255
    bg = _where(flip, bg.flip(-1), bg)
    bg[kind == 2] = 1.0
    bg[kind == 3] = 0.0
    return bg


def _augment(rgba, crops, flip, k, points, gravity, resolution) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmented uint8 (B, resolution, resolution, 3) images and the keypoints (NaN: none) under their new cameras."""
    b = len(rgba)
    device = rgba.device
    u = torch.rand(b, device=device)
    kind = torch.bucketize(u, torch.tensor([0.765, 0.9, 0.95], device=device), right=True)
    if not crops:
        kind.fill_(2)
    active = ~((u >= 0.9) & (u < 0.925))
    source = rgba.permute(0, 3, 1, 2).float() / 255
    layers = torch.cat([source[:, :3] * source[:, 3:], source[:, 3:]], 1)  # premultiplied
    alpha_grid = F.adaptive_avg_pool2d(layers[:, 3:], 128)[:, 0]
    h = _camera_change(k, alpha_grid, points, torch.rand(b, device=device) < 0.5)

    shift = torch.eye(3, device=device).repeat(b, 1, 1)  # along world-down, in the render
    shift[:, :2, 2] = gravity * 0.03 * torch.rand(b, 1, device=device)
    blurred_alpha = _gaussian(layers[:, 3:], 8 + 16 * torch.rand(b, device=device), 72, padding="constant")
    opacity = 0.2 + 0.3 * torch.rand(b, 1, 1, 1, device=device)
    shadowed = active & (kind != 3) & (torch.rand(b, device=device) < 0.25)
    shadow = _warp(blurred_alpha, shift) * opacity * shadowed[:, None, None, None]
    layers, shadow = _warp(torch.cat([layers, shadow], 1), h).split([4, 1], 1)
    feathered = active & (torch.rand(b, device=device) < 0.3)
    layers = _where(feathered, _gaussian(layers, torch.rand(b, device=device), 3, padding="constant"), layers)
    layers = _erase(layers, active & (torch.rand(b, device=device) < 0.15 / 0.975))

    bg = _background(crops, flip, kind)
    bg = _where(kind == 1, _gaussian(bg, 4 + 12 * torch.rand(b, device=device), 48), bg)
    image = layers[:, :3] + (1 - layers[:, 3:]) * bg * (1 - shadow)
    image = F.interpolate(image, size=(resolution, resolution), mode="bicubic", antialias=True).clamp(0, 1)
    image = _photometric(image, active & (torch.rand(b, device=device) < 0.8))
    image = _sensor(image, active, active & (kind == 1) & (torch.rand(b, device=device) < 0.3))
    moved = torch.where(points.isfinite(), _project(points.nan_to_num(0.5), h)[0], float("nan"))
    return (image.permute(0, 2, 3, 1) * 255).round().byte(), moved


def training_images(batch: dict, encoder, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Image features (`DINO` or another encoder) and keypoints of a batch's augmented renders, two per row (zero/NaN
    where dropped); pops the images."""
    fields = [batch.pop(key) for key in ("image", "background", "flip", "k", "points", "gravity")]
    fields = [x if isinstance(x, list) else x.flatten(0, 1) for x in fields]
    features = []
    points = []
    for start in range(0, len(fields[0]), chunk):
        pixels, moved = _augment(*[x[start : start + chunk] for x in fields], encoder.resolution)
        features.append(encoder(pixels))
        points.append(moved)
    dropped = batch.pop("image_drop").flatten()[:, None, None]
    features = torch.where(dropped, 0, torch.cat(features))
    points = torch.where(dropped, float("nan"), torch.cat(points)).unflatten(0, (-1, 2))
    second = second_sample(batch["lengths"][:, 0], batch["types"].shape[1])
    row = torch.arange(len(points), device=dropped.device)[:, None]
    return features, points[row, second.long(), batch["original"]]


def on_white(image, resolution: int = 512) -> np.ndarray:
    """Centered square uint8 RGB image at the encoder's resolution."""
    image = image.convert("RGBA")
    side = max(image.size)
    square = Image.new("RGBA", (side, side), "white")
    square.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))
    return np.asarray(square.convert("RGB").resize((resolution, resolution), Image.BICUBIC))


class DINO(nn.Module):
    """Frozen DINOv3 ViT-L/16, the image encoder of IC models: uint8 (B, R, R, 3) RGB at `resolution` R -> the CLS
    and patch tokens after block 18 of 24 (B, 1 + P, D), in bf16, compiled once for every batch size."""

    resolution = 512

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval().requires_grad_(False)
        self.model.model.layer = self.model.model.layer[:18]
        for layer in self.model.model.layer:
            layer.mlp.act_fn = nn.GELU(approximate="tanh")
        self.model = compiled(self.model.to(torch.bfloat16), dynamic=True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        """uint8 (B, 512, 512, 3) -> fp16 (B, 1025, 1024): the CLS token, then the patch tokens."""
        x = (images.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        with torch.autocast("cuda", torch.bfloat16):
            hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states[18]
        return torch.cat([hidden[:, :1], hidden[:, 5:]], 1).half()
