"""DiT denoiser on part poses."""

import functools
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from procbrick import canonical_id, load_appearance_policy

from .parts import TYPE_CLASSES
from .paths import find

try:
    from . import _triton
except ImportError:
    _triton = None

PART_TYPES = 14583  # catalog part ids
UNKNOWN_MATERIAL = 260
MAX_PARTS = 100
SIZES = {"S": (384, 12), "B": (768, 12), "L": (1024, 24)}  # width, depth
_MATERIAL_CLASSES = 272  # ProcBrick's 260 materials and "unknown", padded to a multiple of 16
_LATENTS = 256
_KEYPOINT_BLOCK = 5  # the keypoint head reads the stream after block 6


@functools.lru_cache(maxsize=4)
def _materials(assets: Path) -> dict[int, int]:
    return {int(color): m for m, color in enumerate(load_appearance_policy(assets)["colors"])}


def material_ids(colors, assets: str | Path = "assets") -> np.ndarray:
    """IC models' material of each LDraw color: its index in ProcBrick's appearance policy, else UNKNOWN_MATERIAL."""
    materials = _materials(find(assets))
    return np.array([materials.get(int(color), UNKNOWN_MATERIAL) for color in colors], np.int64)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices of quaternions (w, x, y, z), not necessarily unit."""
    w, x, y, z = q.unbind(-1)
    s = 2 / (q * q).sum(-1)
    rows = [
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ]
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def rotvec_to_quat(v: torch.Tensor) -> torch.Tensor:
    """Exp: rotation vectors to quaternions."""
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), v * 0.5 * torch.sinc(angle / (2 * math.pi))], -1)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Log of quaternions with w >= 0."""
    half_angle = torch.atan2(q[..., 1:].norm(dim=-1, keepdim=True), q[..., :1])
    return q[..., 1:] / (0.5 * torch.sinc(half_angle / math.pi))


def second_sample(split, n: int) -> torch.Tensor:
    """(B, n) mask of the tokens of each packed row's second sample, which starts at token split (B,)."""
    return torch.arange(n, device=split.device) >= split[:, None]


def compiled(function: Callable, **options) -> Callable:
    """Compile with Triton when available."""
    return function if _triton is None else torch.compile(function, **options)


def packed_attention(q, k, v, split) -> torch.Tensor:
    """softmax(q k^T / 8) v within each sample; q, k, v (B, N, H, 64), split (B,) the first sample's size."""
    if _triton is not None and q.is_cuda and q.shape[1] <= _triton._TILE.value:
        return _triton.packed_attention(q, k, v, split)
    first = ~second_sample(split, q.shape[1])
    same = (first[:, :, None] == first[:, None, :])[:, None]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=same)
    return out.transpose(1, 2)


def _modulate(x, shift, scale) -> torch.Tensor:
    """LayerNorm(x) (1 + scale) + shift, without affine parameters."""
    return F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale) + shift


def residual_norm(x, y, gate, shift, scale) -> tuple[torch.Tensor, torch.Tensor]:
    """(x + gate y, _modulate(x + gate y, shift, scale))."""
    if _triton is not None and x.is_cuda and torch.is_grad_enabled():
        return _triton.residual_norm(x, y, gate, shift, scale)
    x = x + gate * y
    return x, _modulate(x, shift, scale)


def _split_heads(x):
    """(B, L, H * 64) -> (B, H, L, 64)."""
    return x.unflatten(-1, (-1, 64)).transpose(1, 2)


def _merge_heads(x):
    """(B, H, L, 64) -> (B, L, H * 64)."""
    return x.transpose(1, 2).flatten(2)


def _stacked(depth, out, inp, zero=False):
    """Layer-stacked weights (depth, out, in) and biases with nn.Linear's initialization."""
    bound = 0.0 if zero else inp**-0.5
    weight = nn.Parameter(torch.empty(depth, out, inp).uniform_(-bound, bound))
    bias = nn.Parameter(torch.empty(depth, out).uniform_(-bound, bound))
    return weight, bias


def _zeroed(module):
    """`module` with its parameters set to zero."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class _ScalarEmbedding(nn.Module):
    """MLP of the sinusoidal features of 1000 s."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp_in = nn.Linear(256, width)
        self.mlp_out = nn.Linear(width, width)

    def forward(self, s) -> torch.Tensor:
        """Embedding of scalars s in [0, 1]."""
        frequencies = torch.exp(-math.log(1e4) / 128 * torch.arange(128, device=s.device))
        angle = 1000 * s[..., None] * frequencies
        features = torch.cat([angle.cos(), angle.sin()], -1)
        return self.mlp_out(F.silu(self.mlp_in(features)))


class _Head(nn.Module):
    """Linear(_modulate(x, shift, scale)) with (shift, scale) from the conditioning; zero init."""

    def __init__(self, width: int, out: int) -> None:
        super().__init__()
        self.adaln = _zeroed(nn.Linear(width, 2 * width))
        self.linear = _zeroed(nn.Linear(width, out))

    def forward(self, x, c) -> torch.Tensor:
        """Head output for tokens x with conditioning c."""
        shift, scale = self.adaln(c)[:, None].chunk(2, -1)
        return self.linear(_modulate(x, shift, scale))


class _Perceiver(nn.Module):
    """_LATENTS learned queries cross-attending to the image tokens in pre-LN layers."""

    def __init__(self, width: int, features: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(_LATENTS, width) * 0.02)
        self.feature_proj = nn.Linear(features, width)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                dict(
                    norm_q=nn.LayerNorm(width),
                    norm_kv=nn.LayerNorm(width),
                    q=nn.Linear(width, width, bias=False),
                    kv=nn.Linear(width, 2 * width, bias=False),
                    attn_out=_zeroed(nn.Linear(width, width)),
                    norm_mlp=nn.LayerNorm(width),
                    mlp_in=nn.Linear(width, 4 * width),
                    mlp_out=_zeroed(nn.Linear(4 * width, width)),
                )
            )
            for _ in range(6)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens) -> torch.Tensor:
        """Latents read from image tokens."""
        latents = self.queries.expand(len(tokens), -1, -1)
        for layer in self.layers:
            query = _split_heads(layer["q"](layer["norm_q"](latents)))
            key, value = layer["kv"](layer["norm_kv"](tokens)).chunk(2, -1)
            read = F.scaled_dot_product_attention(query, _split_heads(key), _split_heads(value))
            latents = latents + layer["attn_out"](_merge_heads(read))
            latents = latents + layer["mlp_out"](F.gelu(layer["mlp_in"](layer["norm_mlp"](latents))))
        return self.norm(latents)


class _Blocks(nn.Module):
    """Layer-stacked pre-LN DiT blocks with adaLN-Zero and RMS-normalized q and k; `_w` weights go to Muon."""

    def __init__(self, width: int, depth: int, image: bool) -> None:
        super().__init__()
        self.depth = depth
        self.adaln_w, self.adaln_b = _stacked(depth, 6 * width, width, zero=True)
        self.qkv_w, self.qkv_b = _stacked(depth, 3 * width, width)
        self.attn_out_w, self.attn_out_b = _stacked(depth, width, width)
        self.mlp_in_w, self.mlp_in_b = _stacked(depth, 4 * width, width)
        self.mlp_out_w, self.mlp_out_b = _stacked(depth, width, 4 * width)
        if image:
            self.image_kv_w, _ = _stacked(depth, 2 * width, width)
            self.image_out_w, self.image_out_b = _stacked(depth, width, width, zero=True)
            self.image_gate_w, self.image_gate_b = _stacked(depth, width, width, zero=True)
            nn.init.constant_(self.image_gate_b, 0.1)
            self.image_q_w, _ = _stacked(depth, width, width)

    def read_images(self, i: int, h, image_kv, split) -> torch.Tensor:
        """Block i's cross-attention to the images; in a packed row each query keeps its own sample's image."""
        batch = len(h)
        images = len(image_kv) // batch
        query = F.rms_norm(_split_heads(F.linear(h, self.image_q_w[i])), (64,), eps=1e-6)
        key, value = image_kv.unflatten(-1, (2, -1, 64)).permute(2, 0, 3, 1, 4)
        read = F.scaled_dot_product_attention(query.repeat_interleave(images, 0), key, value)
        read = read.unflatten(0, (batch, images))
        own = read[:, 0]
        if images == 2:
            own = torch.where(second_sample(split, h.shape[1])[:, None, :, None], read[:, 1], own)
        return F.linear(_merge_heads(own), self.image_out_w[i], self.image_out_b[i])

    def forward(self, x, c, split, image_kv=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stream after the last block, and after block _KEYPOINT_BLOCK + 1 for the keypoint head (IC models; else
        None)."""
        c = c.expand(self.depth, -1, -1)
        modulation = torch.baddbmm(self.adaln_b[:, None], c, self.adaln_w.mT)[:, :, None]
        shift1, scale1, gate1, shift2, scale2, gate2 = zip(*[m.chunk(6, -1) for m in modulation.unbind(0)])
        if image_kv is not None:
            image_gate = torch.baddbmm(self.image_gate_b[:, None], c, self.image_gate_w.mT)[:, :, None]
        h = _modulate(x, shift1[0], scale1[0])
        keypoint_input = None
        for i in range(self.depth):
            q, k, v = F.linear(h, self.qkv_w[i], self.qkv_b[i]).chunk(3, -1)
            q = F.rms_norm(_split_heads(q), (64,), eps=1e-6).transpose(1, 2)
            k = F.rms_norm(_split_heads(k), (64,), eps=1e-6).transpose(1, 2)
            attention = packed_attention(q, k, v.unflatten(-1, (-1, 64)), split).flatten(2)
            branch = F.linear(attention, self.attn_out_w[i], self.attn_out_b[i])
            gate = gate1[i]
            if image_kv is not None:
                x = x + gate * branch
                branch = self.read_images(i, h, image_kv[i], split)
                gate = image_gate[i]
            x, h = residual_norm(x, branch, gate, shift2[i], scale2[i])
            mlp = F.relu(F.linear(h, self.mlp_in_w[i], self.mlp_in_b[i])).square()
            branch = F.linear(mlp, self.mlp_out_w[i], self.mlp_out_b[i])
            if i + 1 == self.depth:
                x = x + gate2[i] * branch
            else:
                x, h = residual_norm(x, branch, gate2[i], shift1[i + 1], scale1[i + 1])
            if i == _KEYPOINT_BLOCK:
                keypoint_input = x
        return x, keypoint_input


class Denoiser(nn.Module):
    """BrickFlow denoiser of `width` and `depth` (`SIZES`); with `image`, an IC model reading an image encoder's
    `features`-wide tokens: a summary token, then `patches` patch tokens (DINOv3: 1024 and 32 x 32)."""

    def __init__(self, width: int, depth: int, image: bool = False, features: int = 1024, patches: int = 1024) -> None:
        super().__init__()
        if width % 64:
            raise ValueError(f"width {width} is not a multiple of 64, the attention head size")
        if image and depth <= _KEYPOINT_BLOCK:
            raise ValueError(f"an IC model of depth {depth}: its keypoint head reads block {_KEYPOINT_BLOCK + 1}")
        self.image = image
        self.type_emb = nn.Embedding(TYPE_CLASSES, 16)
        self.type_proj = nn.Linear(16, width, bias=False)
        nn.init.normal_(self.type_proj.weight, std=0.25)
        self.trans_proj = nn.Linear(3, width)
        self.rot_proj = nn.Linear(6, width)
        self.time_emb = _ScalarEmbedding(width)
        self.size_emb = _ScalarEmbedding(width)
        _zeroed(self.size_emb.mlp_out)
        if image:
            self.material_emb = nn.Embedding(_MATERIAL_CLASSES, 8)
            self.material_proj = _zeroed(nn.Linear(8, width, bias=False))  # IC models start as their PT model
            self.image_pos_emb = nn.Parameter(torch.randn(patches, width) * 0.02)
            self.perceiver = _Perceiver(width, features)
        self.blocks = _Blocks(width, depth, image)
        self.head_trans = _Head(width, 3)
        self.head_rot = _Head(width, 3)
        if image:
            self.head_keypoint = _Head(width, 4)

    def encode_image(self, features) -> torch.Tensor:
        """Every block's image keys and values (depth, B, _LATENTS, 2W) from encoder features (B, 1 + patches, D)."""
        with torch.autocast("cuda", torch.bfloat16):
            position = F.pad(self.image_pos_emb, (0, 0, 1, 0))  # the summary (CLS) token gets none
            latents = self.perceiver(self.perceiver.feature_proj(features) + position)
            return torch.stack([F.linear(latents, weight) for weight in self.blocks.image_kv_w])

    def forward(
        self, trans, quat, types, t, split, materials=None, features=None, image_kv=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Velocity (B, N, 3), body-frame angular velocity (B, N, 3), and IC keypoints (B, N, 4): projected center of
        mass (x, y) and its log-variance (None for PT models). Parts are at scaled centers of mass trans (B, N, 3), quat
        (B, N, 4) with types (B, N) at t (B,); split (B,) is each row's first sample size."""
        n = types.shape[1]
        rotation = quat_to_matrix(quat)
        first_two_columns = torch.cat([rotation[..., 0], rotation[..., 1]], -1)
        x = self.type_proj(self.type_emb(types)) + self.trans_proj(trans) + self.rot_proj(first_two_columns)
        parts = torch.where(second_sample(split, n), n - split[:, None], split[:, None])
        x = x + self.size_emb(parts / MAX_PARTS)
        if self.image:
            x = x + self.material_proj(self.material_emb(materials))
            if image_kv is None:
                image_kv = self.encode_image(features)
        c = F.silu(self.time_emb(t))
        with torch.autocast("cuda", torch.bfloat16):
            x, keypoint_input = self.blocks(x, c, split, image_kv)
        keypoint = self.head_keypoint(keypoint_input, c) if self.image else None
        return self.head_trans(x, c), self.head_rot(x, c), keypoint

    def muon_parameters(self) -> list[nn.Parameter]:
        """Hidden matrices: the block weights and the Perceiver layers' 2-D weights."""
        params = [p for name, p in self.blocks.named_parameters() if name.endswith("_w")]
        if self.image:
            params += [p for p in self.perceiver.layers.parameters() if p.ndim == 2]
        return params


def load_model(path: str | Path, device: str | torch.device = "cpu") -> Denoiser:
    """Denoiser from a weights file (path or release name), sized from its shapes, in eval mode without gradients."""
    state = torch.load(find(path), map_location="cpu", weights_only=True)["model"]
    width = state["trans_proj.weight"].shape[0]
    depth = state["blocks.qkv_w"].shape[0]
    image = "perceiver.queries" in state
    tokens = ()
    if image:
        tokens = (state["perceiver.feature_proj.weight"].shape[1], len(state["image_pos_emb"]))
    model = Denoiser(width, depth, image, *tokens)
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False)


def representative_rows(state: dict) -> dict:
    """Weights for full catalog ids: each id takes its physical-class representative's type embedding."""
    embedding = state["type_emb.weight"].clone()
    embedding[:PART_TYPES] = embedding[torch.as_tensor(canonical_id(np.arange(PART_TYPES)))]
    return dict(state, **{"type_emb.weight": embedding})
