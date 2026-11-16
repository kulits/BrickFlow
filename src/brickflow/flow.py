"""Flow matching and sampling on part poses."""

import dataclasses
import functools
import math
from collections.abc import Callable, Generator
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from procbrick import canonical_id

from .connect import Workers
from .model import (
    MAX_PARTS,
    Denoiser,
    compiled,
    quat_multiply,
    quat_to_matrix,
    quat_to_rotvec,
    rotvec_to_quat,
    second_sample,
)
from .parts import by_type, table

COORD_SCALE = 1 / 56.6  # 1 / the spread of assembly positions, LDU
_NOISE_FIT = (-1.3701, 0.4587, -0.0332)  # log sigma(k) = c0 + c1 log k + c2 log^2 k: the spread of k-part prefixes
_TRANS_CLAMP = 5.0
_ROT_T_CLIP = 0.9
_KEYPOINT_WEIGHT = 10.0
_BUCKET = 8


@dataclasses.dataclass(frozen=True)
class Settings:
    """Sampler settings; the defaults are the paper's."""

    schedule: tuple[float, ...] = tuple(i / 50 for i in range(51))  # increasing t from 0 to 1: one Heun step each
    min_t: float = 0.01  # the field is evaluated at max(t, min_t)
    rot_rate: float = 25.0  # rotations turn min(rot_rate dt, 1) of the way to the predicted endpoint per step
    cfg_weight: float = 2.0  # IC: classifier-free guidance weight of the image
    handoff: float = 0.85  # IC: from t = handoff on, the PT model (`pt`), else the IC model without the image
    guidance_start: float | None = None  # guidance from t >= guidance_start (None: 0.7 for PT models, IC the handoff)
    gamma: float = 2.0  # guidance step size
    clip: float = 1.0  # guidance step cap per part, LDU in the metric sqrt(|d_v|^2 + r2 |d_w|^2)
    seed: int | torch.Generator | None = None

    def __post_init__(self):
        grid = self.schedule
        if len(grid) < 2 or grid[0] != 0 or grid[-1] != 1 or any(b <= a for a, b in zip(grid, grid[1:])):
            raise ValueError("the schedule must increase from t = 0 to t = 1")

    def start(self, image: bool) -> float:
        """t from which the guidance acts: `guidance_start`, else 0.7 for PT models and the handoff for IC models."""
        if self.guidance_start is not None:
            return self.guidance_start
        return self.handoff if image else 0.7


def loss(out, batch, mass, inertia) -> tuple[torch.Tensor, dict]:
    """Packed-row velocity loss and its translation, rotation, and optional keypoint terms."""
    trans_vel, omega, keypoint = out
    types = batch["types"]
    t = batch["t"][:, None]
    parts = batch["lengths"].float()
    second = second_sample(batch["lengths"][:, 0], types.shape[1])
    sample_of_part = torch.stack([~second, second], -1).float()

    def sample_sums(per_part):
        return torch.einsum("bn,bns->bs", per_part, sample_of_part)

    def weighted_mean(per_sample, weight):
        return (per_sample * weight).sum() / weight.sum().clamp(min=1)

    trans_error = (trans_vel - batch["trans_vel"]).square().sum(-1)
    trans = (sample_sums(trans_error) / (3 * parts)).clamp(max=_TRANS_CLAMP)
    s = (1 - t).clamp(min=1e-6)
    omega_error = omega - batch["rot_tangent"] / s[..., None]
    metric = inertia[types] / mass[types].clamp(min=1e-12)[..., None, None]
    rot_error = torch.einsum("bni,bnij,bnj->bn", omega_error, metric, omega_error)
    rot_error = rot_error * (s / (1 - t.clamp(max=_ROT_T_CLIP))).square()
    rot = sample_sums(rot_error) / (3 * parts)
    logs = dict(trans=weighted_mean(trans, parts), rot=weighted_mean(rot, parts))
    total = logs["trans"] + logs["rot"]
    if keypoint is not None:
        target = batch["keypoints"]
        real = target.isfinite().all(-1)
        log_var = keypoint[..., 2:].clamp(-7, 7)
        error = torch.where(real[..., None], target - keypoint[..., :2], 0)
        nll = 0.5 * (error.square() / log_var.exp() + log_var) * log_var.detach().exp()
        per_sample = sample_sums(nll.sum(-1) * real) / parts
        has_image = sample_sums(real.float()) > 0
        logs["keypoint"] = weighted_mean(per_sample, parts * has_image)
        total = total + _KEYPOINT_WEIGHT * logs["keypoint"]
    return total, logs


def padded(sizes) -> np.ndarray:
    """Part counts padded to the sampler's buckets, multiples of _BUCKET (at most MAX_PARTS, unless an assembly has
    more parts)."""
    sizes = np.asarray(sizes, np.int64)
    return np.maximum(np.minimum(-(-sizes // _BUCKET) * _BUCKET, MAX_PARTS), sizes)


def noise_scale(k: int) -> float:
    """Translation noise std of a k-part assembly (scaled units)."""
    log_k = math.log(k)
    return math.exp(_NOISE_FIT[0] + _NOISE_FIT[1] * log_k + _NOISE_FIT[2] * log_k**2)


def _remove_weighted_mean(x, weights) -> torch.Tensor:
    """x (B, N, 3) minus its weighted centroid."""
    centroid = (weights[..., None] * x).sum(-2, keepdim=True) / weights.sum(-1)[:, None, None]
    return x - centroid


def initial_noise(n, weights, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial noise of each sample: centered translations (noise_scale std) and Haar-random rotations."""
    real = weights > 0
    scale = torch.tensor([noise_scale(k) for k in n.tolist()], device=weights.device)[:, None, None]
    trans = torch.randn(*weights.shape, 3, device=weights.device, generator=generator) * scale * real[..., None]
    quat = F.normalize(torch.randn(*weights.shape, 4, device=weights.device, generator=generator), dim=-1)
    quat = torch.where(real[..., None], torch.where(quat[..., :1] < 0, -quat, quat), quat.new_tensor([1.0, 0, 0, 0]))
    return _remove_weighted_mean(trans, weights), quat


def ldraw_pose(x, q, com) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 rotations and LDraw origins (LDU) of parts at centers of mass x; com in the parts' frames (LDU)."""
    rotation = quat_to_matrix(F.normalize(q.double(), dim=-1))
    origin = x.double() / COORD_SCALE - (rotation @ com.double()[..., None])[..., 0]
    return rotation, origin


class Guidance(NamedTuple):
    """Guidance from t >= start: energy(R0, o0) -> wait, the parts' centers of mass and squared gyration radii."""

    start: float
    energy: Callable
    com: torch.Tensor
    r2: torch.Tensor


def _heun_step(field, x, q, t, t_next, weights, v, turn, settings: Settings):
    """(x, q) after a Heun step from t to t_next given the field at t; rotations turn min(rot_rate dt, 1) of the way
    to the predicted endpoint, all of it on the last step."""
    dt = t_next - t
    last = t_next == 1
    fraction = dt / (1 - t) if last else min(settings.rot_rate * dt, 1.0)
    q_next = quat_multiply(q, rotvec_to_quat(fraction * quat_to_rotvec(turn)))
    x_euler = x + dt * _remove_weighted_mean(v, weights)
    if last:
        return x_euler, q_next
    v_next, _ = field(x_euler, q_next, t_next)
    return x + dt * _remove_weighted_mean((v + v_next) / 2, weights), q_next


def _guided_step(
    field, x, q, t, t_next, weights, guidance: Guidance, settings: Settings
) -> Generator[Callable, None, tuple[torch.Tensor, torch.Tensor]]:
    """Heun step with endpoint guidance; yields the energy wait and returns updated poses."""
    delta = torch.zeros(*x.shape[:-1], 6, device=x.device, requires_grad=True)
    with torch.enable_grad():
        x_perturbed = x + COORD_SCALE * delta[..., 3:]
        q_perturbed = q + 0.5 * quat_multiply(F.pad(delta[..., :3], (1, 0)), q)
        v, turn = field(x_perturbed, q_perturbed, max(t, settings.min_t))
        x1 = x_perturbed + (1 - t) * v
        q1 = quat_multiply(q_perturbed, turn)
        rotation, origin = ldraw_pose(x1, q1, guidance.com)
    R0 = rotation.detach()
    o0 = origin.detach()
    wait = guidance.energy(R0.cpu().numpy(), o0.cpu().numpy())
    x, q = _heun_step(field, x, q, t, t_next, weights, v.detach(), turn.detach(), settings)
    yield wait
    _, G = wait()
    G = torch.from_numpy(G).to(R0.device)
    with torch.enable_grad():
        q0 = F.normalize(q1.detach().double(), dim=-1)
        q0[..., 1:] = -q0[..., 1:]
        relative = quat_multiply(q0, F.normalize(q1.double(), dim=-1))
        twist = torch.cat([2 * relative[..., 1:], (R0.mT @ (origin - o0)[..., None])[..., 0]], -1)
        (g,) = torch.autograd.grad((G * twist).sum(), delta)
    g = g.double()
    d_w = -settings.gamma * g[..., :3] / guidance.r2
    d_v = -settings.gamma * g[..., 3:]
    size = (d_v.square().sum(-1, keepdim=True) + guidance.r2 * d_w.square().sum(-1, keepdim=True)).sqrt()
    shrink = (settings.clip / size.clamp(min=1e-12)).clamp(max=1)
    d_v = _remove_weighted_mean(d_v * shrink, weights.double())
    d = torch.cat([d_w * shrink, d_v], -1).float()
    return x + COORD_SCALE * d[..., 3:], quat_multiply(rotvec_to_quat(d[..., :3]), q)


@torch.no_grad()
def sample(
    field,
    x,
    q,
    weights,
    guidance: Guidance | None = None,
    settings: Settings = Settings(),
    on_step: Callable[[float, torch.Tensor, torch.Tensor], None] | None = None,
) -> Generator[Callable | None, None, tuple[torch.Tensor, torch.Tensor]]:
    """A Heun step from noise per interval of `settings.schedule` (the field at max(t, min_t)), guided from
    guidance.start; with guidance, yields per step (the wait, or None). `on_step(t, x, q)` observes each state."""
    grid = settings.schedule
    if on_step is not None:
        on_step(grid[0], x, q)
    for t, t_next in zip(grid, grid[1:]):
        if guidance is not None and t >= guidance.start:
            x, q = yield from _guided_step(field, x, q, t, t_next, weights, guidance, settings)
        else:
            v, turn = field(x, q, max(t, settings.min_t))
            x, q = _heun_step(field, x, q, t, t_next, weights, v, turn, settings)
            if guidance is not None:
                yield None
        if on_step is not None:
            on_step(t_next, x, q)
    return x, q


@functools.cache
def _graphed(model: Denoiser) -> Callable:
    """Model compiled once for every batch shape, as replayed graphs without gradients (a replay overwrites the
    previous outputs)."""
    # a graph per side of the kernel's 128-token limit, per model (IC: two), and per grad mode (DPS)
    torch._dynamo.config.recompile_limit = 32
    graphs = compiled(model, mode="reduce-overhead", dynamic=True)
    plain = compiled(model, dynamic=True)

    def call(*args, **kwargs):
        return (plain if torch.is_grad_enabled() else graphs)(*args, **kwargs)

    return call


@functools.cache
def _columns(full: bool, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass, center of mass, and squared gyration radius by type id, on the device."""
    mass, com, second, *_ = by_type(table(full))
    r2 = np.maximum(np.trace(second, axis1=1, axis2=2) / np.where(mass > 0, mass, np.inf), 1.0)
    return (
        torch.as_tensor(mass, dtype=torch.float32, device=device),
        torch.as_tensor(com, dtype=torch.float32, device=device),
        torch.as_tensor(r2, device=device),
    )


@functools.cache
def null_image_kv(model: Denoiser) -> torch.Tensor:
    """IC model's image keys and values of no image (one image, zero features): the classifier-free guidance's null
    condition."""
    patches, width = model.image_pos_emb.shape[0], model.perceiver.feature_proj.in_features
    zeros = torch.zeros(1, 1 + patches, width, device=model.type_emb.weight.device)
    return model.encode_image(zeros)


def generate(
    model: Denoiser,
    part_ids,
    samples: int,
    guidance: Workers | None = None,
    materials=None,
    image_kv: torch.Tensor | None = None,
    pt: Denoiser | None = None,
    settings: Settings = Settings(),
    on_step: Callable[[float, np.ndarray], None] | None = None,
) -> Generator[Callable | None, None, np.ndarray]:
    """Sampling steps returning quaternions and origins (inputs, samples, padded parts, 7).
    Guided steps yield energy waits; `run` drives them to completion. `on_step(t, poses)` receives each state."""
    sizes = np.array([len(ids) for ids in part_ids], np.int64)
    if sizes.min() < 1:
        raise ValueError("an assembly without parts")
    torch.backends.cuda.matmul.allow_tf32 = model.image
    device = model.type_emb.weight.device
    parts = int(padded(sizes.max()))
    real = np.arange(parts) < sizes[:, None]
    catalog = np.zeros((len(sizes), parts), np.int64)
    catalog[real] = np.concatenate(part_ids)
    ids = canonical_id(catalog).astype(np.uint32)
    table(model.image).rows((catalog if model.image else ids)[real])
    assembly = np.repeat(np.arange(len(sizes)), samples)
    n = sizes[assembly]
    mass, com, _ = _columns(model.image, device)
    types = torch.as_tensor((catalog if model.image else ids.astype(np.int64))[assembly], device=device)
    pt_types = torch.as_tensor(ids[assembly].astype(np.int64), device=device)
    split = torch.as_tensor(n, device=device)  # padding attends only to padding
    weights = mass[types] * torch.as_tensor(real[assembly], device=device)
    generator = settings.seed
    if isinstance(generator, int):
        generator = torch.Generator(device).manual_seed(generator)
    x, q = initial_noise(n, weights, generator)
    graphed = _graphed(model)
    graphed_pt = graphed if pt is None else _graphed(pt)
    handoff = 0.0  # PT models: the model itself throughout
    if model.image:
        handoff = settings.handoff
        material_ids = np.zeros((len(sizes), parts), np.int64)
        material_ids[real] = np.concatenate(materials)
        material_ids = torch.as_tensor(material_ids[assembly], device=device)
        image_kv = image_kv.repeat_interleave(samples, 1)
        null_kv = null_image_kv(model).expand(-1, len(assembly), -1, -1).contiguous()

    def field(x, q, t):
        """Velocity and the turn from q_t to the predicted endpoint; IC: classifier-free guidance with the image
        (`settings.cfg_weight`) before the handoff."""
        times = torch.full((len(x),), t, device=x.device)
        if t >= handoff:
            if pt is None and model.image:
                v, omega, _ = graphed(x, q, types, times, split, material_ids, image_kv=null_kv)
            else:
                v, omega, _ = graphed_pt(x, q, pt_types, times, split)
            v = v.clone()
        else:
            seen = [out.clone() for out in graphed(x, q, types, times, split, material_ids, image_kv=image_kv)[:2]]
            unseen = graphed(x, q, types, times, split, material_ids, image_kv=null_kv)[:2]
            v, omega = (blind + settings.cfg_weight * (sighted - blind) for sighted, blind in zip(seen, unseen))
        return v, rotvec_to_quat((1 - times)[:, None, None] * omega)

    guided = None
    if guidance is not None:
        r2 = _columns(False, device)[2][pt_types][..., None]
        guided = Guidance(settings.start(model.image), guidance.energy(ids[assembly], n, assembly), com[types], r2)

    def placed(x, q):
        _, origin = ldraw_pose(x, q, com[types])
        return torch.cat([q, origin.float()], -1).cpu().numpy().reshape(len(sizes), samples, parts, 7)

    def observe(t: float, x, q) -> None:
        on_step(t, placed(x, q))

    x, q = yield from sample(field, x, q, weights, guided, settings, observe if on_step is not None else None)
    return placed(x, q)


def run(steps) -> np.ndarray:
    """Result of a sampling generator (`generate`), driven to the end; each guided step waits for its energy."""
    try:
        while True:
            next(steps)
    except StopIteration as finished:
        return finished.value
