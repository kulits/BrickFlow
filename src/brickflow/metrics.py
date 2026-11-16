"""Mass movement, connectivity, and collision metrics."""

import math
from dataclasses import dataclass

import bricknet
import numpy as np
from bricknet import collision
from procbrick import canonical_id

from .parts import _arrays, _parsed, table

_DEPTH = 1.1  # LDU: the paper's collision allowance (Valid, Coll.)


@dataclass(frozen=True)
class Movement:
    """Mass movement, its bound, per-part displacement, matching, and rigid motion; see DATA.md."""

    rms: float
    bound: float
    certified: bool
    trans: float
    rot: float
    per_part: np.ndarray
    order: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def mass_movement(
    assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray],
    target: bricknet.Graph | tuple[np.ndarray, np.ndarray],
    ic: bool = False,
    certify: float = 1e-4,
) -> Movement:
    """Mass movement onto a target with the same parts; IC matches part ids and colors, PT matches representatives."""
    part_ids, poses, colors = _arrays(assembly)
    target_ids, target_poses, target_colors = _arrays(target)
    if ic:
        keys, target_keys = part_ids << 32 | colors, target_ids << 32 | target_colors
    else:
        part_ids, target_ids = canonical_id(part_ids), canonical_id(target_ids)
        keys, target_keys = part_ids, target_ids
    if sorted(zip(keys.tolist(), part_ids.tolist())) != sorted(zip(target_keys.tolist(), target_ids.tolist())):
        raise ValueError("the assembly and the target must hold the same parts")
    matched = np.empty(len(keys), np.int64)  # the target part of each part: equal keys in order
    matched[np.argsort(keys, kind="stable")] = np.argsort(target_keys, kind="stable")
    result = table(ic)._movement(part_ids, keys, poses, target_poses[matched], certify)
    total = len(part_ids)
    trans2, rot2 = result["trans2"], result["rot2"]
    return Movement(
        rms=math.sqrt(result["cost"] / total),
        bound=math.sqrt(max(result["bound"], 0.0) / total),
        certified=bool(result["certified"]),
        trans=math.sqrt(trans2.sum() / total),
        rot=math.sqrt(rot2.sum() / total),
        per_part=np.sqrt(trans2 + rot2),
        order=matched[result["order"]],
        rotation=result["rotation"],
        translation=result["translation"],
    )


def connections(assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray]) -> bricknet.Graph:
    """The paper's connectivity: BrickNet's parse (`bricknet.Graph`) of each part's physical-class representative at its
    pose."""
    part_ids, poses, _ = _arrays(assembly)
    part_ids = canonical_id(part_ids).tolist()
    return _parsed(part_ids, [16] * len(part_ids), poses)


def _penetrations(assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray]) -> tuple:
    """Connectivity and depths (i, j, depth), i < j, of BrickNet's candidates not FIXED-joined (None: no collider)."""
    graph = connections(assembly)
    ids, poses = list(graph.part_ids), graph.transforms
    candidates = collision.colliding_pairs(ids, poses)
    if candidates is None:
        return graph, None
    fixed = graph.edges[graph.edges["family"] == "fixed"]
    joined = set(zip(np.minimum(fixed["a"], fixed["b"]).tolist(), np.maximum(fixed["a"], fixed["b"]).tolist()))
    pairs = []
    for i, j in candidates:
        if (i, j) not in joined:
            pairs.append((i, j, collision.penetration(ids[i], poses[i], ids[j], poses[j])))
    return graph, pairs


def collisions(
    assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray], depth: float = _DEPTH
) -> list[tuple[int, int, float]] | None:
    """Pairs (i, j, depth) penetrating deeper than `depth` (>= 1.0) LDU, except FIXED-joined; None: no collider."""
    if depth < 1.0:
        raise ValueError(f"depth {depth} is below 1.0 LDU: BrickNet proposes only pairs deeper than 1.0 LDU")
    _, pairs = _penetrations(assembly)
    return None if pairs is None else [pair for pair in pairs if pair[2] > depth]


def score(
    assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray],
    target: bricknet.Graph | tuple[np.ndarray, np.ndarray] | None = None,
    ic: bool = False,
    certify: float = 1e-4,
) -> dict:
    """valid, single, components, max_component, and collisions (NaN without a collider); with a target also mm,
    mm_trans, and mm_rot (`ic`: IC protocol)."""
    graph, pairs = _penetrations(assembly)
    sizes = [len(component) for component in graph.components]
    deep = np.nan if pairs is None else sum(depth > _DEPTH for _, _, depth in pairs)
    out = dict(
        valid=len(sizes) == 1 and deep == 0,
        single=len(sizes) == 1,
        components=len(sizes),
        max_component=max(sizes, default=0),
        collisions=deep,
    )
    if target is not None:
        movement = mass_movement(assembly, target, ic, certify)
        out.update(mm=movement.rms, mm_trans=movement.trans, mm_rot=movement.rot)
    return out


def bootstrap(values: np.ndarray, assemblies: np.ndarray, n: int = 1000) -> tuple[float, float, float]:
    """Mean (NaN left out) and 95% interval from `n` resamples of the assemblies, each with all its samples."""
    values = np.asarray(values, np.float64)
    ids, members = np.unique(assemblies, return_inverse=True)
    finite = np.isfinite(values)
    sums = np.bincount(members, np.where(finite, values, 0.0), len(ids))
    counts = np.bincount(members, finite.astype(np.float64), len(ids))
    weights = np.random.default_rng().multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=n)
    weights = weights.astype(np.float64)
    means = (weights @ sums) / (weights @ counts)
    return float(sums.sum() / counts.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _score_job(job: tuple) -> dict:
    """Metrics of one sample, with its MM onto its target (NaN if the two hold different parts or one the part table
    lacks)."""
    graph, target, ic, row = job
    try:
        return dict(score(graph, target, ic), row=row)
    except (KeyError, ValueError):
        return dict(score(graph), row=row)
