"""Training rows from ProcBrick trees and renders."""

import itertools
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import procbrick
import torch
from PIL import AvifImagePlugin, Image

from . import _core
from .flow import COORD_SCALE, noise_scale
from .model import MAX_PARTS, UNKNOWN_MATERIAL, material_ids
from .parts import PartTable, by_type
from .paths import find

_MATERIAL_DROP = 0.25
_IMAGE_DROP = 0.1
_IMAGE = np.array([[0.5, 0.0, 0.5], [0.0, -0.5, 0.5], [0.0, 0.0, 1.0]])  # clip (x, y, w) -> image (x, y, 1) w
_START = "fork" if sys.platform == "linux" else "spawn"  # data workers inherit the dataset where fork is safe
AvifImagePlugin.DEFAULT_MAX_THREADS = 1  # one decoder thread per data worker process


def loss_metric(table: PartTable, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass and inertia I = tr(S) 1 - S about the center of mass by type id, for the loss."""
    mass, _, second, *_ = by_type(table)
    inertia = (np.trace(second, axis1=1, axis2=2)[:, None, None] * np.eye(3) - second) * COORD_SCALE**2
    mass = torch.tensor(mass, dtype=torch.float32, device=device)
    return mass, torch.tensor(inertia, dtype=torch.float32, device=device)


def _pairs(
    table: PartTable,
    seed: int,
    t_mean: float,
    assemblies=None,
    assets="assets",
    parts: int = MAX_PARTS,
    threads: int = 4,
) -> _core.Pairs:
    """Coupling with t ~ logit-N(t_mean, 1.5^2)."""
    noise = [noise_scale(max(k, 1)) for k in range(parts + 1)]
    if assemblies is not None:
        assemblies = (find(assemblies), find(assets), procbrick.representatives().tolist())
    return _core.Pairs(assemblies, table._columns, seed, COORD_SCALE, noise, (t_mean, 1.5), threads)


def couple(table: PartTable, part_ids, poses, t: float) -> dict:
    """One assembly coupled to noise as in training, at time t: per slot its part's id (`types`, in `order`), the
    model's inputs `trans_t` and `quat_t`, and the targets `trans_vel` and `rot_tangent` (Log(q_t^-1 q1))."""
    part_ids = np.asarray(part_ids, np.int64)
    pairs = _pairs(table, int(torch.randint(2**62, ())), 0.0, parts=len(part_ids))
    _, fields, order = pairs.couple(0, part_ids, part_ids, np.asarray(poses, np.float64), t)
    return dict(fields, types=part_ids[order], order=np.asarray(order))


class TreeRows(torch.utils.data.Dataset):
    """Pretraining rows: row r holds the samples at stream positions 2r and 2r + 1. The coupling is made in the data
    worker process."""

    def __init__(self, root: str | Path, assets: str | Path, table: PartTable, seed: int, threads: int = 4) -> None:
        self.root = root
        self.assets = assets
        self.table = table
        self.seed = seed
        self.threads = threads
        self.pairs = None

    def __getitems__(self, rows) -> dict:
        if self.pairs is None:
            self.pairs = _pairs(self.table, self.seed, 1.2, self.root, self.assets, threads=self.threads)
        rows = np.asarray(rows, np.uint64)
        positions = np.stack([2 * rows, 2 * rows + 1], -1).ravel()
        return {key: torch.from_numpy(value) for key, value in self.pairs.batch(positions).items()}


def _camera(camera, poses, types, com):
    """Intrinsics, image direction of world-down, and projected centers of mass (NaN-padded) of a render through
    `camera` (`Renders.camera`)."""
    camera_world, projection, ldraw_to_world = camera
    k = _IMAGE @ projection[[0, 1, 3], :3]
    to_camera = np.linalg.inv(camera_world) @ ldraw_to_world
    origin = k @ to_camera[:3, 3]
    down = k @ to_camera[:3, 1]
    gravity = down[:2] * origin[2] - origin[:2] * down[2]  # image motion of the assembly origin along +y, up to scale
    centers = poses[:, :3, 3] + np.einsum("kij,kj->ki", poses[:, :3, :3], com[types])
    clip = projection @ to_camera @ np.c_[centers, np.ones(len(centers))].T
    projected = _IMAGE @ clip[[0, 1, 3]]
    points = np.full((MAX_PARTS, 2), np.nan, np.float32)
    points[: len(centers)] = (projected[:2] / projected[2]).T
    return k.astype(np.float32), (gravity / np.linalg.norm(gravity)).astype(np.float32), points


class RenderRows(torch.utils.data.Dataset):
    """Finetuning rows: a rendered assembly and its partner with renders and cameras. A sample's materials are unknown
    with probability _MATERIAL_DROP (identical parts: those of equal type) and its image is dropped with _IMAGE_DROP
    (the classifier-free guidance's null condition). The coupling is made in the data worker process."""

    def __init__(
        self,
        root: str | Path,
        assets: str | Path,
        table: PartTable,
        backgrounds: list[str],
        seed: int,
        threads: int = 4,
    ) -> None:
        self.renders = procbrick.load_renders(root, assets)
        self.assets = assets
        if not self.renders.paired():
            raise ValueError(f"{root} holds renders without partners: finetuning reads every render with its partner")
        self.table = table
        _, self.com, *_ = by_type(table)
        self.backgrounds = backgrounds
        self.seed = seed
        self.threads = threads
        self.pairs = None

    def __getitems__(self, rows) -> dict:
        made = [self._row(i) for i in rows]
        batch = {key: torch.from_numpy(np.stack([row[key] for row in made])) for key in made[0] if key != "background"}
        batch["background"] = [torch.from_numpy(crop) for row in made for crop in row["background"]]
        return batch

    def _row(self, i: int) -> dict:
        """Row i: a rendered assembly and its partner, coupled at one t (drawn for the first) with interchangeable parts
        (equal type and material key) matched, with their images and cameras."""
        if self.pairs is None:
            self.pairs = _pairs(self.table, self.seed, 0.0, threads=self.threads)
        rng = np.random.default_rng([self.seed, i])
        render = self.pairs.shuffled(2 * i, len(self.renders))  # the span is even, so a pass meets each pair once
        t = None
        slots = []
        images = []
        crops = []
        for j, index in enumerate((render, self.renders.paired_index(render))):
            (tree,), poses = self.renders._decode([index])
            types = np.array([part.part_id for part in tree.parts], np.int64)
            materials = material_ids([part.color for part in tree.parts], self.assets)
            if rng.random() < _MATERIAL_DROP:
                materials[:] = UNKNOWN_MATERIAL
            keys = types * 65536 + materials
            t, pair, order = self.pairs.couple(2 * i + j, types, keys, poses, t)
            slots.append(dict(pair, types=types[order], materials=materials[order], original=np.asarray(order)))
            k, gravity, points = _camera(self.renders.camera(index), poses, types, self.com)
            crop, flip = self._background(rng) if self.backgrounds else (None, np.bool_(False))
            if crop is not None:
                crops.append(crop)
            dropped = np.bool_(rng.random() < _IMAGE_DROP)
            image = self.renders.image(index)
            images.append(dict(image=image, flip=flip, k=k, gravity=gravity, points=points, image_drop=dropped))
        row = {key: np.concatenate([slot[key] for slot in slots]) for key in slots[0]}
        row.update({key: np.stack([image[key] for image in images]) for key in images[0]})
        lengths = np.array([len(slot["types"]) for slot in slots])
        return dict(row, t=np.float32(t), lengths=lengths, background=crops)

    def _background(self, rng) -> tuple[np.ndarray, np.bool_]:
        """Random square ImageNet crop (40--100% of the short side) and whether to flip it."""
        image = Image.open(self.backgrounds[rng.integers(len(self.backgrounds))]).convert("RGB")
        side = round(min(image.size) * (0.4 + 0.6 * rng.random()))
        left = rng.integers(image.width - side + 1)
        top = rng.integers(image.height - side + 1)
        crop = np.asarray(image.crop((left, top, left + side, top + side)))
        return crop, np.bool_(rng.random() < 0.5)


def _rank_rows(rows: int, rank: int, world: int, start: int) -> Iterator[int]:
    """This rank's row indices from step `start` on."""
    for step in itertools.count(start):
        first = (step * world + rank) * rows
        yield from range(first, first + rows)


def _whole(chunk):
    return chunk


def batches(
    dataset: torch.utils.data.Dataset,
    rows: int,
    chunk: int,
    workers: int,
    rank: int,
    world: int,
    start: int,
    device: str | torch.device,
) -> Iterator[dict]:
    """This rank's rows of each step from `start` on, made by worker processes in chunks of `chunk` rows (one
    __getitems__ call each), on the device."""
    sampler = _rank_rows(rows, rank, world, start)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=chunk,
        sampler=sampler,
        num_workers=workers,
        collate_fn=_whole,
        pin_memory=True,
        multiprocessing_context=_START if workers else None,
    )
    pieces = iter(loader)
    while loaded := list(itertools.islice(pieces, rows // chunk)):
        batch = {}
        for key, value in loaded[0].items():
            if torch.is_tensor(value):
                batch[key] = torch.cat([piece[key].to(device, non_blocking=True) for piece in loaded])
            else:
                batch[key] = [x.to(device, non_blocking=True) for piece in loaded for x in piece[key]]
        yield batch
