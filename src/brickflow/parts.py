"""Part tables, symmetries, and pose conversions."""

import dataclasses
from functools import cache, cached_property
from importlib.resources import files

import bricknet
import numpy as np
from bricknet.data import load_aliases, load_catalog
from procbrick import class_ids, representatives

from . import _core

TYPE_CLASSES = 14592  # the models' type ids: catalog part ids, padded to a multiple of 16


def part_id(name: str) -> int:
    """Part id of a BrickNet stem, description, or same-frame alias ("3001", "brick 2x4", "3023"); KeyError if unknown
    or an alias in another frame."""
    catalog = load_catalog()
    key = name.strip().lower()
    stem = key.removesuffix(".dat")
    if stem in catalog.stem_to_id:
        return catalog.stem_to_id[stem]
    if key in catalog.name_to_id:
        return catalog.name_to_id[key]
    target, wrap = load_aliases().get(stem, (stem, np.eye(4)))
    if not np.array_equal(wrap, np.eye(4)):
        raise KeyError(f"{name!r} is an alias of {target!r} in another frame")
    return catalog.stem_to_id[target]


def pose_matrices(quat, origin) -> np.ndarray:
    """Poses (..., 4, 4) of quaternions (..., 4), normalized in float64, and origins (..., 3)."""
    q = np.asarray(quat, np.float64)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    pose = np.zeros((*q.shape[:-1], 4, 4))
    pose[..., 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    pose[..., 1, :3] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    pose[..., 2, :3] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    pose[..., :3, 3] = origin
    pose[..., 3, 3] = 1
    return pose


class PartTable:
    """Mass properties and symmetries: all represented parts, or their representatives with full=False; see DATA.md."""

    def __init__(self, full: bool = True) -> None:
        with (files("brickflow") / "_data" / "part_table.npz").open("rb") as stream:
            table = dict(np.load(stream))
        printed = table.pop("printed")
        if full:
            representative = representatives()[class_ids()]
            ids = np.flatnonzero(representative >= 0)
            rows = np.full(len(representative), -1, np.int64)
            rows[table["ids"]] = np.arange(len(table["ids"]))
            table = {name: values[rows[representative[ids]]] for name, values in table.items()}
            table["ids"] = ids
            printed = np.searchsorted(ids, printed)
            table["orbit"][printed] = [1.0, 0.0, 0.0, 0.0]
            table["orbit_size"][printed] = 1
            table["axis"][printed] = 0.0
        self.ids = table["ids"]
        self.mass = table["mass"]
        self.com = table["com"]
        self.second = table["second"]
        self.orbit = table["orbit"]
        self.orbit_size = table["orbit_size"]
        self.axis = table["axis"]
        self._row = {part_id: row for row, part_id in enumerate(self.ids.tolist())}

    @property
    def _columns(self) -> tuple[np.ndarray, ...]:
        return self.ids, self.mass, self.com, self.second, self.orbit, self.orbit_size, self.axis

    @cached_property
    def names(self) -> np.ndarray:
        return np.array(load_catalog().id_to_stem)[self.ids]

    def rows(self, part_ids) -> np.ndarray:
        """Row of each part id; KeyError for a part id the table lacks."""
        try:
            return np.array([self._row[part_id] for part_id in np.asarray(part_ids, np.int64).tolist()], np.int64)
        except KeyError as error:
            raise KeyError(f"part id {error.args[0]} is not in this part table of {len(self.ids)} parts") from None

    def symmetries(self, part_id: int) -> np.ndarray:
        """Symmetries of the part as unit quaternions (w, x, y, z) (k, 4), rotations about its center of mass; the first
        is the identity."""
        (row,) = self.rows([part_id])
        return self.orbit[row, : self.orbit_size[row]]

    def _movement(self, part_ids, keys, poses, target_poses, certify: float) -> dict:
        """MM of parts onto target poses by the extension, matched within equal keys (see `mass_movement`)."""
        return self._parts.movement(self.rows(part_ids), keys, poses, target_poses, certify)

    def distance(self, part_ids, poses_a, poses_b) -> np.ndarray:
        """Per part, the RMS displacement of its material (LDU) between two poses, least over its symmetries."""
        poses_a, poses_b = np.asarray(poses_a, np.float64), np.asarray(poses_b, np.float64)
        squared = self._parts.distance2(self.rows(part_ids), poses_a, poses_b)
        return np.sqrt(np.maximum(squared, 0.0))  # zero, not a rounding error below it, at an equivalent pose

    @cached_property
    def _parts(self) -> "_core.Parts":
        return _core.Parts(self._columns)


@cache
def table(full: bool = True) -> PartTable:
    """Bundled PartTable(full), read once."""
    return PartTable(full)


def by_type(part_table: PartTable) -> tuple[np.ndarray, ...]:
    """Columns of a part table but `ids`, by type id (TYPE_CLASSES rows); absent types have mass 0 and no symmetry."""
    identity = np.zeros(part_table.orbit.shape[1:])
    identity[0, 0] = 1
    arrays = []
    for values, absent in zip(part_table._columns[1:], (0.0, 0.0, 0.0, identity, 1.0, 0.0)):
        by_id = np.full((TYPE_CLASSES, *values.shape[1:]), absent)
        by_id[part_table.ids] = values
        arrays.append(by_id)
    return tuple(arrays)


def _arrays(assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Part ids, poses, and colors from a BrickNet container or a (part_ids, poses) pair."""
    if isinstance(assembly, bricknet.Graph):
        part_ids, poses, colors = assembly.part_ids, assembly.transforms, assembly.colors
        if poses is None:
            if len(assembly.components) > 1:
                raise ValueError(f"parts without poses in {len(assembly.components)} components have no shared frame")
            poses = bricknet.decode_graph(assembly)
    else:
        part_ids, poses = assembly
        colors = np.full(len(part_ids), 16)
    part_ids, poses = np.asarray(part_ids, np.int64), np.asarray(poses, np.float64).reshape(-1, 4, 4)
    if len(poses) != len(part_ids):
        raise ValueError(f"an assembly of {len(part_ids)} parts has {len(poses)} poses")
    return part_ids, poses, np.asarray(colors, np.int64)


def _parsed(part_ids: list[int] | np.ndarray, colors: list[int] | np.ndarray, poses: np.ndarray) -> bricknet.Graph:
    """BrickNet's parse of parts and colors at poses, retaining the supplied poses."""
    stems = bricknet.load_catalog().id_to_stem
    lines = []
    for part_id, color, pose in zip(part_ids, colors, poses):
        values = (*pose[:3, 3], *pose[:3, :3].reshape(-1))
        lines.append(f"1 {color} " + " ".join(f"{v:.12g}" for v in values) + f" {stems[part_id]}.dat")
    return dataclasses.replace(bricknet.parse_ldr("\n".join(lines)), transforms=np.asarray(poses, np.float64))
