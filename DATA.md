# Data

## Ids, poses, and symmetries

- **Part ids** index the sorted catalog of 14,583 LDraw part names: they are BrickNet's catalog ids,
  `bricknet.load_catalog().id_to_stem[i]`. `PartTable.names` holds each row's stem. `brickflow.parts.part_id(name)`
  takes a stem (`3001`), a description (`brick 2x4`), or an alias that BrickNet's `parse_ldr` resolves (`3023` for
  `3023b`); an alias whose frame is not its part's (`bricknet.data.load_aliases`) raises a KeyError naming the part,
  since its poses change.
- **Physical classes** are ProcBrick's (`procbrick.class_ids`, `procbrick.representatives`): parts with identical
  colliders and connectors in the same local frame, 6,489 classes. Each class has a representative id (6,339
  representatives); 150 classes, holding 153 part ids, have none (-1). PT models, the atlas, the snap, connectivity, and
  collisions use representative ids (`procbrick.canonical_id`); IC models read part ids and colors.
- **Colors** are LDraw color codes. IC models read each color as its material in ProcBrick's appearance policy
  (`brickflow.model.material_ids`): material m has color `procbrick.load_appearance_policy()["colors"][m]`, and any
  other color is the unknown material, 260.
- **Model type ids** are part ids; the model's type embedding has 14,592 rows (14,583 padded to a multiple of 16).
- **Poses** are float64 (n, 4, 4) in LDU at each part's LDraw origin, LDraw axes (-y up), as in LDraw files. The
  models work on centers of mass in units of 56.6 LDU.
- **Quaternions** are (w, x, y, z), Hamilton, and act on the part frame: x_world = R(q) x_part + t. ProcBrick's
  `pose_dictionary` stores them as (x, y, z, w).
- **Symmetries** are rotations g about the part's center of mass c, applied on the right: the pose (R, o) is equivalent
  to (R g, o + R (c - g c)). About the LDraw origin, the same rotation moves any part whose origin is off its axis (by
  300 LDU for 2637, technic link 16l).

## Part table (`brickflow/_data/part_table.npz`, in the package)

The compressed npz file stores the physical representatives' mass properties and symmetries, and `printed`, the
int64 part ids with prints or stickers. `brickflow.PartTable()` expands the representatives through ProcBrick's
physical classes and removes printed parts' symmetries. `PartTable(full=False)` reads the representatives directly.

PT models and default MM use the 6,339 representatives in physical-class order. IC models and MM with `ic=True`
use the 14,430 represented parts, sorted by id; the other 153 are excluded (141 without geometry, 12 whose class
cannot connect).

| array | dtype, shape | contents |
| --- | --- | --- |
| `ids` | int64 (P,) | part id of each row |
| `mass` | float64 (P,) | volume at density 1, LDU^3 |
| `com` | float64 (P, 3) | center of mass in the LDraw part frame, LDU |
| `second` | float64 (P, 3, 3) | second moment of mass about the center of mass, int (x - c)(x - c)^T dV, LDU^5; MM's per-part matrix is `second / mass` (LDU^2) |
| `orbit` | float64 (P, 48, 4) | rotational symmetries, unit quaternions, each rotation once with either sign: rows `[0, orbit_size)` form a group whose first element is the identity; the other rows are the identity |
| `orbit_size` | int64 (P,) | number of symmetries |
| `axis` | float64 (P, 3) | unit axis of continuous rotational symmetry through the center of mass, along a part axis, or zero; `orbit` also lists the turns about it |

The IC rows equal their representatives' PT rows, except for 2,851 printed parts whose symmetries change: every
printed part gets only the identity and no axis. PT uses the shapes' symmetries, regardless of prints.

A symmetry is a rotation, among the cube rotations and the n-fold turns about the part axes, that maps the part's
watertight mesh and second moment onto themselves and each of its connectors onto a connector of the same kind with the
same direction. Examples:

- Connector directions count: axles and pins have no end-to-end flip (3704, technic axle 2: the quarter turns about x;
  3673, technic pin: the identity and a half turn about x), although their shapes have one.
- Round parts have the 16-fold symmetry of their tessellated meshes (6141, 3062b, 98138), not a continuous one. In
  `PartTable(full=False)` three parts have a continuous axis, 43898, 43898p03, and 97783; in `PartTable()`, 43898 and
  97783.
- The round plate 4032b with reduced axlehole has 2-fold mesh symmetry.

## Released files (`python -m brickflow fetch`)

The weights come from the Hugging Face model repo kulits/BrickFlow, at `brickflow.paths.REVISION`, into `brickflow/`
in BrickNet's data directory; ProcBrick fetches the rest from the dataset repo kulits/BrickFlow into `procbrick/`.
`fetch` defaults to `assets`, `val/procbrick`, and `weights/pt-l`; `--all` gets everything. `assets` also fetches
BrickNet's colliders through `python -m bricknet fetch-meshes`.

| name | repo | files | size | contents |
| --- | --- | --- | --- | --- |
| `assets` | dataset | 10 | 1.11 GiB | `assets/`: ProcBrick's assets with the connection atlas |
| `val/procbrick` | dataset | 207 | 1015 MiB | `val/procbrick/`: the evaluation set (below) |
| `train/pt` | dataset | 3 | 19.4 GiB | `train/pt/`: the pretraining set, a directory of 100,000,000 trees of 100 parts |
| `train/ic` | dataset | 15,651 | 100.5 GiB | `train/ic/`: the finetuning set, 1,000,000 renders with their images |
| `weights/pt-s` | model | 1 | 127 MiB | `weights/pt-s.pt` |
| `weights/pt-b` | model | 1 | 503 MiB | `weights/pt-b.pt` |
| `weights/pt-l` | model | 1 | 1.72 GiB | `weights/pt-l.pt` |
| `weights/ic-s` | model | 1 | 206 MiB | `weights/ic-s.pt` |
| `weights/ic-b` | model | 1 | 811 MiB | `weights/ic-b.pt` |
| `weights/ic-l` | model | 1 | 2.48 GiB | `weights/ic-l.pt` |

## ProcBrick's formats

ProcBrick owns the atlas, pose dictionary, assembly model, trees, and renders; see its DATA.md for their formats.

## Weights

`weights/pt-<size>.pt` and `weights/ic-<size>.pt` for the sizes S, B, and L: `torch.save(dict(model=state_dict))` of a
`Denoiser`, the post-hoc EMA of the training snapshots (`scripts/train.py export`). `brickflow.model.load_model` sizes
the model from the shapes. An IC model finishes sampling with the PT model of its size.

| size | width | depth | EMA (PT) | EMA (IC) |
| --- | --- | --- | --- | --- |
| S | 384 | 12 | 0.02 | 0.07 |
| B | 768 | 12 | 0.02 | 0.13 |
| L | 1024 | 24 | 0.04 | 0.16 |

## Evaluation set (`val/procbrick`)

ProcBrick renders: `procbrick.load_renders("val/procbrick")` reads them. `renders[i]` is the ProcBrick tree of render i,
its parts with print variants and colors, and `renders.poses(i)` its poses; `sample` and `evaluate` read each render's
parts, colors, and poses. IC models read its images.

## BrickNet evaluation set (`val/bricknet`)

The 512 models of BrickNet's validation split as ProcBrick renders, with one image each. It comes with the BrickNet
dataset, not with the released files: request access through the
[BrickNet dataset form](https://forms.gle/dm4eYSa5gh4DqzRT6), then download `brickflow_val.tar.xz` (46 MiB) from the
BrickNet dataset folder you are given and unpack it into `procbrick/` in BrickNet's data directory, which gives
`val/bricknet/`:

```bash
python -m brickflow sample weights/ic-l.pt val/bricknet --pt weights/pt-l.pt --samples 100 --guidance --out bn
python -m brickflow evaluate bn val/bricknet --ic
```

## Samples (`python -m brickflow sample`)

`samples-<k>.npz` holds the k-th sample of each input, in input order: the input's parts and colors and their sampled
float64 poses (LDU), snapped onto connector seats unless `--no-snap`. It is saved as a BrickNet graph batch
(`bricknet.load_graphs`); the poses are its `transforms`.

## Scores (`python -m brickflow evaluate`)

`scores.npz` in a samples directory, or `<file>.scores.npz` beside a samples file: per sample, float64 columns (below)
and `row`, its target.

| array | table column |
| --- | --- |
| `valid` | Valid |
| `single` | Single |
| `components` | Comp. |
| `max_component` | Max comp. |
| `collisions` | Coll. |
| `mm` | MM |
| `mm_trans` | MM_t |
| `mm_rot` | MM_r |

Per sample (`brickflow.score`):

- Connectivity and collisions take each part's representative and its pose. `connections` writes the LDraw lines (12
  significant digits) and parses them with BrickNet: `components`, `max_component` (parts in the largest), `single`
  (one component).
- `collisions`: the pairs that BrickNet's `colliding_pairs` proposes, no fixed connection joins, and that penetrate
  deeper than 1.1 LDU (BrickNet's depth between its colliders). `valid`: one component and no such pair. A part
  without a collider makes `collisions` NaN and the sample not valid.
- `mm`: sqrt of the mean over parts, every part weighted 1, of d^2 = |x - x'|^2 + tr((R - R') S (R - R')^T), x the
  center of mass and S = `second / mass`, least over a rigid motion of the target, the matching of parts with equal
  keys, and each part's symmetries. A branch and bound over rotations gives a lower bound (`Movement.bound`). With
  `certified` true, MM^2 - bound^2 <= certify * max(MM^2, 1 LDU^2), a bound on squared MM. `mm_trans` and `mm_rot` are
  the square roots of the mean first and second terms. MM is NaN when the sample and target hold different parts.
- The PT protocol (the default): representative ids, keys the ids, `PartTable(full=False)`. The IC protocol (`--ic`,
  `ic=True`): part ids, keys the part id and color, `PartTable()`.

Samples match their targets by position (sample i of a samples file, or entry i of an `.npz` file of parts and poses
modulo the targets) or, for LDraw files, by name (`<name>.ldr` and `<name>-<k>.ldr` against target `<name>.ldr`). A
table cell is the mean over all samples; its 95% interval is `brickflow.bootstrap`: 1,000 resamples of the targets,
each with all its samples.
