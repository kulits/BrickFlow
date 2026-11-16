<div align="center">

# BrickFlow: Connectivity-Guided Brick Reconstruction

[Peter Kulits](https://kulits.github.io/) and [Cordelia Schmid](https://cordeliaschmid.github.io/)

NeurIPS 2026

[\[Project Page\]](https://kulits.github.io/BrickFlow/) | [\[Models\]](https://huggingface.co/kulits/BrickFlow) | [\[Dataset\]](https://huggingface.co/datasets/kulits/BrickFlow)

<img src="https://kulits.github.io/BrickFlow/assets/teaser_duck.avif" width="800">

<img src="https://kulits.github.io/BrickFlow/assets/teaser_truck.avif" width="800">

</div>

## Install

```bash
pip install brick-flow
```

The default `fetch` downloads the files for PT-L sampling (3.81 GiB: `assets`, `val/procbrick`, and `weights/pt-l`)
and BrickNet's colliders (191 MiB download, 989 MiB extracted):

```bash
python -m brickflow fetch
```

For IC-L sampling, also fetch `weights/ic-l` (2.48 GiB); see [Generation](#generation).

Files are stored in `brickflow/` and `procbrick/` in BrickNet's data directory; set `BRICKNET_DATA` to use another
location.

## Data Structures

BrickFlow takes parts and colors/materials and samples their poses: float64 (n, 4, 4) matrices in LDU at each part's
LDraw origin. IC models also take an image.

```python
import dataclasses

import bricknet
import brickflow
from brickflow.sampling import sample

model = bricknet.parse_ldr(open("model.ldr").read())
parts = (model.part_ids, model.colors)

poses = sample("weights/pt-l.pt", [parts], samples=4, guidance=True)[0]   # (4, n, 4, 4) float64, LDU
for k, at in enumerate(poses):
    open(f"sample-{k}.ldr", "w").write(bricknet.graph_to_ldr(dataclasses.replace(model, transforms=at)))

print(brickflow.score((model.part_ids, poses[0])))                        # valid, single, components, collisions
print(brickflow.mass_movement((model.part_ids, poses[0]), model).rms)     # mass movement (MM) onto the model, LDU
```

The sampler's settings (`brickflow.flow.Settings`, the paper's by default) include the time schedule, the hand-off,
the classifier-free guidance weight, the guidance's start, strength, and cap, and the seed:

```python
import numpy as np
from brickflow.flow import Settings

settings = Settings(schedule=tuple(np.linspace(0, 1, 101) ** 0.5), seed=0)
poses = sample("weights/pt-l.pt", [parts], samples=4, guidance=True, settings=settings)[0]
```

To record the denoising trajectory, pass `on_step(index, t, poses)` to `sample`. The callback receives the input
index, time, and poses `(samples, n, 4, 4)` in LDU at the initial noise and each step, before snap:

```python
trajectory = []

def record(index, t, poses):
    trajectory.append((index, t, poses))

poses = sample("weights/pt-l.pt", [parts], guidance=True, on_step=record)[0]
```

IC models also take an image of each input, and the PT model of their size:

```python
from PIL import Image

image = Image.open("duck.png")
poses = sample("weights/ic-l.pt", [parts], images=[image], pt="weights/pt-l.pt", guidance=True)[0]
```

Modules:

- `sampling`: sampling poses, in memory and from files into files (`python -m brickflow sample`)
- `parts`: part tables and symmetries
- `metrics`: MM, connectivity, collisions, scores
- `connect`: connectivity energy and snap
- `model`, `flow`: the denoiser and its sampling steps
- `data`, `images`: training rows and image features
- `paths`: the data directory and `fetch`

## Generation

Unconditional generation with a [PT model](https://huggingface.co/kulits/BrickFlow) (pretrained on procedural
assemblies), with connectivity guidance, for every render of the evaluation set:

```bash
python -m brickflow sample weights/pt-l.pt val/procbrick --samples 100 --guidance --out pt-l
```

Image-conditioned generation with an IC model (finetuned from the PT model, which finishes its sampling). IC models
need the gated [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m): accept its license and run
`hf auth login`.

```bash
python -m brickflow fetch weights/ic-l
python -m brickflow sample weights/ic-l.pt val/procbrick --pt weights/pt-l.pt --samples 100 --guidance --out ic-l
```

For your own images, put each image in a directory with an LDraw file of the same name that lists its parts and their
colors (`photos/duck.png` and `photos/duck.ldr`):

```bash
python -m brickflow sample weights/ic-l.pt photos --pt weights/pt-l.pt --samples 4 --guidance --out duck
```

`samples-<k>.npz` holds the k-th sample of each input (`--no-snap`: as sampled). As LDraw files:

```python
for i, sample in enumerate(bricknet.load_graphs("duck/samples-000.npz")):
    open(f"duck-{i}.ldr", "w").write(bricknet.graph_to_ldr(sample))
```

## Evaluation

```bash
python -m brickflow evaluate pt-l val/procbrick        # the paper's metrics and table row
python -m brickflow evaluate ic-l val/procbrick --ic   # IC protocol: parts matched on part id and color
python -m brickflow evaluate outputs targets           # another generator's LDraw files against their targets
```

## Training

Training uses ProcBrick's splits:

```bash
python -m brickflow fetch train/pt   # PT: 100,000,000 ProcBrick trees of 100 parts (19.4 GiB)
python -m brickflow fetch train/ic   # IC: 1,000,000 renders (100.5 GiB)
```

The script selects learning rates and schedules by stage (`pt` or `ic`) and size (`S`, `B`, or `L`). Pretrain PT-S and
export its post-hoc EMA weights:

```bash
torchrun --standalone --nproc-per-node 8 scripts/train.py pt S --out runs/pt-s
python scripts/train.py export runs/pt-s --std 0.02 --step 1000000 --to pt-s.pt
```

Finetune IC-S from it:

```bash
torchrun --standalone --nproc-per-node 8 scripts/train.py ic S --init pt-s.pt --out runs/ic-s
python scripts/train.py export runs/ic-s --std 0.07 --step 200000 --to ic-s.pt
```

IC uses white backgrounds by default. For the paper's background augmentation, add `--backgrounds imagenet.txt`, a text
file listing ImageNet training images. For B or L, select that size in both stages and initialize IC from its PT
weights; the EMA widths for all six models are in [DATA.md](https://github.com/kulits/BrickFlow/blob/master/DATA.md#weights).

## Scripts

```bash
python scripts/train.py --help   # pretraining, finetuning, and exporting the post-hoc EMA weights (Training)
```

## Data

The part tables ship inside the package; the released files are downloaded separately (see [Install](#install)), and
the weights and datasets are on Hugging Face; see [DATA.md](https://github.com/kulits/BrickFlow/blob/master/DATA.md) for schemas. The BrickNet evaluation set,
`val/bricknet`, comes with the BrickNet dataset ([request form](https://forms.gle/dm4eYSa5gh4DqzRT6)); see
[DATA.md](https://github.com/kulits/BrickFlow/blob/master/DATA.md#bricknet-evaluation-set-valbricknet).

## Citation

```bibtex
@InProceedings{kulits2026brickflow,
  author    = {Kulits, Peter and Schmid, Cordelia},
  title     = {BrickFlow: Connectivity-Guided Brick Reconstruction},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
