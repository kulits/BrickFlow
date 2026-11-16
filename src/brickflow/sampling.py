"""In-memory and file-based pose sampling."""

import collections
import itertools
import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import bricknet
import numpy as np
import procbrick
import torch
from PIL import Image
from procbrick import cpus

from . import flow
from .connect import Workers
from .images import DINO, on_white
from .model import load_model, material_ids
from .parts import _parsed, pose_matrices

_TOKENS = {384: 262144, 768: 131072, 1024: 65536}  # largest batch in parts, by model width (others: the next wider's)
_IMAGE_MEMORY = 0.1  # memory share for image keys and values; the null condition takes as much again
_IMAGES = None


def _load(data: str) -> tuple[list[bricknet.Graph], list[str], procbrick.Renders | list[Path | None]]:
    """Parts, colors, and poses in BrickNet containers, their names, and their images from an input path."""
    path = Path(data)
    if path.suffix == ".npz":
        graphs = bricknet.load_graphs(path)
        return graphs, [str(i) for i in range(len(graphs))], [None] * len(graphs)
    if path.is_dir() and not (path / "sizes.npy").exists():
        files = sorted(path.glob("*.ldr"))
        graphs = [bricknet.parse_ldr(file.read_text(encoding="utf-8")) for file in files]
        found = [[file.with_suffix(s) for s in (".png", ".jpg") if file.with_suffix(s).exists()] for file in files]
        return graphs, [file.stem for file in files], [paths[0] if paths else None for paths in found]
    renders = procbrick.load_renders(data)
    trees, poses = renders._decode(range(len(renders)))
    starts = np.r_[0, np.cumsum(renders.sizes)]
    graphs = [
        replace(bricknet.tree_to_graph(tree), transforms=poses[a:b]) for tree, a, b in zip(trees, starts, starts[1:])
    ]
    return graphs, [str(i) for i in range(len(renders))], renders


def _open_images(images) -> None:
    """Hold the input images in a decoding process."""
    global _IMAGES
    _IMAGES = images


def _image_on_white(row: int, resolution: int, images=None) -> np.ndarray:
    """Image of input `row` (a render, a path, or a PIL image) as the image encoder takes it, in a decoding process."""
    images = _IMAGES if images is None else images
    if isinstance(images, procbrick.Renders):
        return on_white(Image.fromarray(images.image(row)), resolution)
    if isinstance(images[row], Image.Image):
        return on_white(images[row], resolution)
    with Image.open(images[row]) as image:
        return on_white(image, resolution)


def _pictures(images, rows, ahead: int, resolution: int, workers: int | None) -> Iterator[np.ndarray]:
    """Images of `rows`, decoded in order with bounded lookahead."""
    if workers == 0:
        yield from (_image_on_white(row, resolution, images) for row in rows)
        return
    with ProcessPoolExecutor(workers or len(cpus()), get_context("spawn"), _open_images, (images,)) as decoder:
        decoding = collections.deque()
        for row in rows:
            decoding.append(decoder.submit(_image_on_white, row, resolution))
            if len(decoding) > ahead:
                yield decoding.popleft().result()
        while decoding:
            yield decoding.popleft().result()


def _image_keys_values(model, images, batches, device, chunk: int, workers: int | None) -> Iterator[torch.Tensor]:
    """Each batch's image keys and values (depth, B, K, 2W), by the image encoder from images decoded ahead."""
    encoder = DINO().to(device)
    pictures = _pictures(images, np.concatenate(batches).tolist(), 8 * chunk, encoder.resolution, workers)

    def keys_values(rows):
        chunks = []
        for group in np.array_split(rows, -(-len(rows) // chunk)):
            features = encoder(torch.as_tensor(np.stack([next(pictures) for _ in group]), device=device))
            chunks.append(model.encode_image(features))
        return torch.cat(chunks, 1)

    return map(keys_values, batches)


def _pipelined(runs: Iterator[Generator], depth: int) -> Iterator:
    """Drive sampling generators and yield their results: up to `depth` guided runs advance oldest first, while the
    next run takes unguided steps during their energies."""
    guided = collections.deque()
    pending = None
    taken = 0
    unguided = None
    while True:
        if pending is None:
            pending, taken = next(runs, None), 0
        if pending is not None and (len(guided) < depth or (taken < unguided and not guided[0][1].done())):
            try:
                wait = next(pending)
            except StopIteration as finished:
                pending = None
                yield finished.value
                continue
            if wait is None:
                taken += 1
                if len(guided) == depth and torch.cuda.is_available():
                    torch.cuda.synchronize()  # bound queued work to one step
            else:
                guided.append((pending, wait))
                pending, unguided = None, taken
        elif guided:
            run, _ = guided.popleft()
            try:
                guided.append((run, next(run)))
            except StopIteration as finished:
                yield finished.value
        else:
            return


def sample(
    weights: str | Path,
    parts: list[tuple[np.ndarray, np.ndarray]],
    samples: int = 1,
    images=None,
    pt: str | Path | None = None,
    guidance: bool = False,
    snap: bool = True,
    assets: str | Path = "assets",
    workers: int | None = 0,
    settings: flow.Settings = flow.Settings(),
    batch_size: int | None = None,
    image_batch_size: int = 64,
    on_step: Callable[[int, float, np.ndarray], None] | None = None,
) -> list[np.ndarray]:
    """Sample poses (samples, n, 4, 4) for each input's part ids and LDraw colors.
    IC models take `images` and a PT model for the handoff.
    `on_step(index, t, poses)` receives independent float64 (samples, n, 4, 4) arrays in LDU at the initial noise
    and each schedule time, before snap. Indices refer to `parts`; batches may interleave their callbacks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = Workers(assets, workers) if guidance else None  # starting while the model loads
    try:
        model = load_model(weights, device)
        if model.image and images is None:
            raise ValueError(f"{weights} is an IC model: give an image of each input")
        sizes = np.array([len(part_ids) for part_ids, _ in parts], np.int64)
        width = model.blocks.qkv_w.shape[2]
        tokens = _TOKENS[min([w for w in _TOKENS if w >= width], default=max(_TOKENS))]
        pt = load_model(pt, device) if model.image and pt is not None else None
        image_rows = tokens  # rows of image keys and values a batch may hold
        if model.image and device.type == "cuda":
            image_rows = int(_IMAGE_MEMORY * torch.cuda.get_device_properties(device).total_memory)
            image_rows //= flow.null_image_kv(model).nbytes
        padded_sizes = flow.padded(sizes)
        in_flight = 3 if guidance else 1  # overlap denoising with the other batches' energies
        batches = []
        for padded in np.unique(padded_sizes):
            members = np.flatnonzero(padded_sizes == padded)
            rows_per_batch = batch_size or max(1, min(tokens // padded, image_rows) // in_flight)
            count = min(-(-len(members) * samples // rows_per_batch), len(members))
            batches += np.array_split(members, count)
        image_kvs = itertools.repeat(None)
        if model.image:
            image_kvs = _image_keys_values(model, images, batches, device, image_batch_size, workers)
        offsets = np.r_[0, np.cumsum(sizes)]
        placed = [np.zeros((offsets[-1], 7), np.float32) for _ in range(samples)]

        def run(members, image_kv):
            """One batch, as a sampling generator."""

            def observe(t: float, placed: np.ndarray) -> None:
                for s, r in enumerate(members):
                    at = placed[s, :, : sizes[r]]
                    on_step(int(r), t, pose_matrices(at[..., :4], at[..., 4:]))

            part_ids = [parts[r][0] for r in members]
            materials = [material_ids(parts[r][1], assets) for r in members] if model.image else None
            steps = flow.generate(
                model,
                part_ids,
                samples,
                pool,
                materials,
                image_kv,
                pt,
                settings,
                observe if on_step is not None else None,
            )
            sampled = yield from steps
            for s, r in enumerate(members):
                for k in range(samples):
                    placed[k][offsets[r] : offsets[r + 1]] = sampled[s, k, : sizes[r]]

        runs = (run(members, image_kv) for members, image_kv in zip(batches, image_kvs))
        for _ in _pipelined(runs, in_flight):  # every run to its end
            pass
        poses = [np.split(pose_matrices(placed[k][:, :4], placed[k][:, 4:]), offsets[1:-1]) for k in range(samples)]
        if snap:
            ids = [part_ids for part_ids, _ in parts]
            pool = pool or Workers(assets, len(cpus()) if workers is None else workers)
            poses = [pool._snapped(list(zip(ids, poses_k))) for poses_k in poses]
        return [np.stack([poses[k][i] for k in range(samples)]) for i in range(len(parts))]
    finally:
        if pool is not None:
            pool.close()


def _run(args) -> None:
    """`python -m brickflow sample`: `sample` of every input of `args.data`, the k-th sample of each written to
    `samples-<k>.npz`."""
    inputs, names, images = _load(args.data)
    if not isinstance(images, procbrick.Renders):
        missing = [name for name, image in zip(names, images) if image is None]
        if missing and len(missing) < len(names):
            raise SystemExit(f"{args.data}: {missing[0]} has no image beside it")
        images = None if missing else images
    parts = [(np.asarray(graph.part_ids, np.int64), np.asarray(graph.colors, np.int64)) for graph in inputs]
    started = time.time()
    try:
        poses = sample(
            args.weights,
            parts,
            args.samples,
            images,
            args.pt,
            args.guidance,
            not args.no_snap,
            args.assets,
            args.workers,
            batch_size=args.batch_size,
            image_batch_size=args.image_batch_size,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(f"{len(parts) * args.samples} samples in {time.time() - started:.0f} s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, colors = zip(*parts)
    with ProcessPoolExecutor(args.workers or len(cpus()), get_context("spawn")) as pool:
        for k in range(args.samples):
            graphs = pool.map(_parsed, ids, colors, (at[k] for at in poses), chunksize=16)
            bricknet.save_graphs(list(graphs), out / f"samples-{k:03d}.npz")
            print(f"{len(parts)} samples -> {out / f'samples-{k:03d}.npz'}", flush=True)
