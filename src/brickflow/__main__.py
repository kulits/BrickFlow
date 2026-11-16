"""Command-line interface: python -m brickflow <command>."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import bricknet
import numpy as np
import procbrick.paths
from procbrick import cpus

from . import __version__
from .sampling import _load, _run
from .metrics import _score_job, bootstrap
from .paths import WEIGHTS, fetch

_DEFAULT = ("assets", "val/procbrick", "weights/pt-l")  # a bare `fetch`
_COLUMNS = dict(  # metric: label, scale
    valid=("Valid %", 100),
    single=("Single %", 100),
    components=("Comp.", 1),
    max_component=("Max comp.", 1),
    collisions=("Coll.", 1),
    mm=("MM", 1),
    mm_trans=("MM_t", 1),
    mm_rot=("MM_r", 1),
)


def _cmd_fetch(args) -> None:
    try:
        fetch(WEIGHTS + procbrick.paths.NAMES if args.all else args.names or _DEFAULT)
    except ValueError as error:
        sys.exit(str(error))


def _jobs(samples: Path, targets: list, names: list[str], ic: bool) -> list[tuple]:
    """Each sample with its target: in `sample`'s output directory, sample i of every `samples-<k>.npz` against target
    i; in an .npz file of parts and poses, entry i against target i modulo the targets; in a directory of LDraw files,
    `<name>.ldr` and `<name>-<k>.ldr` against target `<name>`."""
    if samples.is_dir() and not any(samples.glob("*.ldr")):
        files = sorted(samples.glob("samples-*.npz"))
        graphs = [graph for file in files for graph in bricknet.load_graphs(file)]
        rows = np.tile(np.arange(len(targets)), len(files))
    else:
        graphs, sample_names, _ = _load(str(samples))
        if samples.suffix == ".npz":
            rows = np.arange(len(graphs)) % len(targets)
        else:
            row = {name: index for index, name in enumerate(names)}
            rows = [row[name] if name in row else row[name.rsplit("-", 1)[0]] for name in sample_names]
    if len(graphs) != len(rows):
        sys.exit(f"{samples}: the samples files are not one sample of each of the {len(targets)} targets")
    return [(graph, targets[row], ic, row) for graph, row in zip(graphs, rows)]


def _cmd_evaluate(args) -> None:
    """Score samples against their targets and print the table row, with 95% intervals over the targets."""
    samples = Path(args.samples)
    targets, names, _ = _load(args.targets)
    jobs = _jobs(samples, targets, names, args.ic)
    with ProcessPoolExecutor(args.workers or len(cpus()), get_context("spawn")) as pool:
        results = list(pool.map(_score_job, jobs, chunksize=16))
    columns = {key: np.array([result.get(key, np.nan) for result in results], np.float64) for key in _COLUMNS}
    columns["row"] = np.array([result["row"] for result in results])
    out = samples / "scores.npz" if samples.is_dir() else samples.with_suffix(".scores.npz")
    np.savez(out, **columns)
    print(f"{len(results)} samples of {len(np.unique(columns['row']))} targets -> {out}")
    for key, (label, scale) in _COLUMNS.items():
        mean, low, high = (scale * v for v in bootstrap(columns[key], columns["row"]))
        print(f"{label:10s} {mean:.2f} [{low:.2f}, {high:.2f}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="brickflow", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download released files into the data directory")
    p.add_argument("names", nargs="*", help=f"names in DATA.md (default: {' '.join(_DEFAULT)})")
    p.add_argument("--all", action="store_true", help="fetch everything")
    p.set_defaults(func=_cmd_fetch)

    data = "ProcBrick renders (val/procbrick), parts and poses in BrickNet's .npz format, or a directory of LDraw files"
    p = sub.add_parser("sample", help="sample poses for the parts of each input, then snap them")
    p.add_argument("weights", help="weights file, e.g. weights/pt-l.pt")
    p.add_argument("data", help=f"{data}; for IC models, each LDraw file with its image <name>.png or <name>.jpg")
    p.add_argument("--out", required=True, help="output directory: samples-<k>.npz, the k-th sample of each input")
    p.add_argument(
        "--pt", help="IC models: the PT model of the same size, which finishes sampling (default: the IC model)"
    )
    p.add_argument("--samples", type=int, default=1, help="samples per input")
    p.add_argument("--guidance", action="store_true", help="connectivity guidance")
    p.add_argument("--no-snap", action="store_true", help="keep the samples as sampled")
    p.add_argument("--assets", default="assets", help="assets directory (default: assets)")
    p.add_argument("--workers", type=int, help="sampling worker processes")
    p.add_argument("--batch-size", type=int, help="samples per model batch (default: sized to model and memory)")
    p.add_argument("--image-batch-size", type=int, default=64, help="images per encoder call")
    p.set_defaults(func=_run)

    p = sub.add_parser("evaluate", help="score samples against their targets with the paper's metrics")
    p.add_argument(
        "samples",
        help="output directory of `sample`, parts and poses in BrickNet's .npz format, or directory of LDraw files",
    )
    p.add_argument("targets", help=data)
    p.add_argument("--ic", action="store_true", help="IC protocol: parts matched on part id and color")
    p.add_argument("--workers", type=int, help="scoring worker processes")
    p.set_defaults(func=_cmd_evaluate)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
