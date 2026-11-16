#!/usr/bin/env python3
"""BrickFlow training and weight export.

Usage: torchrun scripts/train.py pt S --out runs/pt-s
       python scripts/train.py export runs/pt-s --std 0.02 --step 1000000 --to pt-s.pt
"""

import argparse
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from brickflow import PartTable, flow
from brickflow.data import RenderRows, TreeRows, batches, loss_metric
from brickflow.images import DINO, training_images
from brickflow.model import SIZES, Denoiser, compiled, load_model, representative_rows

STAGES = dict(  # AdamW rate shared by S/B/L; decay start (1 / sqrt(t) after); EMA snapshot interval
    pt=dict(steps=1_000_000, adamw=4e-4, decay=50_000, snapshot=20_000),
    ic=dict(steps=200_000, adamw=1.41e-4, decay=20_000, snapshot=10_000),
)
MUON = dict(  # hidden-matrix learning rates by stage and model size
    pt=dict(S=2.4e-3, B=1.2e-3, L=9e-4), ic=dict(S=7.0667e-4, B=3.5333e-4, L=2.65e-4)
)
WARMUP = 1000
WEIGHT_DECAY = 0.01
GRAD_CLIP = 2.0
EMA_STDS = (0.05, 0.10)
EMA_EVERY = 8
LOG_EVERY = 50
CHECKPOINT_EVERY = 5000
NEW_IN_IC = ("material_", "image_pos_emb", "perceiver.", "blocks.image_", "head_keypoint.")


def _lr_factor(step: int, decay: int) -> float:
    """Linear warmup, constant until `decay`, then proportional to 1 / sqrt(step)."""
    warmup = min(1.0, (step + 1) / WARMUP)
    return warmup / max((step + 1) / decay, 1.0) ** 0.5


def _orthogonalize(g: torch.Tensor) -> torch.Tensor:
    """Approximately U V^T of every matrix in g: five quintic Newton-Schulz iterations in bf16."""
    transposed = g.size(-2) > g.size(-1)
    x = g.bfloat16().reshape(-1, *g.shape[-2:])
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7)
    for _ in range(5):
        gram = x @ x.mT
        polynomial = torch.baddbmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
        x = torch.baddbmm(x, polynomial, x, beta=3.4445)
    if transposed:
        x = x.mT
    return x.reshape(g.shape)


def _shared_seed(device: torch.device, world: int) -> int:
    """Shared initial random state."""
    seed = torch.randint(2**62, (), device=device)
    if world > 1:
        dist.broadcast(seed, 0)
    return int(seed)


def _orthogonalize_share(g: torch.Tensor) -> tuple[torch.Tensor, dist.Work]:
    """Orthogonalize this rank's share of g (L, A, B); the gathered (>= L, A, B) result and its all-gather's work."""
    world = dist.get_world_size()
    share = -(-len(g) // world)
    padded = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, share * world - len(g)))
    rank = dist.get_rank()
    mine = _orthogonalize(padded[rank * share : (rank + 1) * share]).contiguous()
    gathered = mine.new_empty(share * world, *mine.shape[1:])
    return gathered, dist.all_gather_into_tensor(gathered, mine, async_op=True)


class _Muon(torch.optim.Optimizer):
    """Muon: Nesterov momentum orthogonalized per matrix at AdamW's RMS; ranks share out stacked (L, A, B) weights."""

    def __init__(self, params, lr: float, weight_decay: float) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self) -> None:
        """One Muon step: orthogonalized momentum, each rank updating its share."""
        for group in self.param_groups:
            gathers = []
            for p in group["params"]:
                momentum = self.state[p].setdefault("momentum", torch.zeros_like(p))
                momentum.lerp_(p.grad, 0.05)
                update = p.grad.lerp(momentum, 0.95)
                if dist.is_initialized() and p.ndim == 3:
                    gathers.append((p, *_orthogonalize_share(update)))
                else:
                    self._apply(p, _orthogonalize(update), group)
            for p, gathered, work in gathers:
                work.wait()
                self._apply(p, gathered[: len(p)], group)

    @staticmethod
    def _apply(p, update, group) -> None:
        """Decay p and add the update scaled to its shape."""
        scale = 0.2 * max(p.shape[-2:]) ** 0.5
        p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)


def _ema_gamma(std: float) -> float:
    """Power-function EMA exponent of relative std `std` (Karras et al. 2024, eq. 126)."""
    return float(np.roots([1.0, 7.0, 16.0 - std**-2, 12.0 - std**-2]).real.max())


class _PowerEMA:
    """Power-function EMAs of the parameters, one per relative std in EMA_STDS."""

    def __init__(self, model: Denoiser) -> None:
        self.params = dict(model.named_parameters())
        self.averages = [[p.detach().float().clone() for p in self.params.values()] for _ in EMA_STDS]

    @torch.no_grad()
    def update(self, step: int) -> None:
        """Fold the current weights into the running averages."""
        if step % EMA_EVERY:
            return
        current = [p.detach().float() for p in self.params.values()]
        for average, std in zip(self.averages, EMA_STDS):
            beta = (1 - EMA_EVERY / step) ** (_ema_gamma(std) + 1) if step > EMA_EVERY else 0.0
            torch._foreach_lerp_(average, current, 1 - beta)

    def snapshot(self, step: int) -> dict:
        """State of the averages, in float16, to save."""
        profiles = []
        for average in self.averages:
            profiles.append({name: a.to("cpu", torch.float16) for name, a in zip(self.params, average)})
        return dict(step=step, profiles=profiles)


def _posthoc_ema(paths: Iterable[Path], std: float, step: int) -> dict:
    """EMA weights of relative std `std` at `step`: the snapshots' least-squares fit (Karras et al. 2024, Alg. 3), read
    one file at a time."""

    def inner(t_a, gamma_a, t_b, gamma_b):  # <a, b> of two power-function profiles
        power = np.where(t_a < t_b, gamma_b, -gamma_a)
        return (gamma_a + 1) * (gamma_b + 1) * (t_a / t_b) ** power / ((gamma_a + gamma_b + 1) * np.maximum(t_a, t_b))

    snapshots = sorted((int(Path(path).stem.split("-")[1]), path) for path in paths)
    snapshots = [(s, path) for s, path in snapshots if s <= step]
    if not snapshots:
        intervals = ", ".join(f"{stage.upper()} every {config['snapshot']:,}" for stage, config in STAGES.items())
        raise ValueError(f"no EMA snapshots at or before step {step} (training writes them: {intervals} steps)")
    t = np.repeat([s for s, _ in snapshots], len(EMA_STDS)).astype(np.float64)[:, None]
    gamma = np.tile([_ema_gamma(s) for s in EMA_STDS], len(snapshots))[:, None]
    gram = inner(t, gamma, t.T, gamma.T)
    wanted = inner(t, gamma, np.array([[step]]), np.array([[_ema_gamma(std)]]))
    weights = np.linalg.solve(gram, wanted)[:, 0]
    weights = iter(weights / weights.sum())
    out = {}
    for _, path in snapshots:
        for profile in torch.load(path, map_location="cpu")["profiles"]:
            w = float(next(weights))
            for name, value in profile.items():
                out[name] = out.get(name, 0.0) + w * value.float()
    return out


def _optimizers(model: Denoiser, stage: str, size: str) -> tuple[_Muon, torch.optim.AdamW]:
    """Muon for the hidden matrices; AdamW for the rest."""
    muon_params = model.muon_parameters()
    rest = [p for p in model.parameters() if all(p is not q for q in muon_params)]
    muon = _Muon(muon_params, MUON[stage][size], WEIGHT_DECAY)
    adamw = torch.optim.AdamW(rest, STAGES[stage]["adamw"], weight_decay=WEIGHT_DECAY, fused=True)
    return muon, adamw


class _Run:
    """Stage's model, optimizers, EMA, and data on this rank, resumed from the run's `last.pt` or initialized."""

    def __init__(self, args, device: torch.device, rank: int, world: int) -> None:
        self.stage = STAGES[args.stage]
        self.image_batch_size = args.image_batch_size
        self.image = args.stage == "ic"
        self.out = Path(args.out)
        (self.out / "ema").mkdir(parents=True, exist_ok=True)
        table = PartTable(full=self.image)  # IC models read full catalog ids
        self.mass, self.inertia = loss_metric(table, device)
        state = None
        if (self.out / "last.pt").exists():
            state = torch.load(self.out / "last.pt", map_location=device)
        self.seed = _shared_seed(device, world) if state is None else state["seed"]
        tokens = ()
        if self.image:
            backgrounds = Path(args.backgrounds).read_text(encoding="utf-8").split() if args.backgrounds else []
            self.dataset = RenderRows(args.data, args.assets, table, backgrounds, self.seed, args.threads)
            self.encoder = DINO().to(device)
            blank = torch.zeros(1, self.encoder.resolution, self.encoder.resolution, 3, dtype=torch.uint8)
            _, summary_and_patches, features = self.encoder(blank.to(device)).shape
            tokens = (features, summary_and_patches - 1)
        else:
            self.dataset = TreeRows(args.data, args.assets, table, self.seed, args.threads)
        torch.manual_seed(self.seed)  # the same initialization on every rank
        self.model = Denoiser(*SIZES[args.size], self.image, *tokens).to(device)
        self.muon, self.adamw = _optimizers(self.model, args.stage, args.size)
        self.step = 0
        self.resumable = dict(model=self.model, muon=self.muon, adamw=self.adamw)
        if state is not None:
            for name, part in self.resumable.items():
                part.load_state_dict(state[name])
            self.step = state["step"]
        elif self.image:  # start from the pretrained model; the image modules keep their initialization
            weights = representative_rows(load_model(args.init).state_dict())
            missing, unexpected = self.model.load_state_dict(weights, strict=False)
            if unexpected or not all(name.startswith(NEW_IN_IC) for name in missing):
                raise ValueError(f"{args.init} is not a PT model of size {args.size}")
        seed = np.random.SeedSequence([self.seed, rank, self.step]).generate_state(1)[0]
        torch.cuda.manual_seed(int(seed))  # augmentation
        self.ema = _PowerEMA(self.model)
        if state is not None:
            self.ema.averages = state["ema"]
        self.network = self.model
        if world > 1:
            torch._dynamo.config.optimize_ddp = "python_reducer"
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.model, [device.index], gradient_as_bucket_view=True
            )
        self.forward_loss = compiled(self._forward_loss)

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict]:
        """Loss of a batch and its terms."""
        split = batch["lengths"][:, 0]
        inputs = [batch[key] for key in ("trans_t", "quat_t", "types", "t")]
        out = self.network(*inputs, split, batch.get("materials"), batch.get("features"))
        return flow.loss(out, batch, self.mass, self.inertia)

    def prepared(self, stream: Iterator[dict]) -> Iterator[dict]:
        """Batches of a stream; IC: with their image features and keypoints, made on a side stream while the previous
        step trains."""
        if not self.image:
            yield from stream
            return
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        while True:
            with torch.cuda.stream(side):  # the batch's copies to the device too
                batch = next(stream, None)
                if batch is None:
                    return
                batch["features"], batch["keypoints"] = training_images(batch, self.encoder, self.image_batch_size)
            main.wait_stream(side)
            for value in batch.values():
                value.record_stream(main)  # its memory stays the main stream's until freed there
            yield batch

    def update(self, batch) -> tuple[dict, float]:
        """One optimizer step on a batch: the loss and its terms, and the learning-rate factor."""
        factor = _lr_factor(self.step, self.stage["decay"])
        for optimizer in (self.muon, self.adamw):
            for group in optimizer.param_groups:
                group["lr"] = optimizer.defaults["lr"] * factor
        loss, logs = self.forward_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.muon.step()
        self.adamw.step()
        self.model.zero_grad(set_to_none=True)
        self.step += 1
        self.ema.update(self.step)
        return dict(loss=loss, **logs), factor

    def save(self) -> None:
        """Write the training state to `last.pt`."""
        state = {name: part.state_dict() for name, part in self.resumable.items()}
        torch.save(dict(state, ema=self.ema.averages, step=self.step, seed=self.seed), self.out / "last.pt.tmp")
        os.replace(self.out / "last.pt.tmp", self.out / "last.pt")


def train(args) -> None:
    """Train a stage (PT or IC) of a model size, resuming from `last.pt`."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run = _Run(args, device, rank, world)
    stage = run.stage
    rows = args.batch_size // 2 // world
    chunk = min(8, rows) if run.image else rows
    if args.batch_size % (2 * world) or rows < 1 or rows % chunk:
        raise ValueError("the batch size must divide into whole rows and chunks on each rank")
    clock = time.time()
    window = {}
    for batch in run.prepared(batches(run.dataset, rows, chunk, args.workers, rank, world, run.step, device)):
        if run.step >= stage["steps"]:
            break
        logs, factor = run.update(batch)
        step = run.step
        for key, value in logs.items():
            window[key] = window.get(key, 0.0) + value.detach()
        if step % LOG_EVERY == 0:
            if rank == 0:
                text = " ".join(f"{key} {value.item() / LOG_EVERY:.4f}" for key, value in window.items())
                rate = LOG_EVERY * args.batch_size / (time.time() - clock)
                print(f"step {step} {text} lr_scale {factor:.4f} {rate:.0f} samples/s", flush=True)
            clock = time.time()
            window = {}
        if rank == 0 and step % stage["snapshot"] == 0:
            torch.save(run.ema.snapshot(step), run.out / "ema" / f"ema-{step:08d}.pt")
        if rank == 0 and (step % CHECKPOINT_EVERY == 0 or step == stage["steps"]):
            run.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    stages = (
        ("pt", "pretrain on assemblies", "train/pt", "assemblies"),
        ("ic", "finetune on renders from PT weights", "train/ic", "renders"),
    )
    for name, text, data, kind in stages:
        p = sub.add_parser(name, help=text)
        p.add_argument("size", choices=list(SIZES), help="model size")
        p.add_argument("--data", default=data, help=f"ProcBrick {kind} (default: {data})")
        p.add_argument("--assets", default="assets", help="assets directory (default: assets)")
        p.add_argument("--out", required=True, help="run directory: last.pt and ema/")
        p.add_argument("--init", help="IC: the PT model's weights, e.g. weights/pt-s.pt (required)")
        p.add_argument("--backgrounds", help="IC: a text file listing background images (default: white)")
        p.add_argument("--workers", type=int, default=16, help="data worker processes per rank")
        p.add_argument("--threads", type=int, default=4, help="coupling threads per data worker")
        p.add_argument("--batch-size", type=int, default=4096 if name == "pt" else 2048, help="samples per step")
        p.add_argument("--image-batch-size", type=int, default=128, help="images per encoder call")
    p = sub.add_parser("export", help="post-hoc EMA weights of a run")
    p.add_argument("run", help="run directory")
    p.add_argument("--std", type=float, required=True, help="relative EMA std")
    p.add_argument("--step", type=int, required=True, help="training step")
    p.add_argument("--to", required=True, help="output weights file")
    args = ap.parse_args()
    if args.command == "ic" and args.init is None:
        ap.error("ic needs --init")
    if args.command == "export":
        weights = _posthoc_ema(Path(args.run, "ema").glob("ema-*.pt"), args.std, args.step)
        torch.save(dict(model=weights), args.to)
    else:
        args.stage = args.command
        train(args)


if __name__ == "__main__":
    main()
