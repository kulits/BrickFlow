"""Connectivity energy, snap, and atlas workers."""

import contextlib
import os
import signal
import weakref
from collections import deque
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import bricknet
import numpy as np
from bricknet.collision import penetration
from procbrick import canonical_id, cpus

from . import _core
from .parts import _arrays, _parsed, table
from .paths import find

if os.name == "posix":
    from multiprocessing import resource_tracker

_ATLAS = None
_PENDING = 4  # jobs a worker holds at most; a fifth waits for the first's answer, so pipes never fill
_IDLE = 2  # blocks of shared memory kept for reuse between calls (their pages are mapped already)
_FILES = ("atlas.index", "atlas.tiles", "charts", "parts.json", "pose_dictionary")


def _assets(root) -> Path:
    """Assets directory, checked to hold the files the atlas opens; a FileNotFoundError names what to fetch."""
    for name in _FILES:
        find(f"{root}/{name}")
    return find(root)


def _serve(pipe, assets, cpu) -> None:
    """Atlas worker until None; exceptions return to the caller."""
    global _ATLAS
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # interrupts are the main process's to handle
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    _ATLAS = Atlas(assets)
    try:
        while (job := pipe.recv()) is not None:
            function, args = job
            try:
                pipe.send(function(*args))
            except Exception as error:
                pipe.send(error)
            except BaseException as error:
                pipe.send(RuntimeError(f"{type(error).__name__}: {error}"))
    except (EOFError, OSError):
        pass


def _row(parts: int) -> np.dtype:
    """Row of an energy batch: poses (N, 4, 4), energy, gradients (N, 6), part count, and part ids (N,)."""
    fields = [("pose", np.float64, (parts, 4, 4)), ("energy", np.float64), ("gradient", np.float64, (parts, 6))]
    return np.dtype(fields + [("n", np.int64), ("ids", np.uint32, (parts,))], align=True)


def _energies(name, count, parts, rows) -> None:
    """Write E and dE/d(body twist [w, v]) of `rows` into the batch of `count` rows in shared memory block `name`;
    single parts get E = 0 and padding a zero gradient."""
    block = SharedMemory(name)
    batch = np.ndarray(count, _row(parts), block.buf)
    for b in rows:
        n = batch["n"][b]
        batch["energy"][b] = 0
        batch["gradient"][b] = 0
        if n > 1:
            batch["energy"][b], batch["gradient"][b, :n] = _ATLAS.energy((batch["ids"][b, :n], batch["pose"][b, :n]))
    del batch  # no view may outlive the mapping, or close() fails
    block.close()


def _snap_jobs(jobs: list[tuple]) -> list[np.ndarray]:
    """Snapped poses of a chunk of assemblies, using this worker's atlas."""
    return [_ATLAS.snap(job)[1] for job in jobs]


def _free(block) -> None:
    """Unlink a block of shared memory and close it, unless a view of it is alive (interrupted while filling it)."""
    block.unlink()
    with contextlib.suppress(BufferError):
        block.close()


def _stop(pipes, processes, idle) -> None:
    """Stop worker processes, each after its current job, and free the idle blocks of shared memory; closing its pipe
    stops a worker sending an answer at once."""
    for pipe in pipes:
        try:
            pipe.send(None)
        except OSError:
            pass
        pipe.close()
    for process in processes:
        if process.pid is not None:
            process.join(5)
    while idle:
        _free(idle.pop())


def _keep(idle: list, block) -> None:
    """Keep a block for the next calls, with at most _IDLE others."""
    idle.append(block)
    if len(idle) > _IDLE:
        _free(idle.pop(0))


class Atlas:
    """Connection atlas of an assets directory, in this process; its parts are the physical-class representatives."""

    def __init__(self, assets: str | Path = "assets") -> None:
        self.root = _assets(assets)
        parts = table(False)
        self._atlas = _core.Atlas(self.root, parts._columns)

    def energy(self, assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray]) -> tuple[float, np.ndarray]:
        """Connectivity energy and its gradient in each part's body twist [w, v].

        Each connector hypothesis h has seat distance c_h: the least mass-weighted mean squared material displacement
        that seats the two parts, over the joint coordinates and their common rigid motion. Its conductance is
        g_h = w_h / (rho + q(c_h)), where rho = tau = 1e-8 LDU^2, q(c) = tau H((c - tau) / tau), and
        H(x) = x^3 / (x^2 + 1) for x > 0, else 0. The taper w_h is a quintic smoothstep of connector-origin
        distance (less the slide reach for axles): 1 below 12 LDU, 0 beyond 16 LDU.

        Each part pair conducts (sum_h g_h^4)^(1/4), plus 1 / (16 n) LDU^-2. For effective resistances Omega_ab,
        Omega0 = (n - 1) rho, and phi(x) = Omega0 H(x / Omega0):

        E = 1 / (n - 1) sum_{a<b} phi(Omega_ab - Omega0).

        E and its gradient vanish when seated hypotheses connect all parts, and for n < 2.
        """
        part_ids, poses, _ = _arrays(assembly)
        return self._atlas.energy(canonical_id(part_ids), poses)

    def snap(
        self, assembly: bricknet.Graph | tuple[np.ndarray, np.ndarray]
    ) -> bricknet.Graph | tuple[np.ndarray, np.ndarray]:
        """The assembly snapped onto connector seats, of the kind given."""
        part_ids, poses, colors = _arrays(assembly)
        ids = canonical_id(part_ids).tolist()

        def depths(pairs, at):
            return np.array([penetration(ids[i], at[m, 0], ids[j], at[m, 1]) for m, (i, j) in enumerate(pairs)])

        snapped = self._atlas.snap(ids, poses, depths)
        if isinstance(assembly, bricknet.Graph):
            return _parsed(part_ids, colors, snapped)
        return part_ids, snapped


class _Job:
    """Job sent to a worker; done() tells whether its answer is in, result() waits for it."""

    def __init__(self, workers: "Workers", worker: int) -> None:
        self.workers = workers
        self.worker = worker
        self.answered = False

    def done(self) -> bool:
        """Whether the answer is in, reading what has arrived."""
        while not self.answered and self.workers.pipes[self.worker].poll():
            self.workers._receive(self.worker)
        return self.answered

    def result(self) -> list[np.ndarray] | None:
        """Worker's answer, once in; a worker exception is raised here."""
        while not self.answered:
            self.workers._receive(self.worker)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class _Batch:
    """Energy call's rows in shared memory; done() tells whether the energies are in, calling waits for them."""

    def __init__(self, idle: list, block: SharedMemory | None, count: int, row: np.dtype) -> None:
        self.idle = idle
        self.block = block
        self.count = count
        self.row = row
        self.jobs = []
        if block is not None:
            self.free = weakref.finalize(self, _free, block)  # also if interrupted, before the resource tracker sees it

    def done(self) -> bool:
        """Whether every worker's energies are in."""
        return all(job.done() for job in self.jobs)

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """E (B,) and dE/d(body twist) (B, N, 6) of every row, once the workers are done; the block is kept for the next
        calls."""
        if self.block is None:
            return self.values
        for job in self.jobs:
            job.result()
        rows = np.ndarray(self.count, self.row, self.block.buf)
        E, G = rows["energy"].copy(), rows["gradient"].copy()
        del rows
        self.free.detach()
        _keep(self.idle, self.block)
        return E, G


class Workers:
    """Atlas workers with shared memory, closed explicitly, on collection, or at exit."""

    def __init__(self, assets: str | Path = "assets", workers: int | None = None) -> None:
        root = _assets(assets)
        self.pipes = []
        self.processes = []
        self.idle = []
        self._stop = weakref.finalize(self, _stop, self.pipes, self.processes, self.idle)
        self.local = Atlas(root) if workers == 0 else None
        if self.local is not None:
            return
        context = get_context("spawn")
        available = cpus()
        pinned = available[::-1]
        if os.name == "posix":  # one resource tracker, shared with the workers: their attachments end with the blocks
            resource_tracker.ensure_running()
        workers = max(1, len(available) - 1) if workers is None else workers
        for w in range(workers):
            ours, theirs = context.Pipe()
            args = (theirs, str(root), pinned[w % len(pinned)])
            self.processes.append(context.Process(target=_serve, args=args, daemon=True))
            self.processes[-1].start()
            self.pipes.append(ours)
        self.pending = [deque() for _ in range(workers)]
        self.dealt = 0

    def close(self) -> None:
        """Stop the workers, each after its current job (at most a few seconds); answers not yet read are dropped."""
        self._stop()

    def _send(self, worker: int, function, *args) -> _Job:
        """Send a job to a worker, once it holds fewer than _PENDING."""
        while len(self.pending[worker]) >= _PENDING:
            self._receive(worker)
        job = _Job(self, worker)
        self.pending[worker].append(job)
        self.pipes[worker].send((function, args))
        return job

    def _receive(self, worker: int) -> None:
        """Wait for a worker's next answer, to its oldest job."""
        job = self.pending[worker].popleft()
        job.answer = self.pipes[worker].recv()
        job.answered = True

    def energy(self, ids, n, assemblies) -> Callable[[np.ndarray, np.ndarray], _Batch]:
        """Connectivity energy and its body-twist gradient, as in Atlas.energy."""
        if self.local is not None:

            def start(R0, o0) -> _Batch:
                poses = np.tile(np.eye(4), (*ids.shape, 1, 1))
                poses[..., :3, :3], poses[..., :3, 3] = R0, o0
                E, G = np.zeros(len(n)), np.zeros((*ids.shape, 6))
                for b, size in enumerate(n):
                    if size > 1:
                        E[b], G[b, :size] = self.local.energy((ids[b, :size], poses[b, :size]))
                batch = _Batch([], None, len(n), _row(ids.shape[1]))
                batch.values = E, G
                return batch

            return start
        _, assembly = np.unique(assemblies, return_inverse=True)
        owner = (assembly + self.dealt) % len(self.pipes)
        self.dealt += assembly.max() + 1
        rows = [np.flatnonzero(owner == w) for w in range(len(self.pipes))]
        row = _row(ids.shape[1])

        def start(R0, o0) -> _Batch:
            """Send a call's poses to the workers through a block of shared memory."""
            batch = _Batch(self.idle, self._block(len(n) * row.itemsize), len(n), row)
            poses = np.ndarray(batch.count, row, batch.block.buf)
            poses["pose"][..., :3, :3] = R0
            poses["pose"][..., :3, 3] = o0
            poses["pose"][..., 3, :] = (0.0, 0.0, 0.0, 1.0)  # a reused block holds another call's rows
            poses["n"] = n
            poses["ids"] = ids
            del poses
            for w, r in enumerate(rows):
                if len(r):
                    batch.jobs.append(self._send(w, _energies, batch.block.name, batch.count, ids.shape[1], r))
            return batch

        return start

    def energies(self, assemblies: list) -> list[tuple[float, np.ndarray]]:
        """E and dE/d(body twist [w, v]) (n, 6) of each assembly, as Atlas.energy gives them, one worker each."""
        arrays_of = [_arrays(assembly)[:2] for assembly in assemblies]
        if not arrays_of:
            return []
        n = np.array([len(part_ids) for part_ids, _ in arrays_of])
        ids = np.zeros((len(n), n.max()), np.uint32)
        poses = np.tile(np.eye(4), (len(n), n.max(), 1, 1))
        for b, (part_ids, at) in enumerate(arrays_of):
            ids[b, : n[b]] = canonical_id(part_ids)
            poses[b, : n[b]] = at
        E, G = self.energy(ids, n, np.arange(len(n)))(poses[..., :3, :3], poses[..., :3, 3])()
        return [(float(E[b]), G[b, : n[b]]) for b in range(len(n))]

    def _snapped(self, assemblies: list[tuple]) -> list[np.ndarray]:
        """Snapped poses, with one chunk at a time per worker and completed chunks in input order."""
        if self.local is not None:
            return [self.local.snap(assembly)[1] for assembly in assemblies]
        poses = [None] * len(assemblies)
        pending = iter(range(0, len(assemblies), 16))
        active = {}

        def _submit(worker: int) -> None:
            start = next(pending, None)
            if start is not None:
                job = self._send(worker, _snap_jobs, assemblies[start : start + 16])
                active[self.pipes[worker]] = (worker, start, job)

        for worker in range(len(self.pipes)):
            _submit(worker)
        while active:
            for pipe in wait(active):
                worker, start, job = active.pop(pipe)
                poses[start : start + 16] = job.result()
                _submit(worker)
        return poses

    def _block(self, size: int) -> SharedMemory:
        """Smallest idle block of at least `size` bytes, else a new one; idle blocks all smaller mean a new bucket of
        batches, so they are freed."""
        fits = [block for block in self.idle if block.size >= size]
        if fits:
            block = min(fits, key=lambda block: block.size)
            self.idle.remove(block)
            return block
        while self.idle:
            _free(self.idle.pop())
        return SharedMemory(create=True, size=size)
