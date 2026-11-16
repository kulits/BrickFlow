"""Packed attention and residual normalization kernels."""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_SCALE = tl.constexpr(0.125)  # 1 / sqrt(64)
_SCALE_LOG2 = tl.constexpr(0.125 * 1.4426950408889634)  # softmax in base 2
_PAIRS = tl.constexpr(4)  # attention: (row, head) pairs per program
_TILE = tl.constexpr(128)  # the largest attention tile: a row of up to 128 tokens; the log-sum-exp row stride
_TOKENS_FORWARD = tl.constexpr(4)  # residual norm: tokens per program
_TOKENS_BACKWARD = tl.constexpr(2)  # tokens per iteration of a row's program, with a warp per 512 columns


@triton.jit
def _tile(bh, H, N, tokens):
    """Offsets of the (BLOCK, 64) tile of (row, head) pair bh."""
    return ((bh // H * N + tokens[:, None]) * H + bh % H) * 64 + tl.arange(0, 64)[None, :]


@triton.jit
def _same_sample(SPLIT, bh, H, N, tokens):
    """(BLOCK, BLOCK) mask of pair bh: query i may read key j."""
    first = tokens < tl.load(SPLIT + bh // H)
    return (tokens[None, :] < N) & (first[:, None] == first[None, :])


@triton.jit
def _forward(Q, K, V, OUT, LSE, SPLIT, N, H, BH, BLOCK: tl.constexpr):
    """Attention of a row's tokens within each sample, and its log-sum-exp."""
    tokens = tl.arange(0, BLOCK)
    live = tokens[:, None] < N
    start = tl.program_id(0) * _PAIRS
    for bh in tl.range(start, tl.minimum(start + _PAIRS, BH)):
        at = _tile(bh, H, N, tokens)
        q = tl.load(Q + at, mask=live, other=0.0)
        k = tl.load(K + at, mask=live, other=0.0)
        v = tl.load(V + at, mask=live, other=0.0)
        s = tl.dot(q, tl.trans(k)) * _SCALE_LOG2
        s = tl.where(_same_sample(SPLIT, bh, H, N, tokens), s, float("-inf"))
        row_max = tl.max(s, axis=1)
        p = tl.exp2(s - row_max[:, None])
        row_sum = tl.sum(p, axis=1)
        o = tl.dot(p.to(v.dtype), v) / row_sum[:, None]
        tl.store(OUT + at, o.to(OUT.dtype.element_ty), mask=live)
        tl.store(LSE + bh * _TILE + tokens, row_max + tl.log2(row_sum))


@triton.jit
def _backward(Q, K, V, OUT, DO, LSE, SPLIT, DQ, DK, DV, N, H, BH, BLOCK: tl.constexpr):
    """Gradients of the packed attention."""
    tokens = tl.arange(0, BLOCK)
    live = tokens[:, None] < N
    start = tl.program_id(0) * _PAIRS
    for bh in tl.range(start, tl.minimum(start + _PAIRS, BH)):
        at = _tile(bh, H, N, tokens)
        same_sample = _same_sample(SPLIT, bh, H, N, tokens)  # before the loads: the faster schedule here
        q = tl.load(Q + at, mask=live, other=0.0)
        k = tl.load(K + at, mask=live, other=0.0)
        v = tl.load(V + at, mask=live, other=0.0)
        o = tl.load(OUT + at, mask=live, other=0.0)
        do = tl.load(DO + at, mask=live, other=0.0)
        lse = tl.load(LSE + bh * _TILE + tokens)
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        s = tl.dot(q, tl.trans(k)) * _SCALE_LOG2
        p = tl.where(live & same_sample, tl.exp2(s - lse[:, None]), 0.0)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        ds = (p * (tl.dot(do.to(v.dtype), tl.trans(v)) - delta[:, None])).to(q.dtype)
        tl.store(DQ + at, (tl.dot(ds, k) * _SCALE).to(DQ.dtype.element_ty), mask=live)
        tl.store(DK + at, (tl.dot(tl.trans(ds), q) * _SCALE).to(DK.dtype.element_ty), mask=live)
        tl.store(DV + at, dv.to(DV.dtype.element_ty), mask=live)


def _launch(kernel, block, q, *args, num_warps):
    """Launch a kernel over the (row, head) pairs of q (B, N, H, 64), _PAIRS per program."""
    batch, n, heads, _ = q.shape
    grid = (triton.cdiv(batch * heads, _PAIRS.value),)
    kernel[grid](q, *args, n, heads, batch * heads, BLOCK=block, num_warps=num_warps, num_stages=2)


def _block(n: int) -> int:
    """Attention tile for rows of n tokens: a row is one tile."""
    return max(16, triton.next_power_of_2(n))


@torch.library.custom_op("brickflow::packed_attention", mutates_args=())
def _attention(q: Tensor, k: Tensor, v: Tensor, split: Tensor) -> tuple[Tensor, Tensor]:
    """Packed attention and its log-sum-exp; opaque to the compiler, so the tile follows each call's row size."""
    out = torch.empty_like(q)
    lse = q.new_empty(q.shape[0] * q.shape[2], _TILE.value, dtype=torch.float32)
    _launch(_forward, _block(q.shape[1]), q, k, v, out, lse, split, num_warps=4)
    return out, lse


@_attention.register_fake
def _attention_fake(q, k, v, split):
    """Shapes of the packed attention's outputs."""
    return torch.empty_like(q), q.new_empty(q.shape[0] * q.shape[2], _TILE.value, dtype=torch.float32)


@torch.library.custom_op("brickflow::packed_attention_backward", mutates_args=())
def _attention_backward(
    grad: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, split: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients of the packed attention with respect to q, k, and v."""
    dq = torch.empty_like(q)
    dk = torch.empty_like(q)
    dv = torch.empty_like(q)
    _launch(_backward, _block(q.shape[1]), q, k, v, out, grad, lse, split, dq, dk, dv, num_warps=8)
    return dq, dk, dv


@_attention_backward.register_fake
def _attention_backward_fake(grad, q, k, v, out, lse, split):
    """Shapes of the gradients."""
    return torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)


def _save_attention(ctx, inputs, output):
    """Keep the attention's inputs and outputs for its backward."""
    ctx.save_for_backward(*inputs, *output)


def _attention_gradient(ctx, grad, _):
    """Gradients of the packed attention for autograd."""
    q, k, v, split, out, lse = ctx.saved_tensors
    return *_attention_backward(grad.to(out.dtype).contiguous(), q, k, v, out, lse, split), None


_attention.register_autograd(_attention_gradient, setup_context=_save_attention)


def packed_attention(q, k, v, split) -> torch.Tensor:
    """Packed attention for rows of up to 128 tokens."""
    return _attention(q.contiguous(), k.contiguous(), v.contiguous(), split.to(torch.int32).contiguous())[0]


@triton.jit
def _norm_forward(X, Y, GATE, SHIFT, SCALE, XO, H, MU, RSTD, N, W, GATE_STRIDE, STRIDE, BLOCK: tl.constexpr):
    """x + gate y and its modulated LayerNorm, with the mean and inverse std."""
    b = tl.program_id(0)
    tokens = tl.program_id(1) * _TOKENS_FORWARD + tl.arange(0, _TOKENS_FORWARD)
    columns = tl.arange(0, BLOCK)
    mask = (tokens[:, None] < N) & (columns[None, :] < W)
    at = (b * N + tokens[:, None]) * W + columns[None, :]
    gate = tl.load(GATE + b * GATE_STRIDE + columns, mask=columns < W, other=0.0).to(tl.float32)
    shift = tl.load(SHIFT + b * STRIDE + columns, mask=columns < W, other=0.0).to(tl.float32)
    scale = tl.load(SCALE + b * STRIDE + columns, mask=columns < W, other=0.0).to(tl.float32)
    y = tl.load(Y + at, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + at, mask=mask, other=0.0) + gate[None, :] * y
    tl.store(XO + at, x, mask=mask)
    mean = tl.sum(x, axis=1) / W
    centered = tl.where(mask, x - mean[:, None], 0.0)
    rstd = 1 / tl.sqrt(tl.sum(centered * centered, axis=1) / W + 1e-6)
    h = centered * rstd[:, None] * (1 + scale[None, :]) + shift[None, :]
    tl.store(H + at, h.to(H.dtype.element_ty), mask=mask)
    tl.store(MU + b * N + tokens, mean, mask=tokens < N)
    tl.store(RSTD + b * N + tokens, rstd, mask=tokens < N)


@triton.jit
def _norm_backward(
    XO, Y, GATE, SCALE, MU, RSTD, DH, DXO, DX, DY, DGATE, DSHIFT, DSCALE, N, W, GATE_STRIDE, STRIDE, BLOCK: tl.constexpr
):
    """Gradients of the residual norm."""
    b = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    gate = tl.load(GATE + b * GATE_STRIDE + columns, mask=columns < W, other=0.0).to(tl.float32)
    scale = tl.load(SCALE + b * STRIDE + columns, mask=columns < W, other=0.0).to(tl.float32)
    dgate = tl.zeros([BLOCK], dtype=tl.float32)
    dshift = tl.zeros([BLOCK], dtype=tl.float32)
    dscale = tl.zeros([BLOCK], dtype=tl.float32)
    for start in tl.range(0, N, _TOKENS_BACKWARD):
        tokens = start + tl.arange(0, _TOKENS_BACKWARD)
        mask = (tokens[:, None] < N) & (columns[None, :] < W)
        at = (b * N + tokens[:, None]) * W + columns[None, :]
        mean = tl.load(MU + b * N + tokens, mask=tokens < N, other=0.0)
        rstd = tl.load(RSTD + b * N + tokens, mask=tokens < N, other=0.0)
        x = tl.load(XO + at, mask=mask, other=0.0)
        normed = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        dh = tl.load(DH + at, mask=mask, other=0.0).to(tl.float32)
        dshift += tl.sum(dh, axis=0)
        dscale += tl.sum(dh * normed, axis=0)
        dnormed = dh * (1 + scale[None, :])
        mean_dnormed = tl.sum(dnormed, axis=1) / W
        mean_product = tl.sum(dnormed * normed, axis=1) / W
        dx = rstd[:, None] * (dnormed - mean_dnormed[:, None] - normed * mean_product[:, None])
        dx += tl.load(DXO + at, mask=mask, other=0.0)
        y = tl.load(Y + at, mask=mask, other=0.0).to(tl.float32)
        dgate += tl.sum(dx * y, axis=0)
        tl.store(DX + at, dx, mask=mask)
        tl.store(DY + at, (dx * gate[None, :]).to(DY.dtype.element_ty), mask=mask)
    tl.store(DGATE + b * W + columns, dgate, mask=columns < W)
    tl.store(DSHIFT + b * W + columns, dshift, mask=columns < W)
    tl.store(DSCALE + b * W + columns, dscale, mask=columns < W)


@triton_op("brickflow::residual_norm", mutates_args=())
def _residual_norm(
    x: Tensor, y: Tensor, gate: Tensor, shift: Tensor, scale: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x + gate y, its modulated LayerNorm, and the mean and inverse std."""
    batch, n, width = x.shape
    x_out = torch.empty_like(x)
    h = torch.empty_like(y)
    mean = x.new_empty(batch, n)
    rstd = x.new_empty(batch, n)
    block = triton.next_power_of_2(width)
    wrap_triton(_norm_forward)[(batch, triton.cdiv(n, _TOKENS_FORWARD.value))](
        x, y, gate, shift, scale, x_out, h, mean, rstd, n, width, gate.stride(0), shift.stride(0), BLOCK=block
    )
    return x_out, h, mean, rstd


@triton_op("brickflow::residual_norm_backward", mutates_args=())
def _residual_norm_backward(
    x_out: Tensor, y: Tensor, gate: Tensor, scale: Tensor, mean: Tensor, rstd: Tensor, dh: Tensor, dx_out: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gradients of the residual norm with respect to x, y, gate, shift, and scale."""
    batch, n, width = x_out.shape
    dx = torch.empty_like(x_out)
    dy = torch.empty_like(y)
    dgate = x_out.new_empty(batch, 1, width)
    dshift = x_out.new_empty(batch, 1, width)
    dscale = x_out.new_empty(batch, 1, width)
    tensors = (x_out, y, gate, scale, mean, rstd, dh, dx_out, dx, dy, dgate, dshift, dscale)
    block = triton.next_power_of_2(int(width))  # an int: it sets the warps
    wrap_triton(_norm_backward)[(batch,)](
        *tensors, n, width, gate.stride(0), scale.stride(0), BLOCK=block, num_warps=max(1, block // 512)
    )
    return dx, dy, dgate, dshift, dscale


def _save_norm(ctx, inputs, output):
    """Keep what the residual norm's backward needs."""
    _, y, gate, _, scale = inputs
    x_out, _, mean, rstd = output
    ctx.save_for_backward(x_out, y, gate, scale, mean, rstd)


def _norm_gradient(ctx, dx_out, dh, _mean, _rstd):
    """Gradients of x, y, gate, shift, and scale; the engine casts the last three to their inputs' dtypes."""
    return _residual_norm_backward(*ctx.saved_tensors, dh.contiguous(), dx_out.contiguous())


_residual_norm.register_autograd(_norm_gradient, setup_context=_save_norm)


def residual_norm(x, y, gate, shift, scale) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual and normalization with gradients: one pass forward and one backward."""
    if shift.stride(0) != scale.stride(0) or not gate.stride(-1) == shift.stride(-1) == 1:
        raise ValueError("shift and scale must share their row stride, and gate and shift be contiguous in a row")
    return _residual_norm(x.contiguous(), y.contiguous(), gate, shift, scale)[:2]
