"""Causal GPT in pure NumPy — self-attention, LayerNorm, MLP, embeddings.

No PyTorch, no JAX. Forward + manual backward + AdamW. Optional block-level
gradient checkpointing for phones. Weights are float32; matmuls go through
NumPy (OpenBLAS when Termux has it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import GPTConfig

Array = np.ndarray
EPS = 1e-5
ADAM_EPS = 1e-8


@dataclass
class Param:
    data: Array
    grad: Array
    m: Array
    v: Array
    decay: bool = True

    @staticmethod
    def zeros(shape: tuple[int, ...], decay: bool = True) -> "Param":
        d = np.zeros(shape, dtype=np.float32)
        z = np.zeros_like(d)
        return Param(d, z.copy(), z.copy(), z.copy(), decay)

    @staticmethod
    def randn(
        shape: tuple[int, ...],
        rng: np.random.Generator,
        std: float = 0.02,
        decay: bool = True,
    ) -> "Param":
        d = (rng.standard_normal(shape, dtype=np.float32) * np.float32(std)).astype(np.float32)
        z = np.zeros_like(d)
        return Param(d, z.copy(), z.copy(), z.copy(), decay)


def _gelu(x: Array) -> Array:
    # tanh approximation of GELU (Hendrycks & Gimpel)
    s = np.float32(math.sqrt(2.0 / math.pi))
    u = s * (x + np.float32(0.044715) * x * x * x)
    return np.float32(0.5) * x * (1.0 + np.tanh(u))


def _gelu_bwd(x: Array, dy: Array) -> Array:
    s = np.float32(math.sqrt(2.0 / math.pi))
    k = np.float32(0.044715)
    u = s * (x + k * x * x * x)
    t = np.tanh(u)
    sech2 = 1.0 - t * t
    du = s * (1.0 + np.float32(3.0) * k * x * x)
    return dy * (np.float32(0.5) * (1.0 + t) + np.float32(0.5) * x * sech2 * du)


def _layernorm_fwd(x: Array, gain: Array, bias: Array) -> tuple[Array, tuple]:
    mean = x.mean(axis=-1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xhat = xc * inv
    y = xhat * gain + bias
    return y.astype(np.float32), (xhat, inv, gain)


def _layernorm_bwd(dy: Array, cache: tuple, dg: Array, db: Array) -> Array:
    xhat, inv, gain = cache
    n = xhat.shape[-1]
    db += dy.sum(axis=tuple(range(dy.ndim - 1)))
    dg += (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * gain
    sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
    sum_dxhat_xhat = (dxhat * xhat).sum(axis=-1, keepdims=True)
    dx = (dxhat - xhat * sum_dxhat_xhat / n - sum_dxhat / n) * inv
    return dx.astype(np.float32)


def _linear_fwd(x: Array, w: Array, b: Optional[Array]) -> Array:
    y = x @ w.T
    if b is not None:
        y = y + b
    return y.astype(np.float32)


def _linear_bwd(dy: Array, x: Array, w: Param, b: Optional[Param]) -> Array:
    xf = x.reshape(-1, x.shape[-1])
    dyf = dy.reshape(-1, dy.shape[-1])
    w.grad += dyf.T @ xf
    if b is not None:
        b.grad += dyf.sum(axis=0)
    return (dy @ w.data).astype(np.float32)


def _softmax_last(x: Array) -> Array:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class CausalSelfAttention:
    def __init__(self, config: GPTConfig, rng: np.random.Generator, proj_std: float) -> None:
        D = config.n_embd
        self.n_head = config.n_head
        self.head_dim = D // config.n_head
        self.scale = np.float32(1.0 / math.sqrt(self.head_dim))
        self.dropout_p = config.dropout
        self.c_attn_w = Param.randn((3 * D, D), rng)
        self.c_attn_b = Param.zeros((3 * D,), decay=False) if config.bias else None
        self.c_proj_w = Param.randn((D, D), rng, std=proj_std)
        self.c_proj_b = Param.zeros((D,), decay=False) if config.bias else None
        self.mask = np.tril(np.ones((config.block_size, config.block_size), dtype=np.float32))
        self._cache: tuple | None = None

    def params(self) -> list[tuple[str, Param]]:
        out: list[tuple[str, Param]] = [("c_attn.weight", self.c_attn_w), ("c_proj.weight", self.c_proj_w)]
        if self.c_attn_b is not None:
            out.append(("c_attn.bias", self.c_attn_b))
        if self.c_proj_b is not None:
            out.append(("c_proj.bias", self.c_proj_b))
        return out

    def forward(self, x: Array, train: bool, rng: np.random.Generator, save: bool) -> Array:
        b, t, c = x.shape
        qkv = _linear_fwd(x, self.c_attn_w.data, None if self.c_attn_b is None else self.c_attn_b.data)
        qkv = qkv.reshape(b, t, 3, self.n_head, self.head_dim)
        q = np.transpose(qkv[:, :, 0], (0, 2, 1, 3))  # B,H,T,hs
        k = np.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = np.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        att = (q @ np.swapaxes(k, -1, -2)) * self.scale
        att = np.where(self.mask[:t, :t] > 0, att, np.float32(-1e10))
        att_pre = _softmax_last(att)
        att_used = att_pre
        drop_mask = None
        if train and self.dropout_p > 0:
            drop_mask = (rng.random(att_pre.shape, dtype=np.float32) >= self.dropout_p).astype(np.float32) / np.float32(
                1.0 - self.dropout_p
            )
            att_used = att_pre * drop_mask
        y = att_used @ v  # B,H,T,hs
        y = np.transpose(y, (0, 2, 1, 3)).reshape(b, t, c)
        out = _linear_fwd(y, self.c_proj_w.data, None if self.c_proj_b is None else self.c_proj_b.data)
        if save:
            self._cache = (x, q, k, v, att_pre, att_used, drop_mask, y)
        return out

    def backward(self, dy: Array) -> Array:
        x, q, k, v, att_pre, att_used, drop_mask, y = self._cache  # type: ignore[misc]
        b, t, c = x.shape
        dy_y = _linear_bwd(dy, y, self.c_proj_w, self.c_proj_b)
        dy_y = dy_y.reshape(b, t, self.n_head, self.head_dim)
        dy_heads = np.transpose(dy_y, (0, 2, 1, 3))  # B,H,T,hs
        datt_used = dy_heads @ np.swapaxes(v, -1, -2)
        dv = np.swapaxes(att_used, -1, -2) @ dy_heads
        datt = datt_used if drop_mask is None else datt_used * drop_mask
        sum_datt = (datt * att_pre).sum(axis=-1, keepdims=True)
        dpre = att_pre * (datt - sum_datt)
        dpre = dpre * self.scale
        dq = dpre @ k
        dk = np.swapaxes(dpre, -1, -2) @ q
        dqkv = np.zeros((b, t, 3, self.n_head, self.head_dim), dtype=np.float32)
        dqkv[:, :, 0] = np.transpose(dq, (0, 2, 1, 3))
        dqkv[:, :, 1] = np.transpose(dk, (0, 2, 1, 3))
        dqkv[:, :, 2] = np.transpose(dv, (0, 2, 1, 3))
        dqkv_flat = dqkv.reshape(b, t, 3 * c)
        return _linear_bwd(dqkv_flat, x, self.c_attn_w, self.c_attn_b)


class MLP:
    def __init__(self, config: GPTConfig, rng: np.random.Generator, proj_std: float) -> None:
        D = config.n_embd
        self.dropout_p = config.dropout
        self.fc_w = Param.randn((4 * D, D), rng)
        self.fc_b = Param.zeros((4 * D,), decay=False) if config.bias else None
        self.proj_w = Param.randn((D, 4 * D), rng, std=proj_std)
        self.proj_b = Param.zeros((D,), decay=False) if config.bias else None
        self._cache: tuple | None = None

    def params(self) -> list[tuple[str, Param]]:
        out = [("fc.weight", self.fc_w), ("proj.weight", self.proj_w)]
        if self.fc_b is not None:
            out.append(("fc.bias", self.fc_b))
        if self.proj_b is not None:
            out.append(("proj.bias", self.proj_b))
        return out

    def forward(self, x: Array, train: bool, rng: np.random.Generator, save: bool) -> Array:
        h = _linear_fwd(x, self.fc_w.data, None if self.fc_b is None else self.fc_b.data)
        a = _gelu(h)
        y = _linear_fwd(a, self.proj_w.data, None if self.proj_b is None else self.proj_b.data)
        drop = None
        if train and self.dropout_p > 0:
            drop = (rng.random(y.shape, dtype=np.float32) >= self.dropout_p).astype(np.float32) / np.float32(
                1.0 - self.dropout_p
            )
            y = y * drop
        if save:
            self._cache = (x, h, a, drop)
        return y

    def backward(self, dy: Array) -> Array:
        x, h, a, drop = self._cache  # type: ignore[misc]
        if drop is not None:
            dy = dy * drop
        da = _linear_bwd(dy, a, self.proj_w, self.proj_b)
        dh = _gelu_bwd(h, da)
        return _linear_bwd(dh, x, self.fc_w, self.fc_b)


class Block:
    def __init__(self, config: GPTConfig, rng: np.random.Generator, proj_std: float) -> None:
        D = config.n_embd
        self.ln1_g = Param.zeros((D,), decay=False)
        self.ln1_g.data.fill(1.0)
        self.ln1_b = Param.zeros((D,), decay=False)
        self.attn = CausalSelfAttention(config, rng, proj_std)
        self.ln2_g = Param.zeros((D,), decay=False)
        self.ln2_g.data.fill(1.0)
        self.ln2_b = Param.zeros((D,), decay=False)
        self.mlp = MLP(config, rng, proj_std)
        self._cache: tuple | None = None

    def params(self) -> list[tuple[str, Param]]:
        out = [
            ("ln_1.weight", self.ln1_g),
            ("ln_1.bias", self.ln1_b),
            ("ln_2.weight", self.ln2_g),
            ("ln_2.bias", self.ln2_b),
        ]
        out += [(f"attn.{k}", p) for k, p in self.attn.params()]
        out += [(f"mlp.{k}", p) for k, p in self.mlp.params()]
        return out

    def forward(self, x: Array, train: bool, rng: np.random.Generator, save: bool) -> Array:
        h1, ln1_c = _layernorm_fwd(x, self.ln1_g.data, self.ln1_b.data)
        a = self.attn.forward(h1, train, rng, save)
        x2 = x + a
        h2, ln2_c = _layernorm_fwd(x2, self.ln2_g.data, self.ln2_b.data)
        m = self.mlp.forward(h2, train, rng, save)
        out = x2 + m
        if save:
            self._cache = (ln1_c, ln2_c, x, x2)
        return out

    def backward(self, dy: Array) -> Array:
        ln1_c, ln2_c, x, x2 = self._cache  # type: ignore[misc]
        dx2 = dy
        dm = dy
        dh2 = self.mlp.backward(dm)
        dx2 = dx2 + _layernorm_bwd(dh2, ln2_c, self.ln2_g.grad, self.ln2_b.grad)
        dx = dx2
        da = dx2
        dh1 = self.attn.backward(da)
        dx = dx + _layernorm_bwd(dh1, ln1_c, self.ln1_g.grad, self.ln1_b.grad)
        return dx


class TinyGPT:
    def __init__(self, config: GPTConfig, seed: int = 1337) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.grad_ckpt = False
        D, T, V, L = config.n_embd, config.block_size, config.vocab_size, config.n_layer
        proj_std = 0.02 / math.sqrt(2 * L)
        self.wte = Param.randn((V, D), self.rng)
        self.wpe = Param.randn((T, D), self.rng)
        self.blocks = [Block(config, self.rng, proj_std) for _ in range(L)]
        self.ln_f_g = Param.zeros((D,), decay=False)
        self.ln_f_g.data.fill(1.0)
        self.ln_f_b = Param.zeros((D,), decay=False)
        self._fwd: dict = {}
        self.adam_step = 0

    def named_params(self) -> list[tuple[str, Param]]:
        out: list[tuple[str, Param]] = [
            ("wte.weight", self.wte),
            ("wpe.weight", self.wpe),
            ("ln_f.weight", self.ln_f_g),
            ("ln_f.bias", self.ln_f_b),
        ]
        for i, blk in enumerate(self.blocks):
            out += [(f"h.{i}.{k}", p) for k, p in blk.params()]
        return out

    def n_params(self) -> int:
        # wte is listed once and reused as the tied lm_head
        return int(sum(p.data.size for _, p in self.named_params()))

    def enable_grad_checkpoint(self, enabled: bool = True) -> None:
        self.grad_ckpt = enabled

    def zero_grad(self) -> None:
        for _, p in self.named_params():
            p.grad.fill(0)

    def clip_grad(self, max_norm: float) -> float:
        total = 0.0
        params = self.named_params()
        for _, p in params:
            total += float(np.square(p.grad, dtype=np.float64).sum())
        norm = math.sqrt(total)
        if max_norm > 0 and norm > max_norm:
            scale = np.float32(max_norm / (norm + 1e-6))
            for _, p in params:
                p.grad *= scale
        return norm

    def adamw(self, lr: float, betas: tuple[float, float], wd: float, step: int) -> None:
        b1, b2 = betas
        self.adam_step = step
        bc1 = np.float32(1.0 - b1**step)
        bc2 = np.float32(1.0 - b2**step)
        lr32 = np.float32(lr)
        decay = np.float32(1.0 - lr * wd) if wd else None
        b1_32, b2_32 = np.float32(b1), np.float32(b2)
        omb1, omb2 = np.float32(1.0 - b1), np.float32(1.0 - b2)
        for _, p in self.named_params():
            g = p.grad
            p.m *= b1_32
            p.m += omb1 * g
            p.v *= b2_32
            p.v += omb2 * g * g
            mhat = p.m / bc1
            vhat = p.v / bc2
            if p.decay and decay is not None:
                p.data *= decay
            p.data -= lr32 * mhat / (np.sqrt(vhat) + np.float32(ADAM_EPS))

    def forward(self, idx: Array, targets: Optional[Array] = None) -> tuple[Array, Optional[float]]:
        if idx.dtype != np.int64 and idx.dtype != np.int32:
            idx = idx.astype(np.int32)
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} > block_size {self.config.block_size}")
        save = self.training and targets is not None
        tok = self.wte.data[idx]  # B,T,C
        pos = self.wpe.data[np.arange(t, dtype=np.int32)]
        x = (tok + pos).astype(np.float32)
        ckpt_xs: list[Array] = []
        for blk in self.blocks:
            if save and self.grad_ckpt:
                ckpt_xs.append(x)
                x = blk.forward(x, self.training, self.rng, save=False)
            else:
                x = blk.forward(x, self.training, self.rng, save=save)
        x_f, ln_c = _layernorm_fwd(x, self.ln_f_g.data, self.ln_f_b.data)
        # tied lm head
        logits = x_f @ self.wte.data.T
        loss = None
        if targets is not None:
            loss, ce_cache = _cross_entropy(logits, targets)
            if save:
                self._fwd = {
                    "idx": idx,
                    "x_f": x_f,
                    "ln_c": ln_c,
                    "x_pre_ln": x,
                    "logits": logits,
                    "targets": targets,
                    "ce": ce_cache,
                    "ckpt_xs": ckpt_xs,
                }
        return logits, loss

    def backward(self) -> None:
        c = self._fwd
        idx: Array = c["idx"]
        targets: Array = c["targets"]
        dlogits = _cross_entropy_bwd(c["logits"], targets, c["ce"])
        # logits = x_f @ wte.T  →  dx_f = dlogits @ wte ; dwte += dlogits^T @ x_f
        x_f: Array = c["x_f"]
        xf = x_f.reshape(-1, x_f.shape[-1])
        dlf = dlogits.reshape(-1, dlogits.shape[-1])
        self.wte.grad += dlf.T @ xf
        dx_f = dlogits @ self.wte.data
        dx = _layernorm_bwd(dx_f, c["ln_c"], self.ln_f_g.grad, self.ln_f_b.grad)
        ckpt_xs: list[Array] = c["ckpt_xs"]
        if self.grad_ckpt and ckpt_xs:
            for i in range(len(self.blocks) - 1, -1, -1):
                x_in = ckpt_xs[i]
                self.blocks[i].forward(x_in, True, self.rng, save=True)
                dx = self.blocks[i].backward(dx)
        else:
            for blk in reversed(self.blocks):
                dx = blk.backward(dx)
        # embeddings
        t = idx.shape[1]
        self.wpe.grad[:t] += dx.sum(axis=0)
        np.add.at(self.wte.grad, idx, dx)

    def generate(
        self,
        idx: Array,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
    ) -> Array:
        was = self.training
        self.training = False
        out = idx.astype(np.int32)
        for _ in range(max_new_tokens):
            cond = out if out.shape[1] <= self.config.block_size else out[:, -self.config.block_size :]
            logits, _ = self.forward(cond, None)
            logits = logits[:, -1, :].astype(np.float64)
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                thresh = np.partition(logits, -k, axis=-1)[:, -k]
                logits = np.where(logits < thresh[:, None], -1e10, logits)
            logits = logits - logits.max(axis=-1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=-1, keepdims=True)
            next_id = np.array([[self.rng.choice(p.shape[1], p=p[0])]], dtype=np.int32)
            out = np.concatenate([out, next_id], axis=1)
            if eos_id is not None and int(next_id[0, 0]) == eos_id:
                break
        self.training = was
        return out


def _cross_entropy(logits: Array, targets: Array) -> tuple[float, tuple]:
    """Mean CE over tokens whose target != -1."""
    b, t, v = logits.shape
    flat = logits.reshape(-1, v)
    tgt = targets.reshape(-1)
    valid = tgt >= 0
    n = int(valid.sum())
    if n == 0:
        return 0.0, (valid, n)
    z = flat - flat.max(axis=-1, keepdims=True)
    log_z = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    safe_tgt = np.where(valid, tgt, 0)
    nll = -log_z[np.arange(flat.shape[0]), safe_tgt]
    nll = np.where(valid, nll, 0.0)
    return float(nll.sum() / n), (valid, n)


def _cross_entropy_bwd(logits: Array, targets: Array, cache: tuple) -> Array:
    valid, n = cache
    b, t, v = logits.shape
    flat = logits.reshape(-1, v)
    tgt = targets.reshape(-1)
    z = flat - flat.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    d = p
    rows = np.arange(flat.shape[0])
    safe_tgt = np.where(valid, tgt, 0)
    d[rows, safe_tgt] -= 1.0
    d = np.where(valid[:, None], d, 0.0)
    if n > 0:
        d /= n
    return d.reshape(b, t, v).astype(np.float32)
