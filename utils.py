"""Checkpointing, LR schedule, and logging. NumPy only — checkpoints are .npz."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from config import GPTConfig, TrainConfig, count_params, estimate_train_ram_mb
from model import Param, TinyGPT


def configure_runtime(train: TrainConfig) -> np.random.Generator:
    random.seed(train.seed)
    np.random.seed(train.seed)
    return np.random.default_rng(train.seed)


def cosine_lr(step: int, t: TrainConfig) -> float:
    if step < t.warmup_steps:
        return t.learning_rate * (step + 1) / max(1, t.warmup_steps)
    if step >= t.max_steps:
        return t.min_lr
    progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return t.min_lr + coeff * (t.learning_rate - t.min_lr)


def save_ckpt(
    path: str | Path,
    model: TinyGPT,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, p in model.named_params():
        arrays[f"w:{name}"] = p.data
        arrays[f"m:{name}"] = p.m
        arrays[f"v:{name}"] = p.v
    # numpy savez appends .npz unless the name already ends with it
    tmp = path.with_name(path.name + ".saving.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    meta = {
        "config": model.config.to_dict(),
        "step": step,
        "adam_step": model.adam_step,
        "extra": extra or {},
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_ckpt(path: str | Path) -> tuple[TinyGPT, dict[str, Any]]:
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing sidecar {meta_path} (config json next to the .npz)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = GPTConfig(**meta["config"])
    model = TinyGPT(cfg, seed=0)
    blob = np.load(path)
    by_name = dict(model.named_params())
    for key in blob.files:
        kind, _, name = key.partition(":")
        if name not in by_name:
            continue
        p: Param = by_name[name]
        arr = np.array(blob[key], dtype=np.float32)
        if kind == "w":
            p.data[...] = arr
        elif kind == "m":
            p.m[...] = arr
        elif kind == "v":
            p.v[...] = arr
    model.adam_step = int(meta.get("adam_step", meta.get("step", 0)))
    return model, meta


def print_banner(preset: str, model: TinyGPT, train: TrainConfig) -> None:
    cfg = model.config
    n = model.n_params()
    ram = estimate_train_ram_mb(cfg, train)
    print("=" * 56)
    print(f" TinyGPT  (NumPy)  preset={preset}  device=cpu")
    print(f" layers={cfg.n_layer}  heads={cfg.n_head}  d={cfg.n_embd}  T={cfg.block_size}  V={cfg.vocab_size}")
    print(f" parameters={n:,}  (~{n/1e6:.2f}M)   est. train RAM={ram} MB")
    print(f" batch={train.batch_size} x accum={train.grad_accum}  lr={train.learning_rate}  steps={train.max_steps}")
    print(f" counted={count_params(cfg):,} (formula)  grad_ckpt={train.grad_checkpoint}")
    print("=" * 56)


def ppl(loss: float) -> float:
    try:
        return float(math.exp(min(loss, 20)))
    except OverflowError:
        return float("inf")
