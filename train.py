#!/usr/bin/env python3
"""Next-token pretraining (pure NumPy).

    python train.py --preset phone --max-steps 400
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from runtime import setup

setup()

from config import get_preset, with_overrides  # noqa: E402
from dataset import load_pretrain  # noqa: E402
from model import TinyGPT  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from utils import (  # noqa: E402
    configure_runtime,
    cosine_lr,
    load_ckpt,
    ppl,
    print_banner,
    save_ckpt,
)


def estimate_loss(model: TinyGPT, split, batches: int, rng, bs: int) -> float:
    was = model.training
    model.training = False
    total = 0.0
    for _ in range(batches):
        x, y = split.sample(bs, rng)
        _, loss = model.forward(x, y)
        total += float(loss or 0.0)
    model.training = was
    return total / max(1, batches)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain TinyGPT on a markdown corpus (NumPy)")
    p.add_argument("--preset", default="phone", choices=["nano", "phone", "tiny", "compact"])
    p.add_argument("--data", default="data/pretrain.md")
    p.add_argument("--out", default="out")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-ckpt", action="store_true", help="recompute attention/MLP in backward (saves RAM)")
    p.add_argument("--resume", default=None, help="path to a .npz checkpoint")
    p.add_argument("--threads", type=int, default=None, help="BLAS threads (or TINYGPT_THREADS)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        setup(args.threads)

    gpt_cfg, train_cfg = get_preset(args.preset)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
        train_cfg.min_lr = args.lr / 10
    if args.grad_ckpt:
        train_cfg.grad_checkpoint = True
    if args.seed is not None:
        train_cfg.seed = args.seed

    rng = configure_runtime(train_cfg)
    tok = Tokenizer()
    gpt_cfg = with_overrides(gpt_cfg, vocab_size=tok.vocab_size)

    train_buf, val_buf = load_pretrain(args.data, tok, gpt_cfg.block_size)
    print(f"tokens  train={len(train_buf):,}  val={len(val_buf):,}")

    if args.resume:
        model, blob = load_ckpt(args.resume)
        start_step = int(blob.get("step", 0))
    else:
        model = TinyGPT(gpt_cfg, seed=train_cfg.seed)
        start_step = 0
    model.enable_grad_checkpoint(train_cfg.grad_checkpoint)
    model.training = True
    print_banner(args.preset, model, train_cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    t0 = time.time()

    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, train_cfg)
        model.zero_grad()
        loss_acc = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = train_buf.sample(train_cfg.batch_size, rng)
            _, loss = model.forward(x, y)
            model.backward()
            loss_acc += float(loss or 0.0)
        # grads summed over accum steps — scale to mean
        if train_cfg.grad_accum > 1:
            scale = 1.0 / train_cfg.grad_accum
            for _, p in model.named_params():
                p.grad *= scale
        model.clip_grad(train_cfg.grad_clip)
        model.adamw(lr, train_cfg.betas, train_cfg.weight_decay, step + 1)

        if step % 10 == 0 or step == train_cfg.max_steps - 1:
            dt = time.time() - t0
            toks = train_cfg.batch_size * gpt_cfg.block_size * train_cfg.grad_accum
            tps = toks / max(dt, 1e-6)
            print(
                f"step {step:5d}/{train_cfg.max_steps}  "
                f"loss={loss_acc / train_cfg.grad_accum:.4f}  lr={lr:.2e}  {tps:.0f} tok/s"
            )
            t0 = time.time()

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps - 1:
            val = estimate_loss(model, val_buf, train_cfg.eval_batches, rng, train_cfg.batch_size)
            print(f"  eval  val_loss={val:.4f}  ppl={ppl(val):.1f}")
            save_ckpt(out / "last.npz", model, step, {"val_loss": val, "preset": args.preset})
            if val < best:
                best = val
                save_ckpt(out / "best.npz", model, step, {"val_loss": val, "preset": args.preset})
                print("  saved out/best.npz")

    print(f"done. best val_loss={best:.4f}  ckpt={out / 'best.npz'}")
    print(f"next: python sft.py --ckpt {out / 'best.npz'} --preset {args.preset}")


if __name__ == "__main__":
    main()
