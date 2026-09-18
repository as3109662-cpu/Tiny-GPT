#!/usr/bin/env python3
"""Lightweight supervised fine-tune on chat-formatted markdown (NumPy).

    python sft.py --ckpt out/best.npz --preset phone --data data/chat.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from runtime import setup

setup()

from config import get_preset  # noqa: E402
from dataset import load_sft  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from utils import configure_runtime, load_ckpt, print_banner, save_ckpt  # noqa: E402


def estimate_loss(model, split, batches, rng, bs) -> float:
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
    p = argparse.ArgumentParser(description="SFT TinyGPT for chat")
    p.add_argument("--ckpt", required=True, help="pretrain checkpoint (out/best.npz)")
    p.add_argument("--preset", default="phone", choices=["nano", "phone", "tiny", "compact"])
    p.add_argument("--data", default="data/chat.md")
    p.add_argument("--out", default="out")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-mask", action="store_true", help="train on user tokens too")
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        setup(args.threads)

    _, train_cfg = get_preset(args.preset)
    train_cfg.max_steps = args.steps or train_cfg.sft_steps
    train_cfg.learning_rate = args.lr or train_cfg.sft_lr
    train_cfg.min_lr = train_cfg.learning_rate / 5
    train_cfg.warmup_steps = min(20, train_cfg.max_steps // 5)
    train_cfg.mask_prompt = not args.no_mask

    rng = configure_runtime(train_cfg)
    model, _blob = load_ckpt(args.ckpt)
    model.enable_grad_checkpoint(train_cfg.grad_checkpoint)
    tok = Tokenizer()
    train_buf, val_buf = load_sft(args.data, tok, model.config.block_size, train_cfg.mask_prompt)
    print(f"chat pairs  train={len(train_buf)}  val={len(val_buf)}  mask_prompt={train_cfg.mask_prompt}")
    print_banner(args.preset + "/sft", model, train_cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.training = True
    t0 = time.time()
    best = float("inf")
    accum = max(1, train_cfg.grad_accum // 2)

    for step in range(train_cfg.max_steps):
        lr = train_cfg.learning_rate
        model.zero_grad()
        loss_acc = 0.0
        for _ in range(accum):
            x, y = train_buf.sample(train_cfg.batch_size, rng)
            _, loss = model.forward(x, y)
            model.backward()
            loss_acc += float(loss or 0.0)
        if accum > 1:
            scale = 1.0 / accum
            for _, p in model.named_params():
                p.grad *= scale
        model.clip_grad(train_cfg.grad_clip)
        model.adamw(lr, train_cfg.betas, train_cfg.weight_decay, model.adam_step + 1)

        if step % 10 == 0 or step == train_cfg.max_steps - 1:
            print(
                f"sft {step:5d}/{train_cfg.max_steps}  "
                f"loss={loss_acc / accum:.4f}  {time.time() - t0:.1f}s"
            )
            t0 = time.time()

        if step % max(10, train_cfg.eval_every // 2) == 0 or step == train_cfg.max_steps - 1:
            val = estimate_loss(model, val_buf, 4, rng, train_cfg.batch_size)
            print(f"  eval  val_loss={val:.4f}")
            save_ckpt(out / "sft-last.npz", model, step, {"val_loss": val, "phase": "sft"})
            if val < best:
                best = val
                save_ckpt(out / "sft.npz", model, step, {"val_loss": val, "phase": "sft"})
                print("  saved out/sft.npz")

    print(f"SFT done. chat with: python chat.py --ckpt {out / 'sft.npz'}")


if __name__ == "__main__":
    main()
