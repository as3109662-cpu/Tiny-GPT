#!/usr/bin/env python3
"""One-shot completion from a NumPy checkpoint."""

from __future__ import annotations

import argparse

import numpy as np

from runtime import setup

setup()

from tokenizer import Tokenizer  # noqa: E402
from utils import load_ckpt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="A language model is")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_ckpt(args.ckpt)
    tok = Tokenizer()
    ids = tok.encode(args.prompt)
    x = np.asarray([ids], dtype=np.int32)
    out = model.generate(
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
