#!/usr/bin/env python3
"""Minimal local chat CLI (NumPy).

    python chat.py --ckpt out/sft.npz

Commands: /reset  /temp 0.7  /tok 80  /q
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from runtime import setup

setup()

from tokenizer import Tokenizer  # noqa: E402
from utils import load_ckpt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with a TinyGPT checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temperature", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=80)
    return p.parse_args()


def generate_reply(
    model,
    tok: Tokenizer,
    history: str,
    user: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
) -> str:
    prompt = history + f"<|user|>\n{user}\n<|assistant|>\n"
    ids = tok.encode(prompt)
    block = model.config.block_size
    if len(ids) >= block:
        ids = ids[-(block - 1) :]
    x = np.asarray([ids], dtype=np.int32)
    eos = tok.encode("<|end|>")[0]
    out = model.generate(
        x,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos,
    )
    new_ids = out[0, len(ids) :].tolist()
    if eos in new_ids:
        new_ids = new_ids[: new_ids.index(eos)]
    reply = tok.decode(new_ids).strip()
    if not reply:
        reply = "I do not know."
    return reply


def main() -> None:
    args = parse_args()
    model, _blob = load_ckpt(args.ckpt)
    tok = Tokenizer()
    temp, max_tok = args.temperature, args.max_tokens
    history = ""
    n = model.n_params()
    print(f"TinyGPT chat  numpy  params={n:,}  T={model.config.block_size}")
    print("type a message, or /reset  /temp 0.7  /tok 60  /q")
    print("-" * 48)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/q", "/quit", "/exit"}:
            break
        if user == "/reset":
            history = ""
            print("(context cleared)")
            continue
        if user.startswith("/temp"):
            parts = user.split()
            if len(parts) == 2:
                temp = float(parts[1])
                print(f"temperature={temp}")
            continue
        if user.startswith("/tok"):
            parts = user.split()
            if len(parts) == 2:
                max_tok = int(parts[1])
                print(f"max_tokens={max_tok}")
            continue
        reply = generate_reply(model, tok, history, user, temp, args.top_k, max_tok)
        history += f"<|user|>\n{user}\n<|assistant|>\n{reply}<|end|>"
        sys.stdout.write(f"gpt> {reply}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
