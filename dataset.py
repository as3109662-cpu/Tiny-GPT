"""Tokenize markdown, 90/10 train/val split, memory-light sampling.

Pretrain: sliding windows over a uint16 token buffer (one copy of the corpus).
SFT: packed chat examples with optional prompt masking (labels = -1 on user side).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer import PAD_ID, Tokenizer

VAL_FRAC = 0.1
Array = np.ndarray


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_ids(ids: list[int], val_frac: float = VAL_FRAC) -> tuple[list[int], list[int]]:
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    n_train = max(1, n - n_val)
    return ids[:n_train], ids[n_train:]


class TokenBuffer:
    """Holds a 1-D token stream and samples (x, y) next-token pairs."""

    def __init__(self, ids: list[int], block_size: int) -> None:
        if len(ids) < block_size + 1:
            reps = (block_size + 1) // max(1, len(ids)) + 2
            ids = ids * reps
        self.data = np.asarray(ids, dtype=np.int32)
        self.block_size = block_size

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[Array, Array]:
        n = int(self.data.shape[0]) - self.block_size - 1
        ix = rng.integers(0, max(1, n), size=batch_size, dtype=np.int32)
        x = np.stack([self.data[i : i + self.block_size] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y


def load_pretrain(path: str | Path, tokenizer: Tokenizer, block_size: int) -> tuple[TokenBuffer, TokenBuffer]:
    ids = tokenizer.encode(read_text(path))
    train_ids, val_ids = split_ids(ids)
    return TokenBuffer(train_ids, block_size), TokenBuffer(val_ids, block_size)


def parse_dialogues(text: str) -> list[tuple[str, str]]:
    """Parse `### user` / `### assistant` blocks from a markdown file."""
    pairs: list[tuple[str, str]] = []
    user: str | None = None
    for chunk in text.split("### "):
        body = chunk.strip()
        if not body:
            continue
        nl = body.find("\n")
        if nl < 0:
            continue
        tag, rest = body[:nl].strip().lower(), body[nl + 1 :].strip()
        if tag == "user":
            user = rest
        elif tag == "assistant" and user is not None:
            pairs.append((user, rest))
            user = None
    return pairs


class ChatBuffer:
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Tokenizer,
        block_size: int,
        mask_prompt: bool = True,
    ) -> None:
        xs: list[Array] = []
        ys: list[Array] = []
        pad = PAD_ID
        for user, assistant in pairs:
            prefix = tokenizer.encode(f"<|user|>\n{user}\n<|assistant|>\n")
            reply = tokenizer.encode(f"{assistant}<|end|>")
            ids = prefix + reply
            labels = ([-1] * len(prefix) + reply) if mask_prompt else ids[:]
            if len(ids) > block_size:
                ids = ids[-block_size:]
                labels = labels[-block_size:]
            x = ids + [pad] * (block_size - len(ids))
            y = labels[1:] + [pad] * (block_size - len(labels) + 1)
            y = y[:block_size]
            y = [-1 if t == pad else t for t in y]
            xs.append(np.asarray(x[:block_size], dtype=np.int32))
            ys.append(np.asarray(y[:block_size], dtype=np.int32))
        if not xs:
            raise ValueError("No chat pairs parsed from SFT file.")
        self.xs = xs
        self.ys = ys

    def __len__(self) -> int:
        return len(self.xs)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[Array, Array]:
        n = len(self.xs)
        ix = rng.integers(0, n, size=batch_size, dtype=np.int32)
        x = np.stack([self.xs[int(i)] for i in ix])
        y = np.stack([self.ys[int(i)] for i in ix])
        return x, y


def load_sft(
    path: str | Path,
    tokenizer: Tokenizer,
    block_size: int,
    mask_prompt: bool = True,
) -> tuple[ChatBuffer, ChatBuffer]:
    pairs = parse_dialogues(read_text(path))
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    train, val = pairs[:-n_val], pairs[-n_val:]
    if not train:
        train = pairs
    return (
        ChatBuffer(train, tokenizer, block_size, mask_prompt),
        ChatBuffer(val, tokenizer, block_size, mask_prompt),
    )
