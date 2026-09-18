"""Model and training presets for TinyGPT (pure NumPy).

All configs stay under ~5M parameters and are sized for a phone CPU
(Termux) with a few hundred MB of RAM. No YAML, no PyTorch — one file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    block_size: int = 128
    vocab_size: int = 128
    dropout: float = 0.0
    bias: bool = False  # Linear bias off; LayerNorm still has affine params
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 4
    grad_accum: int = 4  # effective batch = batch_size * grad_accum
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 50
    max_steps: int = 800
    eval_every: int = 50
    eval_batches: int = 8
    grad_clip: float = 1.0
    grad_checkpoint: bool = False
    seed: int = 1337
    num_threads: int = 2  # BLAS threads; set TINYGPT_THREADS before import for best effect
    # SFT
    sft_lr: float = 1e-4
    sft_steps: int = 200
    mask_prompt: bool = True  # only train on assistant tokens during SFT


PRESETS: dict[str, GPTConfig] = {
    # ~64K — smoke test / in-browser demo size
    "nano": GPTConfig(n_layer=2, n_head=4, n_embd=48, block_size=48, dropout=0.0),
    # ~0.82M — recommended Termux starting point
    "phone": GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0),
    # ~2.71M — the default "tiny GPT"
    "tiny": GPTConfig(n_layer=6, n_head=6, n_embd=192, block_size=128, dropout=0.0),
    # ~4.88M — top of the 1–5M band
    "compact": GPTConfig(n_layer=8, n_head=8, n_embd=224, block_size=128, dropout=0.1),
}

TRAIN_PRESETS: dict[str, TrainConfig] = {
    "nano": TrainConfig(batch_size=8, grad_accum=1, max_steps=200, warmup_steps=20, eval_every=20, sft_steps=80),
    "phone": TrainConfig(batch_size=4, grad_accum=4, max_steps=600, warmup_steps=40, eval_every=50, grad_checkpoint=False, sft_steps=150),
    "tiny": TrainConfig(batch_size=4, grad_accum=4, max_steps=800, warmup_steps=50, eval_every=50, grad_checkpoint=True, sft_steps=200),
    "compact": TrainConfig(batch_size=2, grad_accum=8, max_steps=800, warmup_steps=50, eval_every=50, grad_checkpoint=True, sft_steps=200),
}


def get_preset(name: str) -> tuple[GPTConfig, TrainConfig]:
    key = name.lower().strip()
    if key not in PRESETS:
        raise SystemExit(f"Unknown preset '{name}'. Choose from: {', '.join(PRESETS)}")
    return PRESETS[key], TRAIN_PRESETS[key]


def count_params(c: GPTConfig) -> int:
    """Match model.py: tied embeddings, LN affine, optional Linear bias."""
    D, L, T, V, bias = c.n_embd, c.n_layer, c.block_size, c.vocab_size, c.bias

    def lin(in_f: int, out_f: int) -> int:
        return out_f * in_f + (out_f if bias else 0)

    token_emb = V * D
    pos_emb = T * D
    ln = 2 * D
    attn = lin(D, 3 * D) + lin(D, D)
    mlp = lin(D, 4 * D) + lin(4 * D, D)
    blocks = L * (2 * ln + attn + mlp)
    return token_emb + pos_emb + blocks + ln


def estimate_train_ram_mb(c: GPTConfig, t: TrainConfig) -> int:
    """Rough peak RSS: fp32 weights + Adam m/v + activations."""
    params = count_params(c)
    bytes_per_param = 12  # param + m + v
    act = t.batch_size * c.block_size * c.n_embd * c.n_layer * 8 * 4
    if t.grad_checkpoint:
        act //= 4
    return int((params * bytes_per_param + act) / (1024 * 1024))


def with_overrides(c: GPTConfig, **kwargs: Any) -> GPTConfig:
    return replace(c, **{k: v for k, v in kwargs.items() if v is not None})
