# TinyGPT

A GPT-style language model in **pure NumPy** — no PyTorch, no CUDA, no compile
step. About **0.1–5M parameters**, sized to train on an Android phone in Termux
with a few hundred MB of RAM.

```
token + position embeddings
  → N × (LayerNorm → causal self-attention → + → LayerNorm → GELU MLP → +)
  → LayerNorm → tied linear head → next-token loss
```

## Layout

```
tinygpt/
  config.py        presets, param count, RAM estimate
  tokenizer.py     character-level + chat specials (128 ids)
  model.py         transformer, backward pass, AdamW, generate
  dataset.py       pretrain windows, 90/10 split, SFT masking
  train.py         next-token pretraining
  sft.py           lightweight chat fine-tune
  generate.py      one-shot completion
  chat.py          interactive CLI
  utils.py         .npz checkpoints
  runtime.py       BLAS thread cap (import first)
  data/pretrain.md
  data/chat.md
```

## Presets

| name    | layers | heads | d_model | context | ~params | notes                    |
|---------|--------|-------|---------|---------|---------|--------------------------|
| nano    | 2      | 4     | 48      | 48      | 64K     | smoke test / tiny RAM    |
| phone   | 4      | 4     | 128     | 128     | 0.82M   | Termux default           |
| tiny    | 6      | 6     | 192     | 128     | 2.71M   | target 1–5M model        |
| compact | 8      | 8     | 224     | 128     | 4.88M   | top of the band          |

Dropout is 0 on the first three (saves RAM and time). Linear layers have **no
bias**; LayerNorm still has gain and bias. Input and output embeddings are **tied**.

## Termux

```
pkg update
pkg install python
pip install numpy
# copy this folder onto the phone, then:
cd tinygpt
python train.py --preset phone --max-steps 400
python sft.py --ckpt out/best.npz --preset phone
python chat.py --ckpt out/sft.npz
```

Optional: `export TINYGPT_THREADS=2` before running so OpenBLAS does not spawn
one thread per big.LITTLE core.

If `pip install numpy` fails, try `pkg install python-numpy`.

## Training

Pretrain (`train.py`) samples windows from `data/pretrain.md`, 90/10 train/val,
AdamW, cosine LR with warmup, grad clip 1.0. Checkpoints:

- `out/best.npz` + `out/best.json` — lowest val loss
- `out/last.npz` — most recent step

SFT (`sft.py`) reads `### user` / `### assistant` blocks in `data/chat.md` and
masks the prompt (`label = -1`) so the loss only lands on assistant tokens.

```
python train.py --preset nano --max-steps 80     # sanity check, ~minutes
python train.py --preset tiny --grad-ckpt        # 2.7M, lower peak RAM
python generate.py --ckpt out/best.npz --prompt "A language model is"
```

Flags worth knowing: `--batch-size`, `--lr`, `--resume out/last.npz`,
`--threads 2`.

## Chat

```
python chat.py --ckpt out/sft.npz
you> What is self-attention?
gpt> ...
```

`/reset` clears context, `/temp 0.7` and `/tok 60` tweak decoding, `/q` exits.

Format the model sees:

```
<|user|>
your question
<|assistant|>
the reply<|end|>
```

## Memory

Rough peak RSS is `12 bytes × params` (fp32 weights + Adam m/v) plus activations.
`--grad-ckpt` recomputes each block in backward and cuts activation RAM.
Keep `--batch-size 2` or `4`. Do not raise BLAS threads above 2–4 on a phone.

This is a teaching model. The bundled markdown is small on purpose — the nano
preset will memorize it. Swap `data/pretrain.md` for your own notes to learn
something you actually care about.
