# TinyGPT

A GPT-style language model in pure NumPy — no PyTorch, no CUDA, no compile step. About 0.1–5M parameters, sized to train on an Android phone in Termux with a few hundred MB of RAM.

```
token + position embeddings
  → N × (LayerNorm → causal self-attention → + → LayerNorm → GELU MLP → +)
  → LayerNorm → tied linear head → next-token loss
```

## What this repo is

This project is a compact, phone-friendly training loop for a tiny GPT-style language model using only NumPy. It includes:

- next-token pretraining on a markdown corpus
- train/validation split
- lightweight supervised fine-tune for chat behavior
- local CLI chat inference
- compact checkpointing and model presets
- Termux-friendly runtime settings to avoid CPU thrash

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

| name    | layers | heads | d_model | context | ~params | notes                  |
|---------|--------|-------|---------|---------|---------|------------------------|
| nano    | 2      | 4     | 48      | 48      | 64K     | smoke test / tiny RAM  |
| phone   | 4      | 4     | 128     | 128     | 0.82M   | Termux default         |
| tiny    | 6      | 6     | 192     | 128     | 2.71M   | target 1–5M model      |
| compact | 8      | 8     | 224     | 128     | 4.88M   | top of the band        |

Dropout is 0 on the first three (saves RAM and time). Linear layers have no bias; LayerNorm still has gain and bias. Input and output embeddings are tied.

## Termux quick start

```bash
pkg update
pkg install python
pip install numpy

cd tinygpt
export TINYGPT_THREADS=2

python train.py --preset nano --max-steps 80
python sft.py --ckpt out/best.npz --preset nano
python chat.py --ckpt out/sft.npz
```

For a fuller model:

```bash
python train.py --preset phone --max-steps 400
python sft.py --ckpt out/best.npz --preset phone
python chat.py --ckpt out/sft.npz
```

If `pip install numpy` fails, try:

```bash
pkg install python-numpy
```

## Training

Pretrain (`train.py`) samples windows from `data/pretrain.md`, performs a 90/10 train/val split, uses AdamW, cosine LR with warmup, and gradient clipping. Checkpoints:

- `out/best.npz` + `out/best.json` — lowest validation loss
- `out/last.npz` — most recent step

SFT (`sft.py`) reads `### user` / `### assistant` blocks in `data/chat.md` and masks the prompt (`label = -1`) so the loss only lands on assistant tokens.

```bash
python train.py --preset nano --max-steps 80      # smoke test, minutes
python train.py --preset tiny --grad-ckpt         # 2.7M, lower peak RAM
python generate.py --ckpt out/best.npz --prompt "A language model is"
```

Helpful flags:

- `--batch-size`
- `--lr`
- `--resume out/last.npz`
- `--threads 2`
- `--grad-ckpt`

## Chat

```bash
python chat.py --ckpt out/sft.npz
```

Example:

```text
you> What is self-attention?
gpt> Self-attention lets each token look at earlier tokens, score them, and mix their values.
```

Commands:

- `/reset` clears context
- `/temp 0.7` changes decoding temperature
- `/tok 60` changes max generation tokens
- `/q` exits

Format seen by the model:

```text
<|user|>
your question
<|assistant|>
the reply<|end|>
```

## Good practice for mobile training

- keep `--preset nano` or `phone` for Termux
- use `TINYGPT_THREADS=2` or lower
- keep `--batch-size` small
- prefer `grad-ckpt` when RAM is tight
- use short, clean markdown data instead of noisy large dumps
- do a small smoke test first before long pretraining runs

## Hallucination control

Small models can invent text. To reduce garbage output on a phone:

- lower temperature (`0.2` to `0.4` is safer than `0.8`)
- keep `top-k` small (example: 8–20)
- use chat fine-tuning examples with clear refusal patterns
- keep the dataset concise and factual
- if the model is unsure, teach it to answer: `I do not know.`

This is a teaching model. The bundled markdown is small on purpose — the nano preset will memorize it. Swap `data/pretrain.md` for your own notes to learn something you actually care about.

## Current repo status

This repository is already in the correct direction for a Termux-friendly, NumPy-only tiny GPT project. The main remaining work is practical usage:

1. fill `data/pretrain.md` with your corpus
2. fill `data/chat.md` with user/assistant examples
3. run a small smoke test
4. then train on the phone preset
5. then run local chat inference

## Releases

No releases published yet.

## Languages

Python 100%
