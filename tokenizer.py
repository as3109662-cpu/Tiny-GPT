"""Character-level tokenizer with atomic chat specials.

Fixed 128-id vocabulary so checkpoints stay compatible across machines:

  0–3    <|user|> <|assistant|> <|end|> <|pad|>
  4      newline
  5–99   printable ASCII (codes 32–126)
  100–127 reserved (never produced by encode)

No regex, no downloads, no sentencepiece. Fast on a phone.
"""

from __future__ import annotations

from dataclasses import dataclass

SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>", "<|pad|>")
PAD_ID = 3
VOCAB_SIZE = 128
NEWLINE = "\n"
PRINTABLE = "".join(chr(i) for i in range(32, 127))  # 95 chars
BASE_CHARS = NEWLINE + PRINTABLE  # 96 chars; ids 4..99


def _build_tables() -> tuple[list[str], dict[str, int]]:
    itos: list[str] = list(SPECIAL_TOKENS) + list(BASE_CHARS)
    while len(itos) < VOCAB_SIZE:
        itos.append(f"<|res{len(itos)}|>")
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


ITOS, STOI = _build_tables()


@dataclass
class Tokenizer:
    vocab_size: int = VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(text)
        specials = SPECIAL_TOKENS
        while i < n:
            matched = False
            if text[i] == "<":
                for tok in specials:
                    if text.startswith(tok, i):
                        ids.append(STOI[tok])
                        i += len(tok)
                        matched = True
                        break
            if matched:
                continue
            ch = text[i]
            ids.append(STOI.get(ch, STOI.get("?", 0)))
            i += 1
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(ITOS):
                tok = ITOS[i]
                if tok.startswith("<|res"):
                    continue
                out.append(tok)
        return "".join(out)

    def encode_chat(self, user: str, assistant: str | None = None) -> list[int]:
        text = f"<|user|>\n{user}\n<|assistant|>\n"
        if assistant is not None:
            text += f"{assistant}<|end|>"
        return self.encode(text)

    def prompt_len(self, user: str) -> int:
        return len(self.encode(f"<|user|>\n{user}\n<|assistant|>\n"))
