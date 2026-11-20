"""Token counting for budgets.

The default counter is a deterministic approximation of byte-pair-encoding counts (one token per short word,
punctuation mark or group of three digits, extra tokens for long words). It needs no download, which keeps tests
and CI offline. ``TiktokenCounter`` gives exact counts for a real tokeniser when ``tiktoken`` is installed.
"""

from __future__ import annotations

import re
from typing import Protocol

_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class TokenCounter(Protocol):
    name: str

    def __call__(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Approximate BPE token count without any vocabulary file."""

    name = "approx"

    def __call__(self, text: str) -> int:
        count = 0
        for match in _PIECE.finditer(text):
            piece = match.group()
            if piece[0].isalpha():
                count += 1 + (len(piece) - 1) // 8
            elif piece[0].isdigit():
                count += (len(piece) + 2) // 3
            else:
                count += 1
        return count


class TiktokenCounter:
    """Exact token count for a tiktoken encoding (``o200k_base`` by default)."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def __call__(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def get_token_counter(name: str = "approx") -> TokenCounter:
    """Return ``approx`` or ``tiktoken[:encoding]``."""
    if name == "approx":
        return ApproxTokenCounter()
    if name.startswith("tiktoken"):
        _, _, encoding = name.partition(":")
        return TiktokenCounter(encoding or "o200k_base")
    raise ValueError(f"unknown token counter {name!r} (use 'approx' or 'tiktoken[:encoding]')")
