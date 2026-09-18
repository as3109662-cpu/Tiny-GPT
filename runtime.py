"""Must be imported before NumPy. Caps BLAS threads so Termux does not thrash."""

from __future__ import annotations

import os


def setup(threads: int | None = None) -> int:
    n = threads
    if n is None:
        raw = os.environ.get("TINYGPT_THREADS", "2")
        n = max(1, int(raw))
    s = str(n)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, s)
    return n
