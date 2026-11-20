"""Reproduce the README benchmark (same as `tiered-memory bench` with the published settings).

    python benchmarks/run_benchmark.py            # real embedding model (downloads ~70 MB on first run)
    python benchmarks/run_benchmark.py --embedder hashing   # offline, no download

Extra arguments are passed through to the `bench` command and override the defaults below.
"""

from __future__ import annotations

import sys

from tiered_memory.cli import app

DEFAULTS = ["--users", "50", "--sessions", "10", "--seed", "7", "--embedder", "fastembed",
            "--growth-users", "10", "--growth-sessions", "40"]

if __name__ == "__main__":
    app(["bench", *DEFAULTS, *sys.argv[1:]])
