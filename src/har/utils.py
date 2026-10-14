"""Small shared helpers: seeding, device selection, timing and safe output dirs."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs.

    ``deterministic=True`` also asks PyTorch for deterministic kernels; on CPU
    this has negligible cost. Ops without a deterministic version only warn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # the RF pipeline does not need torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(prefer: str | None = None):
    """Return a ``torch.device``: CUDA when available, otherwise CPU.

    ``prefer`` ("cpu" or "cuda") forces a choice; asking for CUDA on a machine
    without it raises instead of silently falling back.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    if prefer is not None:
        raise ValueError(f"Unknown device preference: {prefer!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_id(model_type: str, seed: int, now: datetime | None = None) -> str:
    """Unique, sortable run identifier, e.g. ``rf_s42_20261014-193012``."""
    now = now or datetime.now()
    return f"{model_type}_s{seed}_{now:%Y%m%d-%H%M%S}"


def ensure_new_dir(path: str | Path) -> Path:
    """Create ``path`` and return it; refuse if it already exists.

    Used for checkpoint folders so an earlier run is never overwritten.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {path}")
    path.mkdir(parents=True)
    return path


def project_relative(path: str | Path) -> str:
    """Project-relative POSIX path (absolute if outside the project), for storing in SQL."""
    from har.config import PROJECT_ROOT

    path = Path(path).resolve()
    if PROJECT_ROOT in path.parents:
        return path.relative_to(PROJECT_ROOT).as_posix()
    return path.as_posix()


class Timer:
    """Wall-clock timer usable as a context manager: ``with Timer() as t: ...; t.elapsed``."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start
