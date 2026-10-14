"""Per-channel standardization for the raw inertial windows.

Statistics are fitted on the training rows only and then applied unchanged to
validation and test data, so no information from held-out subjects leaks in.
They are saved with each checkpoint so inference uses the exact same scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ChannelStandardizer:
    """z-score each of the C channels of (N, T, C) windows using train mean/std."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "ChannelStandardizer":
        if windows.ndim != 3:
            raise ValueError(f"expected (N, T, C) windows, got shape {windows.shape}")
        self.mean_ = windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.std_ = windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform")
        return ((windows - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"mean": self.mean_.tolist(), "std": self.std_.tolist(), "eps": self.eps}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStandardizer":
        d = json.loads(Path(path).read_text())
        obj = cls(eps=d["eps"])
        obj.mean_ = np.asarray(d["mean"], dtype=np.float32)
        obj.std_ = np.asarray(d["std"], dtype=np.float32)
        return obj
