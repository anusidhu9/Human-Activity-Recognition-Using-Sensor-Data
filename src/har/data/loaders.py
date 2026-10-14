"""Load the two views of the UCI HAR data for one official split.

Both views describe the *same* windows, row for row:

* ``features`` - (N, 561) engineered features from ``X_<split>.txt`` (Random Forest)
* ``windows``  - (N, 128, 9) raw inertial signals from ``Inertial Signals/`` (CNN, LSTM),
  channels ordered as ``config.INERTIAL_CHANNELS``

``labels`` keep the official ids 1-6 and ``subjects`` the volunteer ids 1-30.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from har.config import INERTIAL_CHANNELS, NUM_FEATURES, PATHS, WINDOW_LENGTH

SPLITS = ("train", "test")


@dataclass
class HARSplit:
    split: str
    features: np.ndarray  # (N, 561) float32
    windows: np.ndarray  # (N, 128, 9) float32
    labels: np.ndarray  # (N,) int64, 1-6
    subjects: np.ndarray  # (N,) int64, 1-30

    def __len__(self) -> int:
        return len(self.labels)


def _read_matrix(path: Path) -> np.ndarray:
    """Read a whitespace-separated numeric text file (fast pandas C parser)."""
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32).to_numpy()


def _read_ints(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)


def split_file_paths(split: str, dataset_dir: Path = PATHS.dataset_dir) -> dict[str, Path]:
    """All files belonging to a split, keyed by role (used again for SQL metadata)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    d = dataset_dir / split
    files = {
        "features": d / f"X_{split}.txt",
        "labels": d / f"y_{split}.txt",
        "subjects": d / f"subject_{split}.txt",
    }
    for ch in INERTIAL_CHANNELS:
        files[ch] = d / "Inertial Signals" / f"{ch}_{split}.txt"
    return files


def load_feature_names(dataset_dir: Path = PATHS.dataset_dir) -> list[str]:
    """The 561 feature names. The official list has duplicates, so repeats get a suffix."""
    raw = pd.read_csv(dataset_dir / "features.txt", sep=r"\s+", header=None)[1].tolist()
    seen: dict[str, int] = {}
    names = []
    for n in raw:
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}__{seen[n]}")
    return names


def load_split(split: str, dataset_dir: Path = PATHS.dataset_dir) -> HARSplit:
    """Load features, raw windows, labels and subjects for ``split`` and check alignment."""
    files = split_file_paths(split, dataset_dir)
    features = _read_matrix(files["features"])
    windows = np.stack([_read_matrix(files[ch]) for ch in INERTIAL_CHANNELS], axis=-1)
    labels = _read_ints(files["labels"])
    subjects = _read_ints(files["subjects"])

    n = len(labels)
    if features.shape != (n, NUM_FEATURES):
        raise ValueError(f"{split}: features shape {features.shape}, expected ({n}, {NUM_FEATURES})")
    if windows.shape != (n, WINDOW_LENGTH, len(INERTIAL_CHANNELS)):
        raise ValueError(f"{split}: windows shape {windows.shape} does not match {n} labels")
    if len(subjects) != n:
        raise ValueError(f"{split}: {len(subjects)} subject ids for {n} labels")
    return HARSplit(split, features, windows, labels, subjects)


def load_cached_split(split: str, dataset_dir: Path = PATHS.dataset_dir,
                      cache_dir: Path = PATHS.processed_dir) -> HARSplit:
    """Like ``load_split`` but cached as ``.npz`` (parsing the text files takes seconds).

    The cache is written only if absent; delete it manually to rebuild.
    """
    cache = cache_dir / f"{split}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return HARSplit(split, z["features"], z["windows"], z["labels"], z["subjects"])
    data = load_split(split, dataset_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=data.features, windows=data.windows,
             labels=data.labels, subjects=data.subjects)
    return data
