"""Shared fixtures: a tiny synthetic dataset laid out like the real UCI HAR folder."""

import numpy as np
import pytest

from har.config import INERTIAL_CHANNELS, NUM_FEATURES, WINDOW_LENGTH
from har.data.loaders import HARSplit

# Each subject performs every activity, like the real data.
TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8]
TEST_SUBJECTS = [2, 4]
PER_SUBJECT = 12  # 2 windows x 6 activities


def make_split(split, subjects, seed=0) -> HARSplit:
    rng = np.random.default_rng(seed)
    n = len(subjects) * PER_SUBJECT
    labels = np.tile(np.repeat(np.arange(1, 7), 2), len(subjects)).astype(np.int64)
    subj = np.repeat(subjects, PER_SUBJECT).astype(np.int64)
    windows = rng.normal(loc=np.arange(len(INERTIAL_CHANNELS)), scale=2.0,
                         size=(n, WINDOW_LENGTH, len(INERTIAL_CHANNELS))).astype(np.float32)
    features = rng.uniform(-1, 1, (n, NUM_FEATURES)).astype(np.float32)
    return HARSplit(split, features, windows, labels, subj)


def write_split(root, data: HARSplit):
    d = root / data.split
    (d / "Inertial Signals").mkdir(parents=True)
    np.savetxt(d / f"X_{data.split}.txt", data.features)
    np.savetxt(d / f"y_{data.split}.txt", data.labels, fmt="%d")
    np.savetxt(d / f"subject_{data.split}.txt", data.subjects, fmt="%d")
    for i, ch in enumerate(INERTIAL_CHANNELS):
        np.savetxt(d / "Inertial Signals" / f"{ch}_{data.split}.txt", data.windows[..., i])


@pytest.fixture
def synthetic_splits():
    return {"train": make_split("train", TRAIN_SUBJECTS, 0),
            "test": make_split("test", TEST_SUBJECTS, 1)}


@pytest.fixture
def synthetic_dataset_dir(tmp_path, synthetic_splits):
    root = tmp_path / "UCI HAR Dataset"
    for data in synthetic_splits.values():
        write_split(root, data)
    return root
