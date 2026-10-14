"""One entry point that turns the dataset into model-ready train/val/test arrays.

Every model calls ``prepare_data`` with its input representation, so all three
models get exactly the same rows, the same validation subjects and the same
test set - only the representation differs:

* ``"features"`` -> (N, 561) engineered features, used as-is (already in [-1, 1])
* ``"windows"``  -> (N, 128, 9) raw signals, standardized per channel on train rows

Labels keep the official ids 1-6. ``*_rows`` are row indices into the official
split files; combine them with ``repository.sample_ids`` to store predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from har.config import N_VAL_SUBJECTS, SEED
from har.data.loaders import load_cached_split
from har.data.preprocess import ChannelStandardizer
from har.data.splits import subject_val_split

REPRESENTATIONS = {"features": "features_561", "windows": "windows_128x9"}


@dataclass
class PreparedData:
    representation: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_rows: np.ndarray  # rows of the official *train* split
    val_rows: np.ndarray  # rows of the official *train* split
    test_rows: np.ndarray  # rows of the official *test* split
    val_subjects: list[int]
    standardizer: ChannelStandardizer | None


def prepare_data(representation: str, n_val_subjects: int = N_VAL_SUBJECTS,
                 split_seed: int = SEED, train=None, test=None) -> PreparedData:
    """Build train/val/test arrays. ``train``/``test`` can be passed in (used by tests)."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {list(REPRESENTATIONS)}")
    train = train if train is not None else load_cached_split("train")
    test = test if test is not None else load_cached_split("test")

    train_rows, val_rows, val_subjects = subject_val_split(train.subjects, n_val_subjects,
                                                           split_seed)
    test_rows = np.arange(len(test))

    standardizer = None
    if representation == "features":
        X_tr, X_va, X_te = (train.features[train_rows], train.features[val_rows],
                            test.features)
    else:
        standardizer = ChannelStandardizer().fit(train.windows[train_rows])  # train rows only
        X_tr = standardizer.transform(train.windows[train_rows])
        X_va = standardizer.transform(train.windows[val_rows])
        X_te = standardizer.transform(test.windows)

    return PreparedData(
        representation=representation,
        X_train=X_tr, y_train=train.labels[train_rows],
        X_val=X_va, y_val=train.labels[val_rows],
        X_test=X_te, y_test=test.labels,
        train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
        val_subjects=val_subjects, standardizer=standardizer,
    )
