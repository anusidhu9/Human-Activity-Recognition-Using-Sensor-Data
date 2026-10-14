"""Subject-grouped validation split carved out of the official training set.

Windows overlap by 50% and every subject's windows are highly correlated, so a
random row-level split would leak near-duplicates into validation. Instead,
whole subjects are held out. The test set is never touched here.
"""

from __future__ import annotations

import numpy as np

from har.config import N_VAL_SUBJECTS, SEED


def subject_val_split(subjects: np.ndarray, n_val_subjects: int = N_VAL_SUBJECTS,
                      seed: int = SEED) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (train_rows, val_rows, val_subjects) for the official training rows.

    The split depends only on ``seed`` (default: the project seed), so every
    model and every training seed sees the same validation subjects.
    """
    unique = np.unique(subjects)
    if not 0 < n_val_subjects < len(unique):
        raise ValueError(f"n_val_subjects must be between 1 and {len(unique) - 1}")
    rng = np.random.default_rng(seed)
    val_subjects = sorted(rng.choice(unique, size=n_val_subjects, replace=False).tolist())
    is_val = np.isin(subjects, val_subjects)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val), val_subjects
