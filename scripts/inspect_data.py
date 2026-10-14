"""Stage 2b: inspect the dataset - shapes, class balance, subjects, value ranges.

Prints a report and runs the sanity checks that later stages rely on
(most importantly: no subject appears in both train and test).
"""

import numpy as np

from har.config import ACTIVITY_LABELS, INERTIAL_CHANNELS
from har.data.loaders import load_cached_split, load_feature_names


def main() -> None:
    train, test = load_cached_split("train"), load_cached_split("test")

    print("== Shapes")
    for s in (train, test):
        print(f"{s.split:5s}: features {s.features.shape}  windows {s.windows.shape}  "
              f"labels {s.labels.shape}  subjects {len(np.unique(s.subjects))}")

    print("\n== Class balance (count / share)")
    print(f"{'activity':20s} {'train':>12s} {'test':>12s}")
    for k, name in ACTIVITY_LABELS.items():
        a, b = (train.labels == k).sum(), (test.labels == k).sum()
        print(f"{name:20s} {a:6d} {a / len(train):5.1%} {b:6d} {b / len(test):5.1%}")

    tr_subj, te_subj = set(train.subjects.tolist()), set(test.subjects.tolist())
    print("\n== Subjects")
    print(f"train ({len(tr_subj)}): {sorted(tr_subj)}")
    print(f"test  ({len(te_subj)}): {sorted(te_subj)}")
    overlap = tr_subj & te_subj
    print(f"overlap: {sorted(overlap) or 'none'}")

    print("\n== Value ranges")
    print(f"features: min {train.features.min():.3f}  max {train.features.max():.3f}  "
          f"(dataset authors pre-scaled to [-1, 1])")
    for i, ch in enumerate(INERTIAL_CHANNELS):
        x = train.windows[..., i]
        print(f"{ch:12s} mean {x.mean():+.4f}  std {x.std():.4f}")

    names = load_feature_names()
    print(f"\n== Feature names: {len(names)} "
          f"({sum('__' in n for n in names)} duplicate names in features.txt were suffixed)")

    nan = sum(np.isnan(s.features).sum() + np.isnan(s.windows).sum() for s in (train, test))
    print(f"NaN values: {nan}")

    assert not overlap, "train/test subjects overlap"
    assert nan == 0, "dataset contains NaNs"
    assert set(train.labels) == set(test.labels) == set(ACTIVITY_LABELS)
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
