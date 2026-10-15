"""Central configuration: paths, dataset constants, seed and model hyperparameters.

Everything that a run depends on lives here so that an experiment can be
reproduced from its stored config (each run's config is saved as JSON next to
its checkpoint and in the ``experiments`` SQL table).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    db_path: Path = PROJECT_ROOT / "db" / "har.sqlite"
    checkpoints_dir: Path = PROJECT_ROOT / "artifacts" / "checkpoints"
    figures_dir: Path = PROJECT_ROOT / "artifacts" / "figures"

    @property
    def dataset_dir(self) -> Path:
        """Folder created by extracting the official archive."""
        return self.raw_dir / "UCI HAR Dataset"


PATHS = Paths()

# ------------------------------------------------------------------ dataset constants
DATASET_NAME = "UCI HAR Dataset"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

# Official label ids (activity_labels.txt) -> names.
ACTIVITY_LABELS: dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
NUM_CLASSES = len(ACTIVITY_LABELS)

# Raw inertial channels, in the fixed order used for the (N, 128, 9) window tensor.
INERTIAL_CHANNELS: tuple[str, ...] = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)
WINDOW_LENGTH = 128  # 2.56 s at 50 Hz, 50% overlap
NUM_FEATURES = 561

# ------------------------------------------------------------------- experiment setup
SEED = 42
SEEDS: tuple[int, ...] = (42, 43, 44)  # used for the multi-seed comparison
N_VAL_SUBJECTS = 4  # training subjects held out (whole) for validation


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    n_jobs: int = 4  # same CPU parallelism as the torch models (fair timing)

    # small grid searched on the validation subjects (test set never used)
    grid: dict = field(default_factory=lambda: {
        "max_features": ["sqrt", 0.2],
        "min_samples_leaf": [1, 5],
    })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TorchTrainConfig:
    """Training settings shared by the CNN and the LSTM for a fair comparison."""

    batch_size: int = 64
    max_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10  # early stopping on validation macro-F1
    num_threads: int = 4  # fixed CPU threads so timings are comparable
    grad_clip: float | None = None  # max gradient norm; standard for RNNs, off for the CNN

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CNNConfig:
    channels: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 5
    dropout: float = 0.3
    train: TorchTrainConfig = field(default_factory=TorchTrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LSTMConfig:
    # Defaults = winner of scripts/tune_lstm.py (validation macro-F1 0.9084, seed 42).
    # The original untuned setting was dropout 0.3, unidirectional, pool "last" (val 0.8790).
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    bidirectional: bool = True
    pool: str = "mean"  # "last" hidden state or "mean" over time
    train: TorchTrainConfig = field(default_factory=lambda: TorchTrainConfig(grad_clip=1.0))

    def to_dict(self) -> dict:
        return asdict(self)
