"""Normalized SQLite schema (SQLAlchemy 2.x ORM) plus analysis views.

Design rules
------------
* Sensor data is NOT copied into SQL. ``data_files`` stores the path (relative
  to the project root) and SHA-256 of each source file; ``samples`` stores one
  row per window with its position (``row_index``) in those files.
* The true label lives only in ``samples.activity_id``; predictions reference the
  sample, and the views join them. Misclassifications and confusion matrices
  are views, not duplicated tables.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    CheckConstraint, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
    create_engine, event, text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------- dataset metadata
class Activity(Base):
    __tablename__ = "activities"
    activity_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # official id 1-6
    name: Mapped[str] = mapped_column(String(32), unique=True)


class Subject(Base):
    __tablename__ = "subjects"
    subject_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # volunteer 1-30
    split: Mapped[str] = mapped_column(String(8))  # official split the subject belongs to
    __table_args__ = (CheckConstraint("split IN ('train', 'test')"),)


class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str] = mapped_column(Text)
    root_path: Mapped[str] = mapped_column(Text)  # relative to project root
    archive_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    archive_size_bytes: Mapped[int] = mapped_column(Integer)
    downloaded_at: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(default=datetime.now)


class DataFile(Base):
    """One source file of the dataset, referenced by path - its contents stay on disk."""

    __tablename__ = "data_files"
    file_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.dataset_id"))
    split: Mapped[str] = mapped_column(String(8))
    role: Mapped[str] = mapped_column(String(32))  # features / labels / subjects / <channel>
    path: Mapped[str] = mapped_column(Text)  # relative to project root
    sha256: Mapped[str] = mapped_column(String(64))
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    __table_args__ = (UniqueConstraint("dataset_id", "split", "role"),)


class Sample(Base):
    """One 2.56 s window. ``row_index`` is its row in every file of its split."""

    __tablename__ = "samples"
    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.dataset_id"))
    split: Mapped[str] = mapped_column(String(8))
    row_index: Mapped[int] = mapped_column(Integer)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.subject_id"), index=True)
    activity_id: Mapped[int] = mapped_column(ForeignKey("activities.activity_id"), index=True)
    __table_args__ = (UniqueConstraint("dataset_id", "split", "row_index"),)


# --------------------------------------------------------------- experiment results
class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.dataset_id"))
    model_type: Mapped[str] = mapped_column(String(16))  # rf / cnn / lstm
    input_representation: Mapped[str] = mapped_column(String(32))  # features_561 / windows_128x9
    hyperparams_json: Mapped[str] = mapped_column(Text)
    seed: Mapped[int] = mapped_column(Integer)
    device: Mapped[str] = mapped_column(String(16))
    val_subjects_json: Mapped[str] = mapped_column(Text)
    n_params: Mapped[int | None] = mapped_column(Integer)
    checkpoint_path: Mapped[str | None] = mapped_column(Text)
    train_time_s: Mapped[float | None] = mapped_column(Float)
    inference_time_s: Mapped[float | None] = mapped_column(Float)
    inference_ms_per_sample: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="running")
    started_at: Mapped[datetime] = mapped_column(default=datetime.now)
    finished_at: Mapped[datetime | None]
    notes: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (CheckConstraint("status IN ('running', 'completed', 'failed')"),)


class Metric(Base):
    """Long format: one value per (experiment, split, metric, averaging, class)."""

    __tablename__ = "metrics"
    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.experiment_id"), index=True)
    eval_split: Mapped[str] = mapped_column(String(8))  # val / test
    metric_name: Mapped[str] = mapped_column(String(32))  # accuracy / precision / recall / f1
    average: Mapped[str | None] = mapped_column(String(16))  # macro / weighted / NULL per class
    activity_id: Mapped[int | None] = mapped_column(ForeignKey("activities.activity_id"))
    value: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    __tablename__ = "predictions"
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.experiment_id"), primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.sample_id"), primary_key=True)
    eval_split: Mapped[str] = mapped_column(String(8))  # val / test
    pred_activity_id: Mapped[int] = mapped_column(ForeignKey("activities.activity_id"))
    confidence: Mapped[float] = mapped_column(Float)  # probability of the predicted class


class PredictionProb(Base):
    __tablename__ = "prediction_probs"
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.experiment_id"), primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.sample_id"), primary_key=True)
    activity_id: Mapped[int] = mapped_column(ForeignKey("activities.activity_id"), primary_key=True)
    prob: Mapped[float] = mapped_column(Float)


class ErrorAnnotation(Base):
    __tablename__ = "error_annotations"
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.experiment_id"), primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("samples.sample_id"), primary_key=True)
    category: Mapped[str] = mapped_column(String(32))  # static_static / dynamic_dynamic / static_dynamic
    notes: Mapped[str | None] = mapped_column(Text)


# --------------------------------------------------------------------------- views
VIEWS = {
    "v_predictions_with_truth": """
        SELECT p.experiment_id, e.run_id, e.model_type, p.eval_split,
               s.sample_id, s.split, s.row_index, s.subject_id,
               s.activity_id AS true_activity_id, ta.name AS true_activity,
               p.pred_activity_id, pa.name AS pred_activity, p.confidence,
               CASE WHEN p.pred_activity_id = s.activity_id THEN 1 ELSE 0 END AS correct
        FROM predictions p
        JOIN samples s     ON s.sample_id = p.sample_id
        JOIN experiments e ON e.experiment_id = p.experiment_id
        JOIN activities ta ON ta.activity_id = s.activity_id
        JOIN activities pa ON pa.activity_id = p.pred_activity_id""",
    "v_misclassified": """
        SELECT * FROM v_predictions_with_truth WHERE correct = 0""",
    "v_confusion_matrix": """
        SELECT experiment_id, eval_split, true_activity_id, true_activity,
               pred_activity_id, pred_activity, COUNT(*) AS n
        FROM v_predictions_with_truth
        GROUP BY experiment_id, eval_split, true_activity_id, pred_activity_id""",
    "v_per_subject_accuracy": """
        SELECT experiment_id, eval_split, subject_id, COUNT(*) AS n,
               AVG(correct) AS accuracy
        FROM v_predictions_with_truth
        GROUP BY experiment_id, eval_split, subject_id""",
}


# -------------------------------------------------------------------------- engine
def get_engine(db_path: str | Path) -> Engine:
    """SQLite engine with foreign-key enforcement switched on (off by default in SQLite)."""
    url = "sqlite://" if str(db_path) == ":memory:" else f"sqlite:///{Path(db_path).as_posix()}"
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    return engine


def create_schema(engine: Engine) -> None:
    """Create all tables and views (idempotent)."""
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for name, sql in VIEWS.items():
            conn.execute(text(f"CREATE VIEW IF NOT EXISTS {name} AS {sql}"))
