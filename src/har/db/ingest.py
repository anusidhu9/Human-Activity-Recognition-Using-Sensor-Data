"""Populate the metadata tables: activities, subjects, dataset, data files, samples."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from har.config import ACTIVITY_LABELS, DATASET_NAME, PATHS, PROJECT_ROOT
from har.data.download import sha256_of
from har.data.loaders import SPLITS, HARSplit, split_file_paths
from har.db.schema import Activity, DataFile, Dataset, Sample, Subject


def _rel(path: Path) -> str:
    """Project-relative POSIX path, so the DB stays valid if the project folder moves.

    Paths outside the project (e.g. test fixtures) are stored absolute.
    """
    path = Path(path).resolve()
    if PROJECT_ROOT in path.parents:
        return path.relative_to(PROJECT_ROOT).as_posix()
    return path.as_posix()


def ingest_metadata(engine: Engine, splits: dict[str, HARSplit], manifest: dict,
                    dataset_dir: Path = PATHS.dataset_dir) -> int:
    """Insert all metadata in one transaction and return the new ``dataset_id``."""
    with Session(engine) as session, session.begin():
        session.add_all(Activity(activity_id=k, name=v) for k, v in ACTIVITY_LABELS.items())

        for split in SPLITS:
            for subject_id in sorted(set(splits[split].subjects.tolist())):
                session.add(Subject(subject_id=subject_id, split=split))

        dataset = Dataset(
            name=DATASET_NAME,
            source_url=manifest["source_url"],
            root_path=_rel(dataset_dir),
            archive_sha256=manifest["sha256"],
            archive_size_bytes=manifest["size_bytes"],
            downloaded_at=manifest["downloaded_at"],
        )
        session.add(dataset)
        session.flush()  # assigns dataset_id

        for split in SPLITS:
            data = splits[split]
            for role, path in split_file_paths(split, dataset_dir).items():
                if role == "features":
                    n_cols = data.features.shape[1]
                elif role in ("labels", "subjects"):
                    n_cols = 1
                else:  # one inertial channel
                    n_cols = data.windows.shape[1]
                session.add(DataFile(dataset_id=dataset.dataset_id, split=split, role=role,
                                     path=_rel(path), sha256=sha256_of(path),
                                     n_rows=len(data), n_cols=n_cols))

            session.execute(insert(Sample), [
                {"dataset_id": dataset.dataset_id, "split": split, "row_index": i,
                 "subject_id": int(subj), "activity_id": int(label)}
                for i, (subj, label) in enumerate(zip(data.subjects, data.labels))
            ])
        return dataset.dataset_id
