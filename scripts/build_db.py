"""Stage 3: create the SQLite database and ingest dataset/sample metadata.

Refuses to touch an existing database (it may hold experiment results) unless
``--force`` is given, in which case the old file is renamed to a timestamped
backup rather than deleted.
"""

import argparse
import json
from datetime import datetime

from sqlalchemy import text

from har.config import PATHS
from har.data.download import MANIFEST_NAME
from har.data.loaders import load_cached_split
from har.db.ingest import ingest_metadata
from har.db.schema import create_schema, get_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="back up an existing database and build a fresh one")
    args = parser.parse_args()

    db_path = PATHS.db_path
    if db_path.exists():
        if not args.force:
            raise SystemExit(f"{db_path} already exists - not modifying it. "
                             "Use --force to back it up and rebuild.")
        backup = db_path.with_name(f"{db_path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}.sqlite")
        db_path.rename(backup)
        print(f"[backup] existing database moved to {backup}")

    manifest_path = PATHS.raw_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit("Dataset not downloaded - run scripts/download_data.py first.")
    manifest = json.loads(manifest_path.read_text())

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    create_schema(engine)
    splits = {s: load_cached_split(s) for s in ("train", "test")}
    dataset_id = ingest_metadata(engine, splits, manifest)
    print(f"[ok] database created at {db_path} (dataset_id={dataset_id})")

    with engine.connect() as conn:
        print("\n== Row counts")
        for table in ("activities", "subjects", "datasets", "data_files", "samples"):
            print(f"{table:12s} {conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()}")
        print("\n== Samples per split")
        for split, n, n_subj in conn.execute(text(
                "SELECT split, COUNT(*), COUNT(DISTINCT subject_id) FROM samples GROUP BY split")):
            print(f"{split:5s} {n} windows from {n_subj} subjects")
        leak = conn.execute(text(
            "SELECT COUNT(DISTINCT a.subject_id) FROM samples a JOIN samples b "
            "ON a.subject_id = b.subject_id WHERE a.split = 'train' AND b.split = 'test'")).scalar()
        print(f"\nsubjects in both splits (SQL check): {leak}")
        views = [r[0] for r in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))]
        print(f"views: {views}")
    assert leak == 0


if __name__ == "__main__":
    main()
