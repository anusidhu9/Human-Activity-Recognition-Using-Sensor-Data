"""Train and evaluate one model for one seed, recording everything in SQL.

    python scripts/train.py --model rf --seed 42
"""

import argparse

from har.config import ACTIVITY_LABELS, PATHS, SEED
from har.data.pipeline import prepare_data
from har.db import repository as repo
from har.db.schema import get_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["rf"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not PATHS.db_path.exists():
        raise SystemExit("Database missing - run scripts/build_db.py first.")
    engine = get_engine(PATHS.db_path)
    dataset_id = repo.latest_dataset_id(engine)

    if args.model == "rf":
        from har.training.sklearn_runner import run_random_forest
        data = prepare_data("features")
        print(f"[rf] seed {args.seed} | train {len(data.X_train)} val {len(data.X_val)} "
              f"test {len(data.X_test)} | val subjects {data.val_subjects}")
        result = run_random_forest(data, engine, dataset_id, args.seed)

    m = result["test"]
    print(f"\n[done] run_id {result['run_id']} (experiment_id {result['experiment_id']})")
    print(f"train time {result['train_time_s']:.2f}s | test inference "
          f"{result['inference_time_s']:.3f}s ({1000 * result['inference_time_s'] / len(data.X_test):.4f} ms/sample)")
    print("\nper-class test F1:")
    for aid, s in m["per_class"].items():
        print(f"  {ACTIVITY_LABELS[aid]:20s} {s['f1']:.4f}  (n={s['support']})")
    print("\ntest confusion matrix (rows = true, cols = predicted, order 1-6):")
    print(m["confusion_matrix"])
    print(f"\ncheckpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
