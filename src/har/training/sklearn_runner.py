"""Train, tune (on validation subjects), evaluate and record the Random Forest.

Protocol (identical in spirit to the torch models):
1. every grid candidate is fitted on the training rows only;
2. the candidate with the best *validation* macro-F1 is selected;
3. that fitted model is evaluated once on the test set;
4. checkpoint, predictions, metrics and timings are stored.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
from sklearn.metrics import f1_score
from sqlalchemy.engine import Engine

from har.config import PATHS, RFConfig
from har.data.pipeline import REPRESENTATIONS, PreparedData
from har.db import repository as repo
from har.evaluation.metrics import LABEL_IDS, summary_line
from har.models.random_forest import build_random_forest, count_nodes
from har.training.experiment import log_evaluation, metrics_json, time_inference, write_json
from har.utils import Timer, ensure_new_dir, make_run_id, project_relative, set_seed


def run_random_forest(data: PreparedData, engine: Engine, dataset_id: int, seed: int,
                      cfg: RFConfig | None = None,
                      checkpoints_dir: Path = PATHS.checkpoints_dir) -> dict:
    if data.representation != "features":
        raise ValueError("The Random Forest baseline uses the engineered features")
    cfg = cfg or RFConfig()
    set_seed(seed)
    run_id = make_run_id("rf", seed)
    ckpt_dir = ensure_new_dir(checkpoints_dir / run_id)

    exp_id = repo.create_experiment(
        engine, run_id=run_id, dataset_id=dataset_id, model_type="rf",
        input_representation=REPRESENTATIONS["features"], hyperparams=cfg.to_dict(),
        seed=seed, device="cpu", val_subjects=data.val_subjects)
    try:
        # ---- 1-2: grid search on validation subjects
        keys = list(cfg.grid)
        tuning, best = [], None
        with Timer() as tune_t:
            for values in itertools.product(*(cfg.grid[k] for k in keys)):
                params = dict(zip(keys, values))
                model = build_random_forest(cfg, seed, **params)
                with Timer() as fit_t:
                    model.fit(data.X_train, data.y_train)
                val_f1 = f1_score(data.y_val, model.predict(data.X_val), average="macro")
                tuning.append({**params, "val_macro_f1": float(val_f1),
                               "fit_time_s": fit_t.elapsed})
                print(f"  grid {params} -> val macro-F1 {val_f1:.4f} ({fit_t.elapsed:.1f}s)")
                if best is None or val_f1 > best[0]:
                    best = (val_f1, params, model, fit_t.elapsed)
        _, best_params, model, train_time = best
        if list(model.classes_) != LABEL_IDS:  # predict_proba columns must be ids 1..6
            raise RuntimeError(f"unexpected class order {model.classes_}")
        print(f"  selected {best_params}")

        # ---- 3: evaluation (validation for reference, then the single test pass)
        val_ids = repo.sample_ids(engine, dataset_id, "train")[data.val_rows]
        test_ids = repo.sample_ids(engine, dataset_id, "test")[data.test_rows]
        val_m = log_evaluation(engine, exp_id, "val", val_ids, data.y_val,
                               model.predict_proba(data.X_val))
        probs, infer_s = time_inference(model.predict_proba, data.X_test)
        test_m = log_evaluation(engine, exp_id, "test", test_ids, data.y_test, probs)

        # ---- 4: checkpoint + run record
        hyperparams = {**cfg.to_dict(), **best_params, "grid_results": tuning,
                       "tuning_time_s": tune_t.elapsed}
        joblib.dump(model, ckpt_dir / "model.joblib", compress=3)
        write_json(ckpt_dir / "config.json", {"run_id": run_id, "seed": seed,
                                              "hyperparams": hyperparams,
                                              "val_subjects": data.val_subjects})
        write_json(ckpt_dir / "metrics.json", {"val": metrics_json(val_m),
                                               "test": metrics_json(test_m)})
        repo.finish_experiment(
            engine, exp_id, hyperparams_json=json.dumps(hyperparams, sort_keys=True),
            n_params=count_nodes(model), checkpoint_path=project_relative(ckpt_dir),
            train_time_s=train_time, inference_time_s=infer_s,
            inference_ms_per_sample=1000 * infer_s / len(data.X_test),
            notes="n_params = total tree nodes")
    except BaseException:
        repo.finish_experiment(engine, exp_id, status="failed")
        raise

    print(f"  val : {summary_line(val_m)}")
    print(f"  test: {summary_line(test_m)}")
    return {"run_id": run_id, "experiment_id": exp_id, "val": val_m, "test": test_m,
            "train_time_s": train_time, "inference_time_s": infer_s,
            "checkpoint": ckpt_dir}
