"""Small LSTM hyperparameter search scored ONLY on the validation subjects.

Each candidate is trained with the standard loop (early stopping on validation
macro-F1). The test set is never evaluated here. Results are written to
artifacts/tuning/lstm_s<seed>.json (refuses to overwrite an existing file).

    python scripts/tune_lstm.py --seed 42
"""

import argparse
import json
from dataclasses import replace

import torch

from har.config import PATHS, SEED, LSTMConfig
from har.data.pipeline import prepare_data
from har.training.torch_trainer import build_model, count_parameters, fit
from har.utils import get_device, set_seed

# the original untuned setting (experiment 3); spelled out because LSTMConfig's
# defaults were later set to the winner of this search
BASE = LSTMConfig(dropout=0.3, bidirectional=False, pool="last")

CANDIDATES = {
    "mean_pool": replace(BASE, pool="mean"),
    "bi_mean_pool": replace(BASE, pool="mean", bidirectional=True),
    "mean_pool_lr5e-4": replace(BASE, pool="mean", train=replace(BASE.train, lr=5e-4)),
    "bi_mean_pool_dropout0.5": replace(BASE, pool="mean", bidirectional=True, dropout=0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_path = PATHS.root / "artifacts" / "tuning" / f"lstm_s{args.seed}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists - not overwriting it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_data("windows")
    device = get_device()
    torch.set_num_threads(BASE.train.num_threads)
    results = []
    for name, cfg in CANDIDATES.items():
        set_seed(args.seed)
        model = build_model("lstm", cfg)
        print(f"\n== {name} ({count_parameters(model):,} params)")
        res = fit(model, data, cfg.train, args.seed, device)
        results.append({"name": name, "config": cfg.to_dict(),
                        "n_params": count_parameters(model),
                        "best_val_macro_f1": res.best_val_f1, "best_epoch": res.best_epoch,
                        "epochs_run": res.epochs_run, "train_time_s": res.train_time_s})
        print(f"   -> best val macro-F1 {res.best_val_f1:.4f} at epoch {res.best_epoch} "
              f"({res.train_time_s:.0f}s)")

    best = max(results, key=lambda r: r["best_val_macro_f1"])
    out_path.write_text(json.dumps({"seed": args.seed, "val_subjects": data.val_subjects,
                                    "selection_metric": "val macro-F1",
                                    "baseline": {"name": "default (last pooling)",
                                                 "config": BASE.to_dict(),
                                                 "note": "scored in experiment 3"},
                                    "results": results, "selected": best["name"]}, indent=2))
    print("\n== Summary (validation macro-F1)")
    for r in sorted(results, key=lambda r: -r["best_val_macro_f1"]):
        print(f"{r['name']:26s} {r['best_val_macro_f1']:.4f}  epoch {r['best_epoch']:2d}/"
              f"{r['epochs_run']:2d}  {r['train_time_s']:5.0f}s")
    print(f"selected: {best['name']}  ->  {out_path}")


if __name__ == "__main__":
    main()
