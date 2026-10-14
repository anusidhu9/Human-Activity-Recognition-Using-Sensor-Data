"""Training loop and full experiment run for the PyTorch models (CNN, LSTM).

Protocol - the same for both networks:
* train on the training rows with AdamW + cross-entropy, mini-batches shuffled
  by a seeded generator;
* after every epoch, score the *validation subjects* (macro-F1); keep the best
  weights, halve the learning rate on plateaus, stop after ``patience`` epochs
  without improvement;
* restore the best weights, evaluate once on the test set, save checkpoint,
  predictions, metrics and timings.

Labels are official ids 1-6 everywhere else; inside this module they are
shifted to 0-5 for ``CrossEntropyLoss``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sqlalchemy.engine import Engine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from har.config import PATHS, CNNConfig, LSTMConfig, TorchTrainConfig
from har.data.pipeline import REPRESENTATIONS, PreparedData
from har.db import repository as repo
from har.evaluation.metrics import summary_line
from har.training.experiment import log_evaluation, metrics_json, time_inference, write_json
from har.utils import Timer, ensure_new_dir, get_device, make_run_id, project_relative, set_seed


# ------------------------------------------------------------------ model factory
def build_model(model_type: str, cfg) -> nn.Module:
    if model_type == "cnn":
        from har.models.cnn1d import build_cnn
        return build_cnn(cfg)
    if model_type == "lstm":
        from har.models.lstm import build_lstm
        return build_lstm(cfg)
    raise ValueError(f"unknown torch model type {model_type!r}")


def default_config(model_type: str):
    return {"cnn": CNNConfig, "lstm": LSTMConfig}[model_type]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------- core helpers
def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y - 1).long())
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device,
                  batch_size: int = 512) -> np.ndarray:
    """Softmax probabilities, (N, 6), columns in activity-id order 1..6."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


@dataclass
class FitResult:
    best_epoch: int
    best_val_f1: float
    epochs_run: int
    train_time_s: float
    history: list[dict] = field(default_factory=list)


def fit(model: nn.Module, data: PreparedData, tcfg: TorchTrainConfig, seed: int,
        device: torch.device, verbose: bool = True) -> FitResult:
    """Train with early stopping on validation macro-F1; leaves the best weights loaded."""
    model.to(device)
    loader = make_loader(data.X_train, data.y_train, tcfg.batch_size, shuffle=True, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state, bad, history = -1.0, 0, None, 0, []
    with Timer() as t:
        for epoch in range(1, tcfg.max_epochs + 1):
            model.train()
            total, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(yb)
                n += len(yb)

            val_probs = predict_proba(model, data.X_val, device)
            val_pred = val_probs.argmax(1) + 1
            val_f1 = f1_score(data.y_val, val_pred, average="macro")
            val_loss = float(-np.log(val_probs[np.arange(len(data.y_val)),
                                               data.y_val - 1] + 1e-12).mean())
            sched.step(val_f1)
            history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                            "val_macro_f1": float(val_f1), "val_acc": float((val_pred == data.y_val).mean()),
                            "lr": opt.param_groups[0]["lr"]})
            improved = val_f1 > best_f1
            if improved:
                best_f1, best_epoch, bad = val_f1, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if verbose:
                print(f"  epoch {epoch:3d} | train loss {total / n:.4f} | val loss {val_loss:.4f} "
                      f"| val macro-F1 {val_f1:.4f}{' *' if improved else ''}")
            if bad >= tcfg.patience:
                if verbose:
                    print(f"  early stop: no improvement for {tcfg.patience} epochs")
                break
    model.load_state_dict(best_state)
    return FitResult(best_epoch, float(best_f1), epoch, t.elapsed, history)


# ------------------------------------------------------------------ full experiment
def run_torch_model(model_type: str, data: PreparedData, engine: Engine, dataset_id: int,
                    seed: int, cfg=None, checkpoints_dir: Path = PATHS.checkpoints_dir,
                    device: str | None = None, verbose: bool = True) -> dict:
    if data.representation != "windows":
        raise ValueError("The CNN and LSTM use the raw inertial windows")
    cfg = cfg or default_config(model_type)
    tcfg: TorchTrainConfig = cfg.train
    torch.set_num_threads(tcfg.num_threads)
    dev = get_device(device)
    set_seed(seed)
    model = build_model(model_type, cfg)

    run_id = make_run_id(model_type, seed)
    ckpt_dir = ensure_new_dir(checkpoints_dir / run_id)
    exp_id = repo.create_experiment(
        engine, run_id=run_id, dataset_id=dataset_id, model_type=model_type,
        input_representation=REPRESENTATIONS["windows"], hyperparams=cfg.to_dict(),
        seed=seed, device=dev.type, val_subjects=data.val_subjects)
    try:
        if verbose:
            print(f"  device {dev} | {count_parameters(model):,} parameters")
        fit_res = fit(model, data, tcfg, seed, dev, verbose)

        val_ids = repo.sample_ids(engine, dataset_id, "train")[data.val_rows]
        test_ids = repo.sample_ids(engine, dataset_id, "test")[data.test_rows]
        val_m = log_evaluation(engine, exp_id, "val", val_ids, data.y_val,
                               predict_proba(model, data.X_val, dev))
        probs, infer_s = time_inference(lambda X: predict_proba(model, X, dev), data.X_test)
        test_m = log_evaluation(engine, exp_id, "test", test_ids, data.y_test, probs)

        hyperparams = {**cfg.to_dict(), "best_epoch": fit_res.best_epoch,
                       "epochs_run": fit_res.epochs_run}
        torch.save({"model_type": model_type, "state_dict": model.state_dict(),
                    "config": cfg.to_dict()}, ckpt_dir / "model.pt")
        data.standardizer.save(ckpt_dir / "standardizer.json")
        write_json(ckpt_dir / "config.json", {"run_id": run_id, "seed": seed, "device": dev.type,
                                              "hyperparams": hyperparams,
                                              "val_subjects": data.val_subjects})
        write_json(ckpt_dir / "history.json", {"history": fit_res.history})
        write_json(ckpt_dir / "metrics.json", {"val": metrics_json(val_m),
                                               "test": metrics_json(test_m)})
        repo.finish_experiment(
            engine, exp_id, hyperparams_json=json.dumps(hyperparams, sort_keys=True),
            n_params=count_parameters(model), checkpoint_path=project_relative(ckpt_dir),
            train_time_s=fit_res.train_time_s, inference_time_s=infer_s,
            inference_ms_per_sample=1000 * infer_s / len(data.X_test),
            notes=f"best epoch {fit_res.best_epoch}/{fit_res.epochs_run}; "
                  "train time includes per-epoch validation")
    except BaseException:
        repo.finish_experiment(engine, exp_id, status="failed")
        raise

    if verbose:
        print(f"  best epoch {fit_res.best_epoch} of {fit_res.epochs_run}")
        print(f"  val : {summary_line(val_m)}")
        print(f"  test: {summary_line(test_m)}")
    return {"run_id": run_id, "experiment_id": exp_id, "val": val_m, "test": test_m,
            "train_time_s": fit_res.train_time_s, "inference_time_s": infer_s,
            "checkpoint": ckpt_dir, "fit": fit_res}


def load_checkpoint(ckpt_dir: Path, device: str | None = None) -> nn.Module:
    """Rebuild a trained network from its checkpoint folder (for inference/analysis)."""
    blob = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = default_config(blob["model_type"])
    for k, v in blob["config"].items():
        if k == "train":
            cfg.train = TorchTrainConfig(**v)
        else:
            setattr(cfg, k, v)
    model = build_model(blob["model_type"], cfg)
    model.load_state_dict(blob["state_dict"])
    return model.to(get_device(device)).eval()
