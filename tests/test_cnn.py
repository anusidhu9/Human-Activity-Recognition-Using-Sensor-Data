"""Stage 5 tests: CNN shapes, ability to learn, trainer, determinism, checkpoints."""

import json

import numpy as np
import pytest
import torch
from sqlalchemy import text

from har.config import CNNConfig, TorchTrainConfig
from har.data.pipeline import prepare_data
from har.db.ingest import ingest_metadata
from har.db.schema import create_schema, get_engine
from har.models.cnn1d import CNN1D, build_cnn
from har.training.torch_trainer import (
    count_parameters, fit, load_checkpoint, predict_proba, run_torch_model,
)
from har.utils import set_seed

MANIFEST = {"source_url": "x", "sha256": "2" * 64, "size_bytes": 1,
            "downloaded_at": "2026-10-14T00:00:00+00:00"}
TINY = CNNConfig(channels=(8, 8, 8), train=TorchTrainConfig(batch_size=16, max_epochs=3,
                                                            patience=2, num_threads=1))


def test_cnn_output_shape():
    model = CNN1D()
    assert model(torch.randn(4, 128, 9)).shape == (4, 6)
    assert count_parameters(model) > 0


def test_cnn_handles_other_window_lengths():
    # global average pooling makes the network length-agnostic
    assert CNN1D()(torch.randn(2, 64, 9)).shape == (2, 6)


def test_cnn_can_overfit_one_batch():
    set_seed(0)
    x, y = torch.randn(32, 128, 9), torch.randint(0, 6, (32,))
    model = CNN1D(dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).float().mean() == 1.0


def test_predict_proba_rows_sum_to_one():
    probs = predict_proba(CNN1D(), np.random.randn(10, 128, 9).astype(np.float32),
                          torch.device("cpu"), batch_size=4)
    assert probs.shape == (10, 6)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)


def test_same_seed_same_initial_weights():
    set_seed(5)
    a = build_cnn(TINY).state_dict()
    set_seed(5)
    b = build_cnn(TINY).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def db_and_data(tmp_path, synthetic_splits, synthetic_dataset_dir):
    # make windows class-dependent so there is signal to learn
    for s in synthetic_splits.values():
        s.windows += s.labels[:, None, None].astype(np.float32)
    engine = get_engine(tmp_path / "cnn.sqlite")
    create_schema(engine)
    did = ingest_metadata(engine, synthetic_splits, MANIFEST, dataset_dir=synthetic_dataset_dir)
    data = prepare_data("windows", n_val_subjects=2,
                        train=synthetic_splits["train"], test=synthetic_splits["test"])
    return engine, did, data


def test_cnn_end_to_end_and_checkpoint_reload(tmp_path, db_and_data):
    engine, did, data = db_and_data
    res = run_torch_model("cnn", data, engine, did, seed=0, cfg=TINY,
                          checkpoints_dir=tmp_path / "ckpt", device="cpu", verbose=False)
    ckpt = res["checkpoint"]
    for f in ("model.pt", "standardizer.json", "config.json", "history.json", "metrics.json"):
        assert (ckpt / f).exists(), f
    assert 1 <= len(json.loads((ckpt / "history.json").read_text())["history"]) <= 3

    # the reloaded checkpoint must reproduce the stored test predictions
    reloaded = load_checkpoint(ckpt, device="cpu")
    probs = predict_proba(reloaded, data.X_test, torch.device("cpu"))
    with engine.connect() as c:
        stored = [r[0] for r in c.execute(text(
            "SELECT p.pred_activity_id FROM predictions p JOIN samples s USING (sample_id) "
            "WHERE p.eval_split = 'test' ORDER BY s.row_index"))]
        status = c.execute(text("SELECT status FROM experiments")).scalar()
    np.testing.assert_array_equal(probs.argmax(1) + 1, stored)
    assert status == "completed"


def test_training_is_reproducible(db_and_data):
    _, _, data = db_and_data
    histories = []
    for _ in range(2):
        set_seed(3)
        model = build_cnn(TINY)
        histories.append(fit(model, data, TINY.train, seed=3, device=torch.device("cpu"),
                             verbose=False).history)
    assert [e["train_loss"] for e in histories[0]] == pytest.approx(
        [e["train_loss"] for e in histories[1]])


def test_torch_runner_rejects_feature_input(tmp_path, synthetic_splits):
    data = prepare_data("features", n_val_subjects=2,
                        train=synthetic_splits["train"], test=synthetic_splits["test"])
    with pytest.raises(ValueError):
        run_torch_model("cnn", data, None, 1, seed=0, checkpoints_dir=tmp_path)
