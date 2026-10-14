"""Stage 6 tests: LSTM shapes, pooling variants, ability to learn, end-to-end run."""

import numpy as np
import pytest
import torch
from sqlalchemy import text

from har.config import LSTMConfig, TorchTrainConfig
from har.data.pipeline import prepare_data
from har.db.ingest import ingest_metadata
from har.db.schema import create_schema, get_engine
from har.models.lstm import LSTMClassifier, build_lstm
from har.training.torch_trainer import load_checkpoint, predict_proba, run_torch_model
from har.utils import set_seed

MANIFEST = {"source_url": "x", "sha256": "3" * 64, "size_bytes": 1,
            "downloaded_at": "2026-10-14T00:00:00+00:00"}


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pool", ["last", "mean"])
def test_lstm_output_shape(bidirectional, pool):
    model = LSTMClassifier(hidden_size=16, bidirectional=bidirectional, pool=pool)
    assert model(torch.randn(3, 128, 9)).shape == (3, 6)


def test_invalid_pool_rejected():
    with pytest.raises(ValueError):
        LSTMClassifier(pool="max")


def test_lstm_default_config_clips_gradients():
    assert LSTMConfig().train.grad_clip == 1.0
    assert TorchTrainConfig().grad_clip is None  # CNN default unchanged


def test_lstm_can_overfit_one_batch():
    set_seed(0)
    x, y = torch.randn(16, 32, 9), torch.randint(0, 6, (16,))
    model = LSTMClassifier(hidden_size=32, num_layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).float().mean() == 1.0


def test_lstm_end_to_end_and_reload(tmp_path, synthetic_splits, synthetic_dataset_dir):
    for s in synthetic_splits.values():
        s.windows += s.labels[:, None, None].astype(np.float32)
    engine = get_engine(tmp_path / "lstm.sqlite")
    create_schema(engine)
    did = ingest_metadata(engine, synthetic_splits, MANIFEST, dataset_dir=synthetic_dataset_dir)
    data = prepare_data("windows", n_val_subjects=2,
                        train=synthetic_splits["train"], test=synthetic_splits["test"])
    cfg = LSTMConfig(hidden_size=8, num_layers=1,
                     train=TorchTrainConfig(batch_size=16, max_epochs=2, patience=2,
                                            num_threads=1, grad_clip=1.0))
    res = run_torch_model("lstm", data, engine, did, seed=0, cfg=cfg,
                          checkpoints_dir=tmp_path / "ckpt", device="cpu", verbose=False)
    reloaded = load_checkpoint(res["checkpoint"], device="cpu")
    assert isinstance(reloaded, LSTMClassifier)
    assert reloaded.lstm.hidden_size == build_lstm(cfg).lstm.hidden_size == 8

    # the reloaded checkpoint must reproduce the predictions stored in SQL
    probs = predict_proba(reloaded, data.X_test, torch.device("cpu"))
    with engine.connect() as c:
        stored = [r[0] for r in c.execute(text(
            "SELECT p.pred_activity_id FROM predictions p JOIN samples s USING (sample_id) "
            "WHERE p.eval_split = 'test' ORDER BY s.row_index"))]
        status, model_type = c.execute(text("SELECT status, model_type FROM experiments")).one()
    np.testing.assert_array_equal(probs.argmax(1) + 1, stored)
    assert (status, model_type) == ("completed", "lstm")
