"""Stage 1 tests: configuration constants and shared helpers."""

from datetime import datetime

import numpy as np
import pytest
import torch

from har import config
from har.utils import Timer, ensure_new_dir, get_device, make_run_id, set_seed


def test_activity_labels_match_official_ids():
    assert list(config.ACTIVITY_LABELS) == [1, 2, 3, 4, 5, 6]
    assert config.NUM_CLASSES == 6
    assert len(config.INERTIAL_CHANNELS) == 9


def test_paths_are_inside_project():
    for p in (config.PATHS.raw_dir, config.PATHS.db_path, config.PATHS.checkpoints_dir):
        assert config.PROJECT_ROOT in p.parents


def test_set_seed_makes_numpy_and_torch_reproducible():
    set_seed(123)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    set_seed(123)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    np.testing.assert_array_equal(a_np, b_np)
    assert torch.equal(a_t, b_t)


def test_get_device_defaults_and_forcing():
    assert get_device("cpu").type == "cpu"
    assert get_device().type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")
    with pytest.raises(ValueError):
        get_device("tpu")


def test_make_run_id_format():
    run_id = make_run_id("cnn", 42, now=datetime(2026, 10, 14, 19, 30, 12))
    assert run_id == "cnn_s42_20261014-193012"


def test_ensure_new_dir_refuses_overwrite(tmp_path):
    d = ensure_new_dir(tmp_path / "run1")
    assert d.is_dir()
    with pytest.raises(FileExistsError):
        ensure_new_dir(d)


def test_timer_measures_elapsed():
    with Timer() as t:
        sum(range(10_000))
    assert t.elapsed > 0
