"""Tests des fonctions de l'interface et du module de deep learning (sans données réelles)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import service  # noqa: E402
from src import eeg_deep as deep  # noqa: E402
from src import eeg_project as eeg  # noqa: E402


def test_hysteresis_starts_high_and_ends_low():
    scores = np.array([0.1, 0.5, 0.8, 0.5, 0.4, 0.2, 0.5, 0.1])
    labels = service.hysteresis(scores, high=0.7, low=0.3, min_windows=1)
    # Début à 0,8 ; reste active tant que le score ne passe pas sous 0,3 ; 0,5 seul ne relance pas.
    assert labels.tolist() == [False, False, True, True, True, False, False, False]


def test_hysteresis_drops_short_runs():
    scores = np.array([0.0, 0.9, 0.1, 0.9, 0.9, 0.9, 0.0])
    labels = service.hysteresis(scores, high=0.5, low=0.5, min_windows=3)
    assert labels.tolist() == [False, False, False, True, True, True, False]


def test_event_statistics():
    stats = service.event_statistics([(10, 20), (50, 54), (100, 106)], duration_s=3600)
    assert stats["count"] == 3
    assert stats["per_hour"] == pytest.approx(3)
    assert stats["total_s"] == pytest.approx(20)
    assert stats["mean_duration_s"] == pytest.approx(20 / 3)
    assert stats["mean_gap_s"] == pytest.approx((30 + 46) / 2)


def test_event_statistics_empty():
    stats = service.event_statistics([], duration_s=600)
    assert stats["count"] == 0 and stats["total_s"] == 0 and np.isnan(stats["mean_gap_s"])


def test_compare_to_reference_uses_iou_threshold():
    truth = [(10.0, 20.0), (100.0, 110.0)]
    detected = [(11.0, 21.0), (104.0, 130.0), (300.0, 310.0)]
    _, tp, fp, fn = service.compare_to_reference(truth, detected)
    # IoU 9/11 pour la première ; 6/30 = 0,2 < 0,3 pour la seconde.
    assert (tp, fp, fn) == (1, 2, 1)


def test_calibration_perfect_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    predicted, observed, counts, brier = service.calibration(y, p)
    assert counts.sum() == p.size
    assert np.abs(predicted - observed).max() < 0.05
    assert brier == pytest.approx(np.mean((p - y) ** 2))


def test_deep_window_starts_match_segmentation():
    fs, n = 256.0, 256 * 37 + 100
    t = np.arange(n) / fs
    _, starts, *_ = eeg.segment_signal(t, np.zeros(n), [])
    assert np.array_equal(deep.window_starts(n, fs), starts)


def test_prepare_downsamples_and_normalises():
    fs = 256.0
    t = np.arange(int(60 * fs)) / fs
    signals = np.stack([np.sin(2 * np.pi * 3 * t) * (k + 1) * 50 + 1000 for k in range(19)])
    x = deep.prepare(signals, fs)
    assert x.shape == (19, len(t) // 2) and x.dtype == np.float32
    # Même forme après normalisation, quelle que soit l'amplitude de la voie.
    middle = slice(1000, -1000)
    assert np.allclose(x[0, middle], x[18, middle], atol=1e-3)
    assert np.abs(np.median(x, axis=1)).max() < 0.05
    assert np.abs(x).max() <= deep.CLIP


def test_prepare_rejects_incompatible_rate():
    with pytest.raises(ValueError):
        deep.prepare(np.zeros((19, 1000)), 200.0)


def test_select_channels_order_and_missing():
    names = list(reversed(deep.CHANNELS))
    signals = np.arange(19)[:, None] * np.ones((19, 4))
    out = deep.select_channels(signals, names)
    assert out[:, 0].tolist() == [names.index(c) for c in deep.CHANNELS]
    with pytest.raises(ValueError, match="Fp1"):
        deep.select_channels(signals[1:], [c for c in deep.CHANNELS if c != "EEGFp1"])


def test_cnn_shapes_mc_dropout_and_saliency():
    model = deep.CNN1D()
    record = np.random.default_rng(0).normal(size=(19, deep.WINDOW * 4)).astype(np.float32)
    items = np.array([[0, 0], [0, deep.WINDOW], [0, 2 * deep.WINDOW]])
    p = deep.predict(model, [record], items)
    assert p.shape == (3,) and ((p > 0) & (p < 1)).all()
    mean, std = deep.predict_mc(model, [record], items, passes=10)
    assert mean.shape == std.shape == (3,) and (std > 0).all()
    assert not any(m.training for m in model.modules()), "le dropout doit être désactivé après MC"
    sal, prob = deep.saliency(model, record[:, : deep.WINDOW])
    assert sal.shape == (19, deep.WINDOW) and (sal >= 0).all() and 0 < prob < 1
