"""Vérifications des calculs, de la temporalité et de l'absence de fuite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import eeg_project as eeg  # noqa: E402


def test_teacher_clock_example():
    """Les horaires de l'exemple doivent reproduire exactement 1291, 1298 et 7 s."""
    cohort, events, audit, _, _ = eeg.load_annotations(ROOT)
    event = events[events.record_id == "190304A-E"].iloc[0]
    assert (event.start_s, event.end_s, event.duration_s) == (1291, 1298, 7)
    assert len(cohort) == 16 and len(events) == 65
    assert not audit.loc[
        audit.record_id.isin(["230406B-A", "230515B-G", "210504B-C"]), "included"
    ].any()


def test_closed_interval_and_half_threshold():
    """Les bornes du PDF et le seuil inclusif sont contrôlés au niveau des échantillons."""
    t = np.arange(8, dtype=float)
    mask = eeg.sample_mask(t, [(1, 2)])
    np.testing.assert_array_equal(mask, [0, 1, 1, 0, 0, 0, 0, 0])
    _, _, labels, ratio, _ = eeg.segment_signal(t, np.arange(8, dtype=float), [(1, 2)], seconds=4)
    assert labels[0] == 1 and ratio[0] == 0.5


def test_features_against_closed_form():
    """Vérifie les statistiques avec une fenêtre dont les résultats sont connus."""
    row = eeg.extract_features(np.array([[1.0, 2.0, 3.0, 4.0]]), fs=4).iloc[0]
    assert row["mean"] == 2.5
    assert row["variance"] == pytest.approx(1.25)
    assert row["energy"] == 30
    assert row["rms"] == pytest.approx(np.sqrt(7.5))


@pytest.mark.parametrize("length", [511, 512])
def test_one_sided_fft_even_and_odd(length):
    """Une sinusoïde de 2 Hz d'amplitude 2 a une puissance de 2 dans Delta."""
    fs = length / 2
    t = np.arange(length) / fs
    window = 2 * np.sin(2 * np.pi * 2 * t)
    row = eeg.extract_features(window[None, :], fs).iloc[0]
    assert row.power_delta == pytest.approx(2, abs=1e-10)
    assert row.power_beta == pytest.approx(0, abs=1e-10)


def test_reconstruction_preserves_window_origin():
    """Un début à 10 s ne doit pas être déplacé à 9 s lors de la reconstruction."""
    frame = pd.DataFrame(
        {
            "record_id": ["a"] * 4,
            "start_s": [10.0, 11.0, 12.0, 13.0],
            "end_s": [12.0, 13.0, 14.0, 15.0],
        }
    )
    assert eeg.reconstruct_intervals(frame, [1, 1, 0, 0]) == [(10, 13)]
    frame.loc[3, "record_id"] = "b"
    with pytest.raises(ValueError):
        eeg.reconstruct_intervals(frame, [1, 1, 0, 1])


def test_event_assignment_does_not_count_duplicates():
    """Deux prédictions pour une crise ne produisent pas deux vrais positifs."""
    matches, fp, fn = eeg.match_intervals([(10, 20)], [(10, 20), (11, 19)])
    assert len(matches) == 1 and fp == 1 and fn == 0


def test_real_data_group_separation():
    """Les tracés de test sont absents de tous les plis d'apprentissage."""
    frame = pd.read_csv(ROOT / "artifacts/final/features.csv")
    train, test, folds = eeg.choose_split(frame)
    held_out = set(frame.iloc[test].record_id)
    assert held_out.isdisjoint(frame.iloc[train].record_id)
    for a, b in folds:
        assert set(frame.iloc[train[a]].record_id).isdisjoint(frame.iloc[train[b]].record_id)
        assert held_out.isdisjoint(frame.iloc[train[a]].record_id)
        assert held_out.isdisjoint(frame.iloc[train[b]].record_id)


def test_sampler_is_inside_each_pipeline():
    """Le rééchantillonnage ne peut pas s'appliquer au test via predict."""
    for strategy in eeg.STRATEGIES:
        pipe = eeg.make_model("dt", strategy, 20)
        assert list(pipe.named_steps) == ["scale", "sampling", "model"]
        assert pipe.named_steps["model"].random_state == 42


def test_short_runs_are_removed_and_long_runs_kept():
    """Une suite plus courte que la durée minimale disparaît, les autres restent intactes."""
    labels = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    kept = eeg.drop_short_runs(labels, 3)
    np.testing.assert_array_equal(kept, [0, 0, 0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(eeg.drop_short_runs(labels, 1), labels)


def test_vectorised_reconstruction_matches_reference():
    """La reconstruction rapide donne les mêmes épisodes que la version ligne à ligne."""
    rng = np.random.default_rng(0)
    starts = np.arange(200, dtype=float)
    frame = pd.DataFrame({"record_id": "a", "start_s": starts, "end_s": starts + 2})
    for _ in range(20):
        labels = rng.random(200) < 0.2
        assert eeg.intervals_from_labels(starts, starts + 2, labels) == eeg.reconstruct_intervals(
            frame, labels
        )


def test_smoothing_preserves_constant_scores():
    """Le lissage normalise ses bords : un score constant reste constant."""
    np.testing.assert_allclose(eeg.smooth_scores(np.full(7, 0.4), 5), 0.4)


def test_highpass_removes_offset():
    """Le filtre passe-haut retire le décalage continu qui identifiait les tracés."""
    fs = 256
    t = np.arange(20 * fs) / fs
    signal = 500 + 10 * np.sin(2 * np.pi * 3 * t)
    filtered = eeg.highpass(signal, fs)
    assert abs(filtered[fs:-fs].mean()) < 0.1
    assert filtered[fs:-fs].std() == pytest.approx(10 / np.sqrt(2), rel=0.02)


def test_robust_normalisation_is_label_free_and_bounded():
    """La normalisation par tracé ne voit que les caractéristiques, et reste bornée."""
    features = pd.DataFrame({"x": np.r_[np.zeros(50), np.ones(50), 1e6]})
    normalised = eeg.robust_normalize(features)
    assert normalised.x.max() == 10
    assert normalised.x.median() == pytest.approx(0, abs=0.5)


def test_balanced_strategy_sets_class_weight():
    """La pondération passe par le classifieur, sans échantillonneur."""
    pipe = eeg.make_model("rf", "balanced", 20)
    assert pipe.named_steps["sampling"] == "passthrough"
    assert pipe.named_steps["model"].class_weight == "balanced"
    with pytest.raises(ValueError):
        eeg.make_model("knn", "balanced", 20)
    assert len(eeg.configurations()) == 38
