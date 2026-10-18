"""Accès aux données et calculs de l'interface, mis en cache par Streamlit.

Toute la chaîne de détection vient de `app/inference.py`, qui appelle les fonctions
du notebook (`src/eeg_project.py`) : l'interface donne exactement les mêmes résultats.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, sosfiltfilt, spectrogram

from app import inference
from src import eeg_project as eeg

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data/exports"
ARTIFACTS = ROOT / "artifacts/final"
MODEL_DIR = ROOT / "app/model"
STATUS_LABELS = {
    "test": "Test réservé",
    "apprentissage": "Apprentissage",
    "sans annotation": "Sans annotation",
    "exclu": "Exclu",
}


class DataError(RuntimeError):
    """Fichier du projet absent ou modifié ; message présentable tel quel."""


def missing_setup():
    """Liste les fichiers indispensables absents, avec la commande qui les produit."""
    needed = {
        MODEL_DIR / "metadata.json": "app/prepare_model.py",
        MODEL_DIR / "recordings.json": "app/prepare_model.py",
        MODEL_DIR / "test_scores.csv": "app/prepare_model.py",
        ARTIFACTS / "test_results.csv": "scripts/execute_final_notebook.py",
    }
    return [
        (path.relative_to(ROOT), command) for path, command in needed.items() if not path.exists()
    ]


# ---------- Modèle et catalogue ----------


@st.cache_resource(show_spinner="Chargement du modèle…")
def detector():
    return inference.load_detector()


def metadata():
    return detector()[1]


@st.cache_data
def catalogue():
    return inference.load_catalogue()


@st.cache_data
def reference():
    return inference.load_reference()


def annotated_events(record_id):
    """Crises annotées (début, fin) en secondes, pour un enregistrement de la cohorte."""
    rows = reference()
    rows = rows[rows.record_id == record_id]
    return list(rows[["start_s", "end_s"]].itertuples(index=False, name=None))


def record_options(require_signal=True):
    """Identifiants du catalogue, test réservé en premier, puis apprentissage."""
    rows = catalogue().values()
    return [row["id"] for row in rows if row["available"] or not require_signal]


def record_label(record_id):
    row = catalogue()[record_id]
    return f"{record_id} · {STATUS_LABELS[row['status']]}"


def export_path(record_id):
    return EXPORTS / f"{record_id.replace('-', '_')}.npz"


def verified_export(record_id):
    """Octets de l'export, refusés si l'empreinte diffère de celle contrôlée contre le MAT."""
    path = export_path(record_id)
    if not path.exists():
        raise DataError(f"Export introuvable : {path.relative_to(ROOT)}.")
    payload = path.read_bytes()
    expected = catalogue()[record_id]["export_sha256"]
    if hashlib.sha256(payload).hexdigest() != expected:
        raise DataError(
            f"{path.name} a changé depuis son contrôle contre le fichier MAT. Relancez app/prepare_model.py."
        )
    return payload, path.name


# ---------- Signaux multivoies (visualisation) ----------


@st.cache_data(max_entries=3, show_spinner="Lecture de l'enregistrement…")
def multichannel(record_id):
    """Toutes les voies EEG de l'export vérifié, avec l'axe temporel mesuré."""
    verified_export(record_id)  # refuse un export dont l'empreinte a changé
    path = export_path(record_id)
    with np.load(path, allow_pickle=False) as bundle:
        names = bundle["channel_names"].tolist()
        signals = bundle["signals"]
        rows = bundle["row_times_ms"]
        keep = [i for i, name in enumerate(names) if name.startswith("EEG")]
        data = signals[keep].astype(np.float32)
    time_values = inference.time_from_rows(rows, data.shape[1])
    fs = float(1 / np.median(np.diff(time_values)))
    return time_values, data, [names[i] for i in keep], fs


@st.cache_data(max_entries=6)
def display_filter(record_id, lowpass):
    """Passe-haut 0,5 Hz du pipeline, et passe-bas 40 Hz optionnel pour l'affichage seul."""
    _, data, _, fs = multichannel(record_id)
    out = eeg.highpass(data.astype(float), fs).astype(np.float32)
    if lowpass:
        sos = butter(4, 40, btype="lowpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out).astype(np.float32)
    return out


@st.cache_data(max_entries=6)
def channel_spectrogram(record_id, channel, fmax=20.0):
    """Spectrogramme d'une voie : fenêtres de 2 s, pas de 0,5 s, puissance en dB."""
    _, _, names, fs = multichannel(record_id)
    signal = display_filter(record_id, False)[names.index(channel)]
    f, t, power = spectrogram(signal, fs=fs, nperseg=int(2 * fs), noverlap=int(1.5 * fs))
    mask = f <= fmax
    return f[mask], t, 10 * np.log10(power[mask] + 1e-12)


def spike_wave_map(record_id, center_s, seconds=2.0):
    """Puissance relative 2,5–4,5 Hz par électrode sur une fenêtre centrée sur `center_s`."""
    time_values, _, names, fs = multichannel(record_id)
    filtered = display_filter(record_id, False)
    n = int(round(seconds * fs))
    start = int(
        np.clip(np.searchsorted(time_values, center_s - seconds / 2), 0, len(time_values) - n)
    )
    spectrum, frequencies = eeg.one_sided_spectrum(filtered[:, start : start + n].astype(float), fs)
    total = eeg.band_power(spectrum, frequencies, 0.5, 30) + 1e-12
    relative = eeg.band_power(spectrum, frequencies, *eeg.SPIKE_WAVE_BAND) / total
    return [name.removeprefix("EEG") for name in names], relative


# ---------- Détection ----------


@st.cache_data(max_entries=8, show_spinner="Analyse en cours…")
def analyse_record(record_id):
    payload, filename = verified_export(record_id)
    return _analyse(payload, filename)


@st.cache_data(max_entries=4, show_spinner="Analyse en cours…")
def analyse_upload(payload, filename):
    return _analyse(payload, filename)


def _analyse(payload, filename):
    pipeline, meta = detector()
    recording = inference.load_recording(payload, filename, meta["fs_hz"])
    result = inference.analyze(recording, pipeline, meta, reference(), catalogue())
    result.pop("overview", None)
    return result


def hysteresis(scores, high, low, min_windows):
    """Une crise commence au-dessus de `high` et se termine sous `low`."""
    labels = np.zeros(len(scores), dtype=bool)
    active = False
    for i, value in enumerate(scores):
        if not active and value >= high:
            active = True
        elif active and value < low:
            active = False
        labels[i] = active
    return eeg.drop_short_runs(labels, min_windows)


def detections_from_labels(result, labels):
    half = metadata()["window_seconds"] / 2
    centers = np.asarray(result["scores"]["time"])
    return eeg.intervals_from_labels(centers - half, centers + half, labels)


def compare_to_reference(truth, detected):
    """Appariement IoU ≥ 0,3, identique à l'évaluation du notebook."""
    matches, fp, fn = eeg.match_intervals(truth, detected)
    return matches, len(matches), fp, fn


def event_statistics(intervals, duration_s):
    """Tableau de bord d'un enregistrement : fréquence, durée cumulée, intervalles."""
    durations = np.array([b - a for a, b in intervals], dtype=float)
    gaps = np.array([intervals[k + 1][0] - intervals[k][1] for k in range(len(intervals) - 1)])
    return {
        "count": len(intervals),
        "per_hour": len(intervals) / (duration_s / 3600) if duration_s else np.nan,
        "total_s": float(durations.sum()),
        "fraction": float(durations.sum() / duration_s) if duration_s else np.nan,
        "mean_duration_s": float(durations.mean()) if len(durations) else np.nan,
        "mean_gap_s": float(gaps.mean()) if len(gaps) else np.nan,
        "durations": durations,
    }


# ---------- Résultats des modèles ----------


@st.cache_data
def model_table():
    """Les 38 configurations : AP de validation interne et métriques du test réservé."""
    test = pd.read_csv(ARTIFACTS / "test_results.csv")
    cv = pd.read_csv(ARTIFACTS / "cv_summary.csv")[
        ["configuration", "average_precision_mean", "f1_mean"]
    ]
    table = test.merge(cv, on="configuration")
    table["Configuration"] = table.configuration.map(eeg.config_label)
    table["Modèle"] = table.model.map(eeg.MODELS)
    table["Caractéristiques"] = table.feature_set.map(eeg.FEATURE_SET_LABELS)
    table["Stratégie"] = table.strategy.map(eeg.STRATEGIES)
    return table.sort_values("average_precision_mean", ascending=False).reset_index(drop=True)


@st.cache_data
def test_scores():
    return pd.read_csv(MODEL_DIR / "test_scores.csv")


@st.cache_data
def loro_table():
    path = ARTIFACTS / "loro_results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def cohort_summary():
    counts = pd.read_csv(ARTIFACTS / "class_counts.csv")
    events = pd.read_csv(ARTIFACTS / "annotations_seconds.csv")
    return {
        "records": len(counts),
        "events": int(events.record_id.isin(counts.record_id).sum()),
        "windows": int(counts.windows.sum()),
        "positive_windows": int(counts.positive_windows.sum()),
        "hours": float(counts.duration_s.sum() / 3600),
        "fs": float(counts.fs_hz.iloc[0]),
    }


@st.cache_data(show_spinner="Projection des fenêtres…")
def window_projection(method, sample=6000):
    """ACP ajustée sur l'apprentissage seulement, ou t-SNE sur un échantillon."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    columns = eeg.IMPROVED_FEATURES
    frame = pd.read_csv(
        ARTIFACTS / "features.csv", usecols=["record_id", "start_s", "Class", *columns]
    )
    test_records = set(metadata()["test_records"])
    frame["split"] = np.where(frame.record_id.isin(test_records), "Test réservé", "Apprentissage")
    rng = np.random.default_rng(eeg.SEED)
    positives = np.flatnonzero(frame.Class.to_numpy() == 1)
    negatives = np.flatnonzero(frame.Class.to_numpy() == 0)
    keep = np.sort(
        np.r_[positives, rng.choice(negatives, max(0, sample - len(positives)), replace=False)]
    )
    shown = frame.iloc[keep].reset_index(drop=True)
    if method == "ACP":
        train = frame[frame.split == "Apprentissage"][columns]
        scaler = StandardScaler().fit(train)
        pca = PCA(n_components=2, random_state=eeg.SEED).fit(scaler.transform(train))
        coords = pca.transform(scaler.transform(shown[columns]))
        explained = pca.explained_variance_ratio_.tolist()
    else:
        scaled = StandardScaler().fit_transform(shown[columns])
        coords = TSNE(
            n_components=2, perplexity=40, random_state=eeg.SEED, init="pca"
        ).fit_transform(scaled)
        explained = None
    shown["x"], shown["y"] = coords[:, 0], coords[:, 1]
    return shown, explained


def test_window_outcomes():
    """VP / FP / FN / VN du modèle retenu sur le test, après le post-traitement réglé hors pli."""
    meta = metadata()
    scores = test_scores()
    labels = np.zeros(len(scores), dtype=bool)
    for _, index in scores.groupby("record_id").indices.items():
        order = index[np.argsort(scores.start_s.to_numpy()[index], kind="stable")]
        p = meta["params"]
        labels[order] = eeg.postprocess(
            scores[meta["configuration"]].to_numpy()[order],
            p["threshold"],
            p["smoothing"],
            p["min_windows"],
        )
    truth = scores.Class.to_numpy() == 1
    outcome = np.select(
        [truth & labels, ~truth & labels, truth & ~labels],
        ["Vrai positif", "Fausse alarme", "Crise manquée"],
        "Vrai négatif",
    )
    return scores[["record_id", "start_s"]].assign(outcome=outcome)


def analysis_json(result):
    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------- Deep learning (CNN 1D, 19 voies) ----------

CNN_FILES = (
    MODEL_DIR / "cnn.pt",
    MODEL_DIR / "cnn_metadata.json",
    MODEL_DIR / "cnn_test_scores.csv",
)


def cnn_available():
    return all(path.exists() for path in CNN_FILES)


@st.cache_resource(show_spinner="Chargement du réseau…")
def cnn_detector():
    import torch

    from src import eeg_deep as deep

    meta = json.loads(CNN_FILES[1].read_text(encoding="utf-8"))
    if hashlib.sha256(CNN_FILES[0].read_bytes()).hexdigest() != meta["model_sha256"]:
        raise DataError("Le fichier cnn.pt a changé. Relancez scripts/train_cnn.py.")
    model = deep.CNN1D()
    model.load_state_dict(torch.load(CNN_FILES[0], map_location="cpu", weights_only=True))
    model.eval()
    return model, meta


@st.cache_data(max_entries=3, show_spinner="Préparation des 19 voies…")
def cnn_input(record_id):
    """Voies dans l'ordre du réseau, filtrées, à 128 Hz, et débuts des fenêtres."""
    from src import eeg_deep as deep

    time_values, data, names, fs = multichannel(record_id)
    try:
        prepared = deep.prepare(deep.select_channels(data, names), fs)
    except ValueError as exc:
        raise DataError(f"Le CNN ne peut pas analyser {record_id} : {exc}") from exc
    starts = deep.window_starts(data.shape[1], fs)
    return prepared, time_values[starts], round(fs / deep.DEEP_FS)


@st.cache_data(max_entries=8, show_spinner="Analyse par le CNN (30 passages Monte-Carlo)…")
def cnn_analyse(record_id, passes=30):
    """Probabilité moyenne et écart-type Monte-Carlo par fenêtre, puis post-traitement réglé hors pli."""
    from src import eeg_deep as deep

    model, meta = cnn_detector()
    prepared, starts_s, factor = cnn_input(record_id)
    items = np.c_[np.zeros(len(starts_s), dtype=int), np.round(starts_s * deep.DEEP_FS).astype(int)]
    mean, std = deep.predict_mc(model, [prepared], items, passes=passes)
    params = meta["params"]
    smoothed = eeg.smooth_scores(mean, params["smoothing"])
    spread = eeg.smooth_scores(std, params["smoothing"])
    labels = eeg.postprocess(mean, params["threshold"], params["smoothing"], params["min_windows"])
    ends_s = starts_s + 2.0
    intervals = eeg.intervals_from_labels(starts_s, ends_s, labels)
    truth = annotated_events(record_id)
    matches = eeg.match_intervals(truth, intervals)[0] if truth else []
    by_prediction = {j: (i, iou) for i, j, iou in matches}
    events = []
    for k, (a, b) in enumerate(intervals):
        inside = (starts_s >= a - 1e-6) & (ends_s <= b + 1e-6)
        match = by_prediction.get(k)
        events.append({
            "id": k + 1, "start_s": a, "end_s": b, "duration_s": b - a,
            "mean_score": float(smoothed[inside].mean()), "peak_score": float(smoothed[inside].max()),
            "mean_std": float(spread[inside].mean()),
            "match": {"reference": match[0] + 1, "iou": match[1]} if match else None,
        })  # fmt: skip
    reference_block = None
    if truth:
        matched = {i for i, _, _ in matches}
        reference_block = {
            "events": [{"id": i + 1, "start_s": a, "end_s": b, "matched": i in matched} for i, (a, b) in enumerate(truth)],
            "tp": len(matches), "fp": len(intervals) - len(matches), "fn": len(truth) - len(matches),
        }  # fmt: skip
    row = catalogue()[record_id]
    duration = float(len(prepared[0]) / deep.DEEP_FS)
    high = max(float(np.quantile(smoothed + spread, 0.999)), params["threshold"])
    return {
        "model": meta["label"],
        "record_id": record_id,
        "membership": row["status"],
        "source": "Export du projet vérifié contre le MAT source, 19 voies",
        "input_sha256": row["export_sha256"],
        "duration_s": duration,
        "analyzed_duration_s": float(ends_s[-1]),
        "n_windows": len(starts_s),
        "positive_windows": int(labels.sum()),
        "events": events,
        "reference": reference_block,
        "params": params,
        "scores": {
            "time": (starts_s + 1.0).round(2).tolist(),
            "value": smoothed.round(4).tolist(),
            "std": spread.round(4).tolist(),
            "threshold": params["threshold"],
            "range": [-0.03, min(1.03, high * 1.1 + 0.02)],
        },
    }


@st.cache_data(max_entries=16, show_spinner="Calcul de la carte de saillance…")
def cnn_saliency(record_id, center_s, span_s=6.0, step_s=0.25):
    """Gradient × entrée moyenné sur des fenêtres de 2 s glissant par 0,25 s autour de `center_s`."""
    from src import eeg_deep as deep

    model, _ = cnn_detector()
    prepared, _, _ = cnn_input(record_id)
    n = prepared.shape[1]
    first = int(
        np.clip(round((center_s - span_s / 2) * deep.DEEP_FS), 0, n - round(span_s * deep.DEEP_FS))
    )
    width = round(span_s * deep.DEEP_FS)
    total = np.zeros((len(deep.CHANNELS), width))
    counts = np.zeros(width)
    probabilities = []
    for offset in range(0, width - deep.WINDOW + 1, round(step_s * deep.DEEP_FS)):
        start = first + offset
        sal, prob = deep.saliency(model, prepared[:, start : start + deep.WINDOW])
        total[:, offset : offset + deep.WINDOW] += sal
        counts[offset : offset + deep.WINDOW] += 1
        probabilities.append(prob)
    sal = total / np.maximum(counts, 1)
    sal /= sal.max() or 1.0
    times = (first + np.arange(width)) / deep.DEEP_FS
    return times, prepared[:, first : first + width], sal, float(np.max(probabilities))


@st.cache_data
def cnn_test_scores():
    return pd.read_csv(CNN_FILES[2])


def cnn_metadata():
    return json.loads(CNN_FILES[1].read_text(encoding="utf-8"))


def cnn_table_row():
    """Ligne du tableau comparatif, mêmes métriques que les 38 configurations (seuil 0,5)."""
    from sklearn.metrics import average_precision_score

    scores = cnn_test_scores()
    y = scores.Class.to_numpy()
    metrics = eeg.score_predictions(y, (scores.cnn.to_numpy() >= 0.5).astype(int))
    return {
        "configuration": "deep__original__cnn",
        "Configuration": "CNN 1D · perte pondérée · 19 voies brutes",
        "Modèle": "CNN 1D",
        "Caractéristiques": "signal brut",
        "Stratégie": "Perte pondérée",
        "average_precision_mean": cnn_metadata()["cv"]["average_precision_mean"],
        "average_precision": float(average_precision_score(y, scores.cnn)),
        **metrics,
    }


def calibration(y, scores, bins=10):
    """Courbe de fiabilité (classes de largeur égale) et score de Brier."""
    from sklearn.metrics import brier_score_loss

    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(scores, edges) - 1, 0, bins - 1)
    rows = [
        (scores[which == b].mean(), y[which == b].mean(), int((which == b).sum()))
        for b in range(bins)
        if (which == b).any()
    ]
    predicted, observed, counts = map(np.array, zip(*rows, strict=True))
    return predicted, observed, counts, float(brier_score_loss(y, scores))


@st.cache_data
def cohort_series():
    """Séries par enregistrement (lecture seule des artefacts) pour les mini-courbes des cartes."""
    counts = pd.read_csv(ARTIFACTS / "class_counts.csv").sort_values("record_id")
    events = pd.read_csv(ARTIFACTS / "annotations_seconds.csv")
    per_record = events.groupby("record_id").size().reindex(counts.record_id, fill_value=0)
    loro = loro_table()
    meta = metadata()
    rf = (
        loro[
            (loro.configuration == meta["configuration"])
            & (loro.postprocessing == "Post-traitement")
        ].sort_values("record_id")
        if loro is not None
        else None
    )
    return {
        "duration_min": (counts.duration_s / 60).tolist(),
        "windows": counts.windows.tolist(),
        "events": per_record.tolist(),
        "positive_share": (counts.positive_windows / counts.windows).tolist(),
        "loro_f1": rf.f1.tolist() if rf is not None else [],
        "loro_false_alarms": rf.false_alarms_per_hour.tolist() if rf is not None else [],
    }


# ---------- Pages pédagogiques : extraits, spectres, étapes du pipeline, erreurs ----------


def excerpt(record_id, start_s, end_s, channels=None, lowpass=True):
    """Tranche filtrée de l'enregistrement (affichage) : temps, voies, noms."""
    time_values, _, names, _ = multichannel(record_id)
    filtered = display_filter(record_id, lowpass)
    keep = [names.index(c) for c in (channels or names) if c in names]
    i0, i1 = np.searchsorted(time_values, [start_s, end_s])
    return time_values[i0:i1], filtered[keep, i0:i1], [names[k] for k in keep]


def excerpt_profile(record_id, start_s, end_s):
    """Fréquence dominante (maximum du spectre moyen des voies EEG, 1–30 Hz) et amplitude efficace."""
    time_values, _, names, fs = multichannel(record_id)
    filtered = display_filter(record_id, False)
    i0, i1 = np.searchsorted(time_values, [start_s, end_s])
    spectrum, frequencies = eeg.one_sided_spectrum(filtered[:, i0:i1].astype(float), fs)
    mean = spectrum.mean(axis=0)
    useful = (frequencies >= 1) & (frequencies < 30)
    return {
        "dominant_hz": float(frequencies[useful][np.argmax(mean[useful])]),
        "rms": float(np.sqrt(np.mean(filtered[:, i0:i1].astype(float) ** 2))),
        "resolution_hz": float(frequencies[1] - frequencies[0]),
        "channels": len(names),
    }


def calm_excerpt(record_id, length_s, margin_s=30.0):
    """Passage typique loin de toute crise annotée : l'amplitude médiane parmi les candidats."""
    time_values, _, _, fs = multichannel(record_id)
    filtered = display_filter(record_id, False)
    events = annotated_events(record_id)
    duration = float(time_values[-1])
    candidates = []
    for start in np.arange(10.0, duration - length_s - 10, length_s / 2):
        end = start + length_s
        if all(end + margin_s < a or start - margin_s > b for a, b in events):
            i0, i1 = np.searchsorted(time_values, [start, end])
            candidates.append(
                (float(np.sqrt(np.mean(filtered[:, i0:i1].astype(float) ** 2))), start)
            )
    if not candidates:
        return None
    candidates.sort()
    return candidates[len(candidates) // 2][1]


@st.cache_data(show_spinner="Calcul des spectres…")
def class_spectra(record_id):
    """Spectre moyen (voie F7 filtrée) des fenêtres en crise et hors crise, comme dans le pipeline."""
    time_values, _, names, fs = multichannel(record_id)
    if eeg.CHANNEL not in names:
        return None
    signal = display_filter(record_id, False)[names.index(eeg.CHANNEL)].astype(float)
    windows, _, labels, _, _ = eeg.segment_signal(time_values, signal, annotated_events(record_id))
    spectrum, frequencies = eeg.one_sided_spectrum(windows, fs)
    keep = frequencies <= 30
    return {
        "frequencies": frequencies[keep],
        "crisis": spectrum[labels == 1][:, keep].mean(axis=0),
        "normal": spectrum[labels == 0][:, keep].mean(axis=0),
        "n_crisis": int((labels == 1).sum()),
        "n_normal": int((labels == 0).sum()),
    }


def window_labels(record_id, start_s, end_s):
    """Fenêtres de 2 s autour d'une crise : début, part annotée, étiquette (règle des 50 %)."""
    time_values, _, names, fs = multichannel(record_id)
    signal = display_filter(record_id, False)[names.index(eeg.CHANNEL)].astype(float)
    _, starts, labels, coverage, _ = eeg.segment_signal(
        time_values, signal, annotated_events(record_id)
    )
    t = time_values[starts]
    keep = (t >= start_s) & (t + 2 <= end_s)
    return t[keep], coverage[keep], labels[keep]


def decision_steps(result):
    """Étapes du post-traitement sur le score lissé : seuil, puis durée minimale, puis épisodes."""
    params = result["params"]
    scores = np.asarray(result["scores"]["value"])
    above = scores >= params["threshold"]
    kept = eeg.drop_short_runs(above, params["min_windows"])
    return np.asarray(result["scores"]["time"]), scores, above, kept


@st.cache_data(show_spinner="Recherche des erreurs du modèle…")
def error_catalogue(model, include_training=False):
    """Crises manquées et fausses alarmes, enregistrement par enregistrement, avec le réglage retenu."""
    rows = []
    for rid in record_options():
        row = catalogue()[rid]
        if row["status"] not in ("test", "apprentissage") or (
            row["status"] == "apprentissage" and not include_training
        ):
            continue
        try:
            result = cnn_analyse(rid) if model == "cnn" else analyse_record(rid)
        except Exception:  # noqa: BLE001 — un enregistrement illisible n'empêche pas les autres
            continue
        truth = [(e["start_s"], e["end_s"]) for e in (result["reference"] or {}).get("events", [])]
        detected = [(e["start_s"], e["end_s"]) for e in result["events"]]
        matches = eeg.match_intervals(truth, detected)[0]
        hit_truth = {i for i, _, _ in matches}
        hit_detected = {j for _, j, _ in matches}
        times, scores = np.asarray(result["scores"]["time"]), np.asarray(result["scores"]["value"])

        def peak(a, b, times=times, scores=scores):
            inside = (times >= a - 1) & (times <= b + 1)
            return float(scores[inside].max()) if inside.any() else float("nan")

        def best_overlap(a, b, others):
            best = 0.0
            for c, d in others:
                inter = max(0.0, min(b, d) - max(a, c))
                best = max(best, inter / (max(b, d) - min(a, c)))
            return best

        for i, (a, b) in enumerate(truth):
            if i not in hit_truth:
                rows.append({"record": rid, "status": row["status"], "kind": "manquee", "start": a, "end": b,
                             "peak": peak(a, b), "overlap": best_overlap(a, b, detected), "n": i + 1})  # fmt: skip
        for j, (a, b) in enumerate(detected):
            if j not in hit_detected:
                rows.append({"record": rid, "status": row["status"], "kind": "fausse_alarme", "start": a, "end": b,
                             "peak": peak(a, b), "overlap": best_overlap(a, b, truth), "n": j + 1})  # fmt: skip
    return pd.DataFrame(rows)


def wilson(successes, total, z=1.96):
    """Intervalle de confiance de Wilson à 95 % pour une proportion."""
    if not total:
        return float("nan"), float("nan")
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return float(centre - half), float(centre + half)


@st.cache_data
def seizure_durations():
    events = pd.read_csv(ARTIFACTS / "annotations_seconds.csv")
    counts = pd.read_csv(ARTIFACTS / "class_counts.csv")
    return events[events.record_id.isin(counts.record_id)].duration_s.to_numpy()


# ---------- Lecteur EEG (composant navigateur) ----------


@st.cache_data(max_entries=6, show_spinner="Préparation du tracé…")
def viewer_signal(record_id, channels, lowpass):
    """Voies filtrées en Int16 + base64. Avec le passe-bas 40 Hz, une valeur sur deux suffit (128 Hz)."""
    import base64

    time_values, _, names, fs = multichannel(record_id)
    filtered = display_filter(record_id, lowpass)
    keep = [names.index(c) for c in channels if c in names]
    factor = 2 if lowpass else 1
    data = filtered[keep, ::factor].astype(np.float32)
    step = float(np.max(np.abs(data)) / 32000) or 1.0
    encoded = [
        base64.b64encode(np.round(row / step).astype("<i2").tobytes()).decode("ascii")
        for row in data
    ]
    return {
        "fs": fs / factor,
        "step": step,
        "channels": [names[k].removeprefix("EEG") for k in keep],
        "data": encoded,
    }
