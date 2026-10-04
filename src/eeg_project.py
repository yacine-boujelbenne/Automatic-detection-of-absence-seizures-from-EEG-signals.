"""Pipeline reproductible du mini-projet, avec horaires corrigés et séparation groupée."""

import hashlib
import importlib.metadata
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import joblib
import numpy as np
import openpyxl
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CHANNEL = "EEGF7"
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
FEATURES = [
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "amplitude",
    "rms",
    "energy",
    "power_delta",
    "power_theta",
    "power_alpha",
    "power_beta",
]
FEATURE_LABELS = [
    "Moyenne",
    "Écart-type",
    "Variance",
    "Minimum",
    "Maximum",
    "Amplitude",
    "RMS",
    "Énergie",
    "Puissance Delta",
    "Puissance Theta",
    "Puissance Alpha",
    "Puissance Beta",
]
MODELS = {"dt": "Arbre de décision", "rf": "Forêt aléatoire", "knn": "KNN", "svm": "SVM"}
STRATEGIES = {
    "original": "Sans équilibrage",
    "under": "Sous-échantillonnage",
    "over": "Suréchantillonnage",
    "smote": "SMOTE",
}


def load_annotations(root):
    """Charge les cellules utiles du classeur mis à jour et conserve toutes les anomalies."""
    path = root / "Annotations MAJ.xlsx"
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        xml = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        populated = [
            int(row.attrib["r"])
            for row in xml.findall("s:sheetData/s:row", ns)
            if row.find("s:c/s:v", ns) is not None or row.find("s:c/s:is", ns) is not None
        ]
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    rows = list(sheet.iter_rows(max_row=max(populated), values_only=True))
    columns = [str(value) if value is not None else "record_id" for value in rows[0]]
    raw = pd.DataFrame(rows[1:], columns=columns)
    raw["excel_row"] = np.arange(2, len(raw) + 2)
    raw["record_id"] = raw["record_id"].astype(str).str.strip()
    raw = raw[raw.record_id.str.fullmatch(r"\d{6}[A-H]-[A-H]")].copy()
    workbook.close()
    raw["duplicate"] = raw.record_id.duplicated(keep=False)
    events, audit = [], []
    for _, row in raw.iterrows():
        rid = row.record_id
        npz = root / "data/exports" / f"{rid.replace('-', '_')}.npz"
        duration = np.nan
        if npz.exists():
            with np.load(npz, allow_pickle=False) as bundle:
                duration = (
                    len(bundle["row_times_ms"]) * np.median(np.diff(bundle["row_times_ms"])) / 1000
                )
        issues = []
        if row.duplicate:
            issues.append("Doublon conflictuel : aucune ligne choisie arbitrairement")
        if not npz.exists():
            issues.append("Export absent")
        if pd.isna(row.DebutTrace):
            issues.append("Horaire de début du tracé absent")
        row_events = []
        for column in columns:
            match = re.fullmatch(r"(?:DebCE|DebutCE|DebtCE)(\d+)", column)
            if not match:
                continue
            start, end = row[column], row.get("FinCE" + match.group(1))
            if pd.isna(start) and pd.isna(end):
                continue
            if pd.isna(start) or pd.isna(end) or pd.isna(row.DebutTrace):
                issues.append("Paire début/fin ou origine incomplète")
                continue
            start_s = round((float(start) - float(row.DebutTrace)) * 86400, 6)
            end_s = round((float(end) - float(row.DebutTrace)) * 86400, 6)
            valid = start_s >= 0 and end_s > start_s and end_s <= duration
            if not valid:
                issues.append(f"Crise {match.group(1)} hors limites du signal")
            event = {
                "record_id": rid,
                "excel_row": int(row.excel_row),
                "event": int(match.group(1)),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "valid_bounds": valid,
            }
            events.append(event)
            row_events.append(event)
        if not row_events:
            issues.append("Aucune annotation exploitable")
        audit.append(
            {
                "record_id": rid,
                "excel_row": int(row.excel_row),
                "age": row.Age,
                "sex": row.Sexe,
                "declared_events": row.NombreCE,
                "actual_events": len(row_events),
                "duration_s": duration,
                "included": not issues,
                "reason": " ; ".join(dict.fromkeys(issues)) or "Retenu",
                "count_mismatch": row.NombreCE != len(row_events),
            }
        )
    audit = pd.DataFrame(audit)
    events = pd.DataFrame(events)
    included = set(audit.loc[audit.included, "record_id"])
    cohort = audit.loc[audit.included].reset_index(drop=True)
    clean_events = events[events.record_id.isin(included)].reset_index(drop=True)
    unannotated = sorted(
        {p.stem.replace("_", "-") for p in (root / "data/exports").glob("*.npz")}
        - set(audit.record_id)
    )
    return cohort, clean_events, audit, events, unannotated


def load_trace(root, record_id, validation, channel=CHANNEL):
    """Charge un canal fixé avant apprentissage et reconstruit le temps à partir des blocs source."""
    info = validation.set_index("record_id").loc[record_id]
    with np.load(
        root / "data/exports" / f"{record_id.replace('-', '_')}.npz", allow_pickle=False
    ) as bundle:
        names = bundle["channel_names"].tolist()
        if channel not in names:
            raise ValueError(f"{record_id} : canal commun {channel} absent.")
        signal = bundle["signals"][names.index(channel)].astype(float)
        row_times = bundle["row_times_ms"].astype(float) / 1000
    block = int(info.samples_per_time_row)
    dt = float(info.time_step_s) / block
    time_values = (row_times - row_times[0])[:, None] + np.arange(block)[None, :] * dt
    time_values = time_values.ravel()
    if len(time_values) != len(signal) or not np.isfinite(signal).all():
        raise ValueError("Temps et signal incompatibles ou valeurs manquantes.")
    measured_fs = 1 / float(np.median(np.diff(time_values)))
    if not np.isclose(measured_fs, info.fs_hz):
        raise ValueError("Fréquence mesurée incompatible avec les blocs source.")
    return time_values, signal, measured_fs


def sample_mask(time_values, intervals):
    """Applique l'intervalle fermé [début, fin], imprimé à la page 4 du PDF."""
    mask = np.zeros(len(time_values), dtype=bool)
    for start, end in intervals:
        mask |= (time_values >= start) & (time_values <= end)
    return mask


def segment_signal(time_values, signal, intervals, seconds=2.0, overlap=0.5):
    """Construit des fenêtres complètes et leur couverture réelle des crises."""
    fs = 1 / float(np.median(np.diff(time_values)))
    n = round(seconds * fs)
    hop = round(n * (1 - overlap))
    if n < 2 or hop < 1 or not 0 <= overlap < 1:
        raise ValueError("Paramètres de segmentation invalides.")
    starts = np.arange(0, len(signal) - n + 1, hop)
    windows = np.lib.stride_tricks.sliding_window_view(signal, n)[::hop]
    mask = sample_mask(time_values, intervals)
    cumulative = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    coverage = (cumulative[starts + n] - cumulative[starts]) / n
    return windows, starts, (coverage >= 0.5).astype(np.int8), coverage, mask


def extract_features(windows, fs):
    """Calcule les huit statistiques et les quatre puissances FFT unilatérales."""
    n = windows.shape[1]
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    energy = (windows**2).sum(axis=1)
    lo, hi = windows.min(axis=1), windows.max(axis=1)
    values = np.column_stack((mean, std, std**2, lo, hi, hi - lo, np.sqrt(energy / n), energy))
    spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2 / n**2
    scale = np.full(spectrum.shape[1], 2.0)
    scale[0] = 1
    if n % 2 == 0:
        scale[-1] = 1
    spectrum *= scale
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    powers = [
        spectrum[:, (frequencies >= a) & (frequencies < b)].sum(axis=1) for a, b in BANDS.values()
    ]
    return pd.DataFrame(np.column_stack((values, *powers)), columns=FEATURES)


def build_features(root, cohort, events, validation):
    """Extrait les fenêtres de chaque tracé sans concaténer les signaux entre eux."""
    frames, counts = [], []
    for rid in cohort.record_id:
        t, signal, fs = load_trace(root, rid, validation)
        intervals = list(
            events.loc[events.record_id == rid, ["start_s", "end_s"]].itertuples(
                index=False, name=None
            )
        )
        windows, starts, labels, coverage, mask = segment_signal(t, signal, intervals)
        frame = extract_features(windows, fs)
        frame["record_id"] = rid
        frame["start_sample"] = starts
        frame["start_s"] = t[starts]
        frame["end_s"] = t[starts] + windows.shape[1] / fs
        frame["coverage"] = coverage
        frame["Class"] = labels
        frames.append(frame)
        counts.append(
            {
                "record_id": rid,
                "samples": len(signal),
                "positive_samples": int(mask.sum()),
                "windows": len(labels),
                "positive_windows": int(labels.sum()),
                "duration_s": len(signal) / fs,
                "fs_hz": fs,
                "channel": CHANNEL,
                "missing_values": int(np.isnan(signal).sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(counts)


def choose_split(frame):
    """Fixe un test groupé proche de 20 %, stratifié approximativement sans regarder les scores."""
    groups = frame.record_id.to_numpy()
    y = frame.Class.to_numpy()
    splitter = GroupShuffleSplit(n_splits=256, test_size=0.2, random_state=SEED)
    candidates = []
    for train, test in splitter.split(frame, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        distance = (
            abs(len(test) / len(y) - 0.2)
            + abs(y[train].mean() - y.mean())
            + abs(y[test].mean() - y.mean())
        )
        candidates.append((distance, train, test))
    if not candidates:
        raise ValueError("Impossible de construire une séparation avec les deux classes.")
    _, train, test = min(candidates, key=lambda row: row[0])
    assert set(groups[train]).isdisjoint(groups[test])
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            frame.iloc[train], y[train], groups[train]
        )
    )
    for a, b in folds:
        assert set(groups[train][a]).isdisjoint(groups[train][b])
        if len(np.unique(y[train][a])) < 2 or len(np.unique(y[train][b])) < 2:
            raise ValueError("Un pli ne contient pas les deux classes.")
    return train, test, folds


def make_model(model_key, strategy, positive_count):
    """Encapsule normalisation, éventuel rééchantillonnage et classifieur dans un pipeline."""
    classifiers = {
        "dt": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
        "rf": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "svm": SVC(C=1.0, kernel="rbf", cache_size=512, random_state=SEED),
    }
    samplers = {
        "original": "passthrough",
        "under": RandomUnderSampler(random_state=SEED),
        "over": RandomOverSampler(random_state=SEED),
        "smote": SMOTE(random_state=SEED, k_neighbors=min(5, positive_count - 1)),
    }
    if positive_count < 2:
        raise ValueError("Trop peu de fenêtres positives pour cette comparaison.")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("sampling", samplers[strategy]),
            ("model", classifiers[model_key]),
        ]
    )


def score_predictions(y, predictions):
    """Renvoie toutes les métriques demandées et les quatre cases de la matrice."""
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def train_comparison(frame, train, test, folds, output_dir):
    """Sélectionne en validation interne, entraîne les 16 configurations et réserve les prédictions test."""
    X = frame[FEATURES]
    y = frame.Class.to_numpy()
    cv_rows, models = [], {}
    for strategy in STRATEGIES:
        for key in MODELS:
            name = f"{strategy}__{key}"
            started = time.perf_counter()
            for fold, (a, b) in enumerate(folds, 1):
                fit_indices, valid_indices = train[a], train[b]
                pipe = make_model(key, strategy, int(y[fit_indices].sum()))
                pipe.fit(X.iloc[fit_indices], y[fit_indices])
                pred = pipe.predict(X.iloc[valid_indices])
                cv_rows.append(
                    {
                        "configuration": name,
                        "strategy": strategy,
                        "model": key,
                        "fold": fold,
                        **score_predictions(y[valid_indices], pred),
                    }
                )
            pipe = make_model(key, strategy, int(y[train].sum()))
            pipe.fit(X.iloc[train], y[train])
            models[name] = pipe
            print(
                f"Entraîné : {STRATEGIES[strategy]} / {MODELS[key]} ({time.perf_counter() - started:.1f} s)",
                flush=True,
            )
            pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby(["configuration", "strategy", "model"], sort=False)[
        ["accuracy", "precision", "recall", "f1"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = (
        summary.reset_index()
        .sort_values(
            ["f1_mean", "recall_mean", "precision_mean", "configuration"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    best = summary.iloc[0].configuration
    # Le choix est fixé avant de calculer les résultats du test final.
    test_rows, predictions, matrices = [], {}, {}
    for name, pipe in models.items():
        strategy, key = name.split("__")
        pred = pipe.predict(X.iloc[test])
        predictions[name] = pred
        matrices[name] = confusion_matrix(y[test], pred, labels=[0, 1])
        decision = (
            pipe.decision_function(X.iloc[test])
            if key == "svm"
            else pipe.predict_proba(X.iloc[test])[:, 1]
        )
        n_positive, n_negative = int(y[train].sum()), int((y[train] == 0).sum())
        after_positive = n_positive if strategy in ("original", "under") else n_negative
        after_negative = n_negative if strategy != "under" else n_positive
        test_rows.append(
            {
                "configuration": name,
                "strategy": strategy,
                "model": key,
                "train_positive_before": n_positive,
                "train_negative_before": n_negative,
                "train_positive_after": after_positive,
                "train_negative_after": after_negative,
                "average_precision": average_precision_score(y[test], decision),
                **score_predictions(y[test], pred),
            }
        )
    results = pd.DataFrame(test_rows)
    results.to_csv(output_dir / "test_results.csv", index=False)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    prediction_table = frame.iloc[test][["record_id", "start_s", "end_s", "Class"]].reset_index(
        drop=True
    )
    for name, pred in predictions.items():
        prediction_table[name] = pred
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)
    joblib.dump(models[best], output_dir / "best_model.joblib")
    (output_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "configuration": best,
                "selection": "F1 moyen en validation interne groupée ; rappel puis précision départagent les égalités",
                "features": FEATURES,
                "channel": CHANNEL,
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "models": models,
        "cv": cv,
        "cv_summary": summary,
        "results": results,
        "best": best,
        "predictions": predictions,
        "matrices": matrices,
        "test_indices": test,
        "train_indices": train,
    }


def run_training(frame, train, test, folds, output_dir, force=False):
    """Invalide les modèles mis en cache dès que le code, les données ou les bibliothèques changent."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(np.asarray(train, dtype=np.int64).tobytes())
    digest.update(np.asarray(test, dtype=np.int64).tobytes())
    for a, b in folds:
        digest.update(np.asarray(a, dtype=np.int64).tobytes())
        digest.update(np.asarray(b, dtype=np.int64).tobytes())
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn")
    }
    digest.update(json.dumps(versions, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    cache = output_dir / "trained_experiment.joblib"
    provenance_path = output_dir / "training_provenance.json"
    if not force and cache.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["fingerprint"] == fingerprint:
            print(
                "Les 16 entraînements vérifiés sont relus du cache ; données, code et partitions identiques."
            )
            return joblib.load(cache)
    result = train_comparison(frame, train, test, folds, output_dir)
    joblib.dump(result, cache, compress=3)
    provenance_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "versions": versions,
                "configurations": 16,
                "cv_folds": 3,
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def reconstruct_intervals(windows, predictions):
    """Fusionne l'union temporelle des fenêtres positives d'un seul enregistrement."""
    if windows.record_id.nunique() != 1:
        raise ValueError("La reconstruction exige un seul enregistrement.")
    intervals = []
    for row, value in zip(windows.itertuples(index=False), predictions, strict=True):
        if not value:
            continue
        a, b = float(row.start_s), float(row.end_s)
        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
        else:
            intervals.append((a, b))
    return intervals


def match_intervals(truth, predictions, min_iou=0.3):
    """Apparie une à une les crises par IoU, avec seuil fixé avant évaluation."""
    scores = np.zeros((len(truth), len(predictions)))
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(predictions):
            intersection = max(0, min(b, d) - max(a, c))
            scores[i, j] = intersection / (max(b, d) - min(a, c))
    matches = []
    if scores.size:
        eligible = scores >= min_iou
        # La cardinalité des appariements valides est prioritaire sur la somme des IoU.
        utility = eligible.astype(float) * (min(scores.shape) + 1) + scores * eligible
        rows, columns = linear_sum_assignment(utility, maximize=True)
        matches = [
            (int(i), int(j), float(scores[i, j]))
            for i, j in zip(rows, columns, strict=True)
            if eligible[i, j]
        ]
    return matches, len(predictions) - len(matches), len(truth) - len(matches)


def temporal_analysis(frame, events, experiment, output_dir):
    """Évalue le modèle sélectionné par fenêtre, enregistrement et événement."""
    test_frame = frame.iloc[experiment["test_indices"]].reset_index(drop=True)
    pred = experiment["predictions"][experiment["best"]]
    test_frame["prediction"] = pred
    window_rows, event_rows, details, predicted_rows = [], [], [], []
    for rid, group in test_frame.groupby("record_id", sort=True):
        truth = list(
            events.loc[events.record_id == rid, ["start_s", "end_s"]].itertuples(
                index=False, name=None
            )
        )
        detected = reconstruct_intervals(group, group.prediction.to_numpy())
        matches, fp, fn = match_intervals(truth, detected)
        tp = len(matches)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        event_rows.append(
            {
                "record_id": rid,
                "true_events": len(truth),
                "predicted_events": len(detected),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            }
        )
        window_rows.append({"record_id": rid, **score_predictions(group.Class, group.prediction)})
        matched_truth = {i: (j, iou) for i, j, iou in matches}
        matched_pred = {j for _, j, _ in matches}
        for i, (a, b) in enumerate(truth):
            j, iou = matched_truth.get(i, (None, 0))
            details.append(
                {
                    "record_id": rid,
                    "status": "TP" if j is not None else "FN",
                    "true_start_s": a,
                    "true_end_s": b,
                    "pred_start_s": detected[j][0] if j is not None else np.nan,
                    "pred_end_s": detected[j][1] if j is not None else np.nan,
                    "IoU": iou,
                }
            )
        for j, (a, b) in enumerate(detected):
            predicted_rows.append({"record_id": rid, "event": j + 1, "start_s": a, "end_s": b})
            if j not in matched_pred:
                details.append(
                    {
                        "record_id": rid,
                        "status": "FP",
                        "true_start_s": np.nan,
                        "true_end_s": np.nan,
                        "pred_start_s": a,
                        "pred_end_s": b,
                        "IoU": 0,
                    }
                )
    result = {
        "windows": pd.DataFrame(window_rows),
        "events": pd.DataFrame(event_rows),
        "details": pd.DataFrame(details),
        "predicted": pd.DataFrame(
            predicted_rows, columns=["record_id", "event", "start_s", "end_s"]
        ),
        "test_frame": test_frame,
    }
    for key in ("windows", "events", "details", "predicted"):
        result[key].to_csv(output_dir / f"temporal_{key}.csv", index=False)
    return result
