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
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew
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
# Bande des pointes-ondes à 3 Hz, signature EEG des absences typiques.
SPIKE_WAVE_BAND = (2.5, 4.5)
HIGHPASS_HZ = 0.5
MIN_IOU = 0.3
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
IMPROVED_FEATURES = [
    "log_std",
    "log_amplitude",
    "log_line_length",
    "skewness",
    "kurtosis",
    "log_power_delta",
    "log_power_theta",
    "log_power_alpha",
    "log_power_beta",
    "rel_delta",
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "rel_spike_wave",
    "log_slow_fast_ratio",
    "peak_frequency",
]
IMPROVED_LABELS = [
    "log écart-type",
    "log amplitude",
    "log longueur de ligne",
    "Asymétrie",
    "Kurtosis",
    "log puissance Delta",
    "log puissance Theta",
    "log puissance Alpha",
    "log puissance Beta",
    "Delta relative",
    "Theta relative",
    "Alpha relative",
    "Beta relative",
    "Pointes-ondes 2,5–4,5 Hz relative",
    "log Delta / (Alpha + Beta)",
    "Fréquence dominante",
]
FEATURE_SETS = {"base": FEATURES, "improved": IMPROVED_FEATURES}
FEATURE_SET_LABELS = {"base": "12 de l'énoncé", "improved": "enrichies"}
MODELS = {"dt": "Arbre de décision", "rf": "Forêt aléatoire", "knn": "KNN", "svm": "SVM"}
STRATEGIES = {
    "original": "Sans équilibrage",
    "under": "Sous-échantillonnage",
    "over": "Suréchantillonnage",
    "smote": "SMOTE",
    "balanced": "Pondération des classes",
}
# Les quatre stratégies de l'énoncé ; la pondération est une comparaison ajoutée.
ASSIGNMENT_STRATEGIES = ("original", "under", "over", "smote")
# Grille de post-traitement fixée avant l'apprentissage.
SMOOTHING = (1, 3, 5)
MIN_WINDOWS = (1, 2, 3, 4, 5)
THRESHOLD_QUANTILES = np.linspace(0.80, 0.995, 40)


def configurations():
    """38 configurations : 2 jeux × 5 stratégies × 4 modèles, sans KNN pondéré."""
    return [
        config_name(feature_set, strategy, model)
        for feature_set in FEATURE_SETS
        for strategy in STRATEGIES
        for model in MODELS
        if not (strategy == "balanced" and model == "knn")
    ]


def config_name(feature_set, strategy, model):
    """Identifiant stable d'une configuration."""
    return f"{feature_set}__{strategy}__{model}"


def parse_config(name):
    """Renvoie (jeu de caractéristiques, stratégie, modèle)."""
    feature_set, strategy, model = name.split("__")
    return feature_set, strategy, model


def config_label(name):
    """Libellé français d'une configuration."""
    feature_set, strategy, model = parse_config(name)
    return f"{MODELS[model]} · {STRATEGIES[strategy]} · caractéristiques {FEATURE_SET_LABELS[feature_set]}"


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


def one_sided_spectrum(windows, fs):
    """Puissance FFT unilatérale |rFFT|²/N², doublée hors DC et Nyquist."""
    n = windows.shape[1]
    spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2 / n**2
    scale = np.full(spectrum.shape[1], 2.0)
    scale[0] = 1
    if n % 2 == 0:
        scale[-1] = 1
    return spectrum * scale, np.fft.rfftfreq(n, d=1 / fs)


def band_power(spectrum, frequencies, low, high):
    """Somme des puissances dans [low, high[."""
    return spectrum[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)


def extract_features(windows, fs):
    """Calcule les huit statistiques et les quatre puissances FFT unilatérales."""
    n = windows.shape[1]
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    energy = (windows**2).sum(axis=1)
    lo, hi = windows.min(axis=1), windows.max(axis=1)
    values = np.column_stack((mean, std, std**2, lo, hi, hi - lo, np.sqrt(energy / n), energy))
    spectrum, frequencies = one_sided_spectrum(windows, fs)
    powers = [band_power(spectrum, frequencies, a, b) for a, b in BANDS.values()]
    return pd.DataFrame(np.column_stack((values, *powers)), columns=FEATURES)


def highpass(signal, fs, cutoff=HIGHPASS_HZ):
    """Retire la composante continue et la dérive lente (Butterworth d'ordre 4, phase nulle)."""
    sos = butter(4, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def extract_improved_features(windows, fs):
    """Caractéristiques sans dépendance au décalage continu, calculées sur le signal filtré."""
    eps = 1e-12
    spectrum, frequencies = one_sided_spectrum(windows, fs)
    powers = {name: band_power(spectrum, frequencies, a, b) for name, (a, b) in BANDS.items()}
    total = band_power(spectrum, frequencies, BANDS["delta"][0], BANDS["beta"][1]) + eps
    spike_wave = band_power(spectrum, frequencies, *SPIKE_WAVE_BAND)
    useful = (frequencies >= 1) & (frequencies < 30)
    peak = frequencies[useful][np.argmax(spectrum[:, useful], axis=1)]
    columns = {
        "log_std": np.log(windows.std(axis=1) + eps),
        "log_amplitude": np.log(np.ptp(windows, axis=1) + eps),
        "log_line_length": np.log(np.abs(np.diff(windows, axis=1)).mean(axis=1) + eps),
        "skewness": skew(windows, axis=1),
        "kurtosis": kurtosis(windows, axis=1),
        **{f"log_power_{name}": np.log(value + eps) for name, value in powers.items()},
        **{f"rel_{name}": value / total for name, value in powers.items()},
        "rel_spike_wave": spike_wave / total,
        "log_slow_fast_ratio": np.log(
            (powers["delta"] + eps) / (powers["alpha"] + powers["beta"] + eps)
        ),
        "peak_frequency": peak,
    }
    return pd.DataFrame(columns)[IMPROVED_FEATURES]


def robust_normalize(features):
    """Centre-réduit chaque caractéristique par la médiane et l'IQR du même enregistrement.

    Aucune étiquette n'est utilisée : seules les statistiques du tracé lui-même,
    dominé à plus de 95 % par l'activité de fond, servent de référence. Les valeurs
    sont bornées à ±10 IQR : les segments plats ou saturés ne dominent plus les distances.
    """
    median = features.median()
    iqr = features.quantile(0.75) - features.quantile(0.25)
    return ((features - median) / iqr.where(iqr > 1e-12, 1.0)).clip(-10, 10)


def recording_features(time_values, signal, fs, intervals=()):
    """Fenêtres, étiquettes et deux jeux de caractéristiques pour un seul enregistrement."""
    windows, starts, labels, coverage, mask = segment_signal(time_values, signal, list(intervals))
    filtered = highpass(signal, fs)
    filtered_windows = np.lib.stride_tricks.sliding_window_view(filtered, windows.shape[1])[starts]
    base = extract_features(windows, fs)
    improved = robust_normalize(extract_improved_features(filtered_windows, fs))
    frame = pd.concat([base, improved], axis=1)
    frame["start_sample"] = starts
    frame["start_s"] = time_values[starts]
    frame["end_s"] = time_values[starts] + windows.shape[1] / fs
    frame["coverage"] = coverage
    frame["Class"] = labels
    return frame, windows, mask


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
        frame, _, mask = recording_features(t, signal, fs, intervals)
        frame.insert(0, "record_id", rid)
        frames.append(frame)
        counts.append(
            {
                "record_id": rid,
                "samples": len(signal),
                "positive_samples": int(mask.sum()),
                "windows": len(frame),
                "positive_windows": int(frame.Class.sum()),
                "duration_s": len(signal) / fs,
                "fs_hz": fs,
                "channel": CHANNEL,
                "missing_values": int(np.isnan(signal).sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(counts)


def grouped_folds(frame, n_splits=3):
    """Plis stratifiés groupés par enregistrement ; indices relatifs au cadre fourni."""
    y = frame.Class.to_numpy()
    groups = frame.record_id.to_numpy()
    folds = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(
            frame, y, groups
        )
    )
    for a, b in folds:
        assert set(groups[a]).isdisjoint(groups[b])
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            raise ValueError("Un pli ne contient pas les deux classes.")
    return folds


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
    return train, test, grouped_folds(frame.iloc[train])


def make_model(model_key, strategy, positive_count):
    """Encapsule normalisation, éventuel rééchantillonnage et classifieur dans un pipeline."""
    if positive_count < 2:
        raise ValueError("Trop peu de fenêtres positives pour cette comparaison.")
    if strategy == "balanced" and model_key == "knn":
        raise ValueError("Le KNN n'accepte pas de pondération des classes.")
    weight = "balanced" if strategy == "balanced" else None
    classifiers = {
        "dt": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, class_weight=weight, random_state=SEED
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight=weight,
            random_state=SEED,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "svm": SVC(C=1.0, kernel="rbf", class_weight=weight, cache_size=1024, random_state=SEED),
    }
    samplers = {
        "original": "passthrough",
        "balanced": "passthrough",
        "under": RandomUnderSampler(random_state=SEED),
        "over": RandomOverSampler(random_state=SEED),
        "smote": SMOTE(random_state=SEED, k_neighbors=min(5, positive_count - 1)),
    }
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("sampling", samplers[strategy]),
            ("model", classifiers[model_key]),
        ]
    )


def model_scores(pipe, X):
    """Score continu de la classe crise : probabilité, ou marge du SVM."""
    model = pipe.named_steps["model"]
    if isinstance(model, SVC):
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, list(model.classes_).index(1)]


def default_threshold(name):
    """Seuil équivalent à predict() : 0 pour la marge du SVM, 0,5 sinon."""
    return 0.0 if parse_config(name)[2] == "svm" else 0.5


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


def fit_and_score(name, frame, fit_index, score_index):
    """Ajuste une configuration et renvoie le pipeline et les scores des fenêtres évaluées."""
    feature_set, strategy, model = parse_config(name)
    columns = FEATURE_SETS[feature_set]
    y = frame.Class.to_numpy()
    pipe = make_model(model, strategy, int(y[fit_index].sum()))
    pipe.fit(frame[columns].iloc[fit_index], y[fit_index])
    return pipe, model_scores(pipe, frame[columns].iloc[score_index])


def train_comparison(frame, train, test, folds, names=None):
    """Valide chaque configuration sur les plis internes, puis l'ajuste sur tout l'apprentissage."""
    names = names or configurations()
    y = frame.Class.to_numpy()
    cv_rows, models, oof, test_scores = [], {}, {}, {}
    for name in names:
        started = time.perf_counter()
        oof[name] = np.full(len(train), np.nan)
        for fold, (a, b) in enumerate(folds, 1):
            _, scores = fit_and_score(name, frame, train[a], train[b])
            oof[name][b] = scores
            truth = y[train[b]]
            cv_rows.append(
                {
                    "configuration": name,
                    "fold": fold,
                    "average_precision": average_precision_score(truth, scores),
                    **score_predictions(truth, scores >= default_threshold(name)),
                }
            )
        models[name], test_scores[name] = fit_and_score(name, frame, train, test)
        print(f"Entraîné : {name} ({time.perf_counter() - started:.1f} s)", flush=True)
    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby("configuration", sort=False)[
        ["accuracy", "precision", "recall", "f1", "average_precision"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    parts = summary.configuration.map(parse_config)
    summary.insert(1, "feature_set", parts.str[0])
    summary.insert(2, "strategy", parts.str[1])
    summary.insert(3, "model", parts.str[2])
    # Règle fixée a priori : AP moyenne en validation interne, indépendante du seuil.
    summary = summary.sort_values(
        ["average_precision_mean", "f1_mean", "configuration"], ascending=[False, False, True]
    ).reset_index(drop=True)
    best = summary.iloc[0]
    summary["tied_with_best"] = summary.average_precision_mean >= (
        best.average_precision_mean - best.average_precision_std
    )
    test_rows = []
    y_test = y[test]
    n_positive, n_negative = int(y[train].sum()), int((y[train] == 0).sum())
    for name in names:
        feature_set, strategy, model = parse_config(name)
        after_positive = n_negative if strategy in ("over", "smote") else n_positive
        after_negative = n_positive if strategy == "under" else n_negative
        test_rows.append(
            {
                "configuration": name,
                "feature_set": feature_set,
                "strategy": strategy,
                "model": model,
                "train_positive_before": n_positive,
                "train_negative_before": n_negative,
                "train_positive_after": after_positive,
                "train_negative_after": after_negative,
                "average_precision": average_precision_score(y_test, test_scores[name]),
                **score_predictions(y_test, test_scores[name] >= default_threshold(name)),
            }
        )
    return {
        "models": models,
        "cv": cv,
        "cv_summary": summary,
        "results": pd.DataFrame(test_rows),
        "best": best.configuration,
        "oof_scores": oof,
        "test_scores": test_scores,
        "train_indices": np.asarray(train),
        "test_indices": np.asarray(test),
    }


def fingerprint(*parts):
    """Empreinte du code, des données, des partitions et des versions des bibliothèques."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for part in parts:
        if isinstance(part, pd.DataFrame):
            digest.update(pd.util.hash_pandas_object(part, index=True).values.tobytes())
        elif isinstance(part, str):
            digest.update(part.encode())
        else:
            digest.update(np.asarray(part, dtype=np.int64).tobytes())
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn")
    }
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return digest.hexdigest(), versions


def cached(output_dir, stem, key, versions, compute, force=False):
    """Relit un calcul seulement si son empreinte est identique."""
    cache = output_dir / f"{stem}.joblib"
    provenance_path = output_dir / f"{stem}_provenance.json"
    if not force and cache.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["fingerprint"] == key:
            print(f"{stem} : relu du cache (code, données et partitions identiques).")
            return joblib.load(cache)
    result = compute()
    joblib.dump(result, cache, compress=3)
    provenance_path.write_text(
        json.dumps({"fingerprint": key, "versions": versions, "seed": SEED}, indent=2),
        encoding="utf-8",
    )
    return result


def run_training(frame, train, test, folds, output_dir, force=False):
    """Entraîne les 38 configurations, avec un cache invalidé dès qu'une entrée change."""
    key, versions = fingerprint(frame, train, test, *[x for fold in folds for x in fold])
    result = cached(
        output_dir,
        "trained_experiment",
        key,
        versions,
        lambda: train_comparison(frame, train, test, folds),
        force,
    )
    result["cv_summary"].to_csv(output_dir / "cv_summary.csv", index=False)
    result["cv"].to_csv(output_dir / "cv_results.csv", index=False)
    result["results"].to_csv(output_dir / "test_results.csv", index=False)
    joblib.dump(result["models"][result["best"]], output_dir / "best_model.joblib")
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


def intervals_from_labels(starts, ends, labels):
    """Version vectorisée de `reconstruct_intervals` pour des fenêtres triées de même durée."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(labels, dtype=np.int8), 0]))
    intervals = []
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        start, end = float(starts[a]), float(ends[b - 1])
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def match_intervals(truth, predictions, min_iou=MIN_IOU):
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


def smooth_scores(scores, width):
    """Moyenne glissante centrée sur `width` fenêtres, bords normalisés."""
    scores = np.asarray(scores, dtype=float)
    if width <= 1:
        return scores
    kernel = np.ones(width)
    return np.convolve(scores, kernel, mode="same") / np.convolve(
        np.ones(len(scores)), kernel, mode="same"
    )


def drop_short_runs(labels, min_windows):
    """Supprime les suites de fenêtres positives plus courtes que `min_windows`."""
    labels = np.asarray(labels, dtype=bool).copy()
    if min_windows <= 1 or not labels.any():
        return labels
    edges = np.flatnonzero(np.diff(np.r_[0, labels.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start < min_windows:
            labels[start:stop] = False
    return labels


def postprocess(scores, threshold, smoothing=1, min_windows=1):
    """Lissage, seuil puis durée minimale, sur les fenêtres ordonnées d'un enregistrement."""
    return drop_short_runs(smooth_scores(scores, smoothing) >= threshold, min_windows)


class Recordings:
    """Fenêtres regroupées par enregistrement, triées, avec leurs crises de référence."""

    def __init__(self, windows, events):
        self.windows = windows.reset_index(drop=True)
        self.groups = []
        starts = self.windows.start_s.to_numpy()
        ends = self.windows.end_s.to_numpy()
        for rid, index in self.windows.groupby("record_id", sort=True).indices.items():
            order = index[np.argsort(starts[index], kind="stable")]
            truth = list(
                events.loc[events.record_id == rid, ["start_s", "end_s"]].itertuples(
                    index=False, name=None
                )
            )
            hours = (ends[order].max() - starts[order].min()) / 3600
            self.groups.append((rid, order, starts[order], ends[order], truth, hours))

    def detect(self, scores, params):
        """Détections par enregistrement et étiquettes de fenêtres après post-traitement."""
        scores = np.asarray(scores, dtype=float)
        labels = np.zeros(len(scores), dtype=bool)
        detections = {}
        for rid, order, starts, ends, _, _ in self.groups:
            record = postprocess(
                scores[order], params["threshold"], params["smoothing"], params["min_windows"]
            )
            labels[order] = record
            detections[rid] = intervals_from_labels(starts, ends, record)
        return detections, labels

    def event_metrics(self, detections):
        """Comptes événementiels (IoU ≥ 0,3) et fausses alarmes par heure."""
        tp = fp = fn = 0
        hours = 0.0
        for rid, _, _, _, truth, record_hours in self.groups:
            matches, record_fp, record_fn = match_intervals(truth, detections.get(rid, []))
            tp, fp, fn = tp + len(matches), fp + record_fp, fn + record_fn
            hours += record_hours
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "event_TP": tp,
            "event_FP": fp,
            "event_FN": fn,
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": f1,
            "false_alarms_per_hour": fp / hours if hours else np.nan,
            "hours": hours,
        }

    def evaluate(self, scores, params):
        """Métriques des fenêtres et des événements pour des paramètres fixés."""
        detections, labels = self.detect(scores, params)
        y = self.windows.Class.to_numpy()
        metrics = {**score_predictions(y, labels.astype(int)), **self.event_metrics(detections)}
        return metrics, detections, labels


def default_params(name):
    """Sortie brute du classifieur : seuil par défaut, ni lissage ni durée minimale."""
    return {"threshold": default_threshold(name), "smoothing": 1, "min_windows": 1}


def tune_postprocessing(recordings, scores, name):
    """Choisit seuil, lissage et durée minimale sur des scores hors pli uniquement."""
    scores = np.asarray(scores, dtype=float)
    rows = []
    for smoothing in SMOOTHING:
        smoothed = np.concatenate(
            [smooth_scores(scores[order], smoothing) for _, order, *_ in recordings.groups]
        )
        thresholds = np.unique(
            np.r_[np.quantile(smoothed, THRESHOLD_QUANTILES), default_threshold(name)]
        )
        for threshold in thresholds:
            for min_windows in MIN_WINDOWS:
                params = {
                    "threshold": float(threshold),
                    "smoothing": smoothing,
                    "min_windows": min_windows,
                }
                detections, _ = recordings.detect(scores, params)
                rows.append({**params, **recordings.event_metrics(detections)})
    grid = pd.DataFrame(rows).sort_values(
        ["event_f1", "event_recall", "false_alarms_per_hour"], ascending=[False, False, True]
    )
    best = grid.iloc[0]
    params = {
        "threshold": float(best.threshold),
        "smoothing": int(best.smoothing),
        "min_windows": int(best.min_windows),
    }
    return params, grid.reset_index(drop=True)


def window_table(frame, index):
    """Colonnes nécessaires à la reconstruction temporelle."""
    return frame.iloc[index][["record_id", "start_s", "end_s", "Class"]].reset_index(drop=True)


def temporal_analysis(frame, events, experiment, output_dir):
    """Règle le post-traitement en validation interne, puis l'applique au test réservé."""
    name = experiment["best"]
    train, test = experiment["train_indices"], experiment["test_indices"]
    train_recordings = Recordings(window_table(frame, train), events)
    test_recordings = Recordings(window_table(frame, test), events)
    oof, scores = experiment["oof_scores"][name], experiment["test_scores"][name]
    params, grid = tune_postprocessing(train_recordings, oof, name)
    rows = []
    for stage, recordings, values in (
        ("Validation interne", train_recordings, oof),
        ("Test réservé", test_recordings, scores),
    ):
        for label, chosen in (("Sortie brute", default_params(name)), ("Post-traitement", params)):
            metrics, _, _ = recordings.evaluate(values, chosen)
            rows.append({"stage": stage, "postprocessing": label, **metrics})
    comparison = pd.DataFrame(rows)
    raw_detections, _ = test_recordings.detect(scores, default_params(name))
    detections, labels = test_recordings.detect(scores, params)
    test_frame = test_recordings.windows.copy()
    test_frame["score"] = scores
    test_frame["prediction"] = labels.astype(int)
    window_rows, event_rows, details, predicted_rows = [], [], [], []
    for rid, order, _, _, truth, hours in test_recordings.groups:
        group = test_frame.iloc[order]
        detected = detections[rid]
        matches, fp, fn = match_intervals(truth, detected)
        tp = len(matches)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        event_rows.append(
            {
                "record_id": rid,
                "true_events": len(truth),
                "raw_predicted_events": len(raw_detections[rid]),
                "predicted_events": len(detected),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                "false_alarms_per_hour": fp / hours,
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
        "configuration": name,
        "params": params,
        "grid": grid,
        "comparison": comparison,
        "windows": pd.DataFrame(window_rows),
        "events": pd.DataFrame(event_rows),
        "details": pd.DataFrame(details),
        "predicted": pd.DataFrame(
            predicted_rows, columns=["record_id", "event", "start_s", "end_s"]
        ),
        "raw_predicted": raw_detections,
        "test_frame": test_frame,
    }
    for key in ("windows", "events", "details", "predicted", "comparison", "grid"):
        result[key].to_csv(output_dir / f"temporal_{key}.csv", index=False)
    (output_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "configuration": name,
                "selection": "AP moyenne maximale en validation interne groupée (3 plis)",
                "postprocessing": params,
                "postprocessing_selection": "F1 événementiel (IoU ≥ 0,3) maximal sur les scores hors pli",
                "features": FEATURE_SETS[parse_config(name)[0]],
                "channel": CHANNEL,
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def loro_shortlist(best):
    """Configurations réévaluées en LORO : le choix interne et trois références fixées à l'avance."""
    references = ["base__smote__dt", "base__original__rf", "improved__original__rf"]
    return list(dict.fromkeys([best, *references]))


def leave_one_record_out(frame, events, names, output_dir, force=False):
    """Validation « un enregistrement exclu » sur toute la cohorte.

    Pour chaque tracé exclu, le post-traitement est réglé sur les scores hors pli
    des autres tracés (validation imbriquée en trois plis), puis appliqué au tracé exclu.
    """
    key, versions = fingerprint(frame, "|".join(names))

    def compute():
        rows = []
        records = sorted(frame.record_id.unique())
        ids = frame.record_id.to_numpy()
        y = frame.Class.to_numpy()
        for name in names:
            started = time.perf_counter()
            for rid in records:
                held, rest = np.flatnonzero(ids == rid), np.flatnonzero(ids != rid)
                oof = np.full(len(rest), np.nan)
                for a, b in grouped_folds(frame.iloc[rest]):
                    _, oof[b] = fit_and_score(name, frame, rest[a], rest[b])
                params, _ = tune_postprocessing(
                    Recordings(window_table(frame, rest), events), oof, name
                )
                _, scores = fit_and_score(name, frame, rest, held)
                held_recordings = Recordings(window_table(frame, held), events)
                for label, chosen in (
                    ("Sortie brute", default_params(name)),
                    ("Post-traitement", params),
                ):
                    metrics, _, _ = held_recordings.evaluate(scores, chosen)
                    rows.append(
                        {
                            "configuration": name,
                            "record_id": rid,
                            "postprocessing": label,
                            "average_precision": average_precision_score(y[held], scores),
                            **chosen,
                            **metrics,
                        }
                    )
            print(f"LORO : {name} ({time.perf_counter() - started:.0f} s)", flush=True)
        return pd.DataFrame(rows)

    result = cached(output_dir, "loro", key, versions, compute, force)
    result.to_csv(output_dir / "loro_results.csv", index=False)
    return result


def loro_summary(loro):
    """Médiane, quartiles et totaux poolés par configuration et par étape."""
    rows = []
    for (name, stage), group in loro.groupby(["configuration", "postprocessing"], sort=False):
        tp, fp, fn = group.event_TP.sum(), group.event_FP.sum(), group.event_FN.sum()
        wtp, wfp, wfn = group.TP.sum(), group.FP.sum(), group.FN.sum()
        rows.append(
            {
                "configuration": name,
                "postprocessing": stage,
                "f1_median": group.f1.median(),
                "f1_q1": group.f1.quantile(0.25),
                "f1_q3": group.f1.quantile(0.75),
                "pooled_window_f1": 2 * wtp / (2 * wtp + wfp + wfn),
                "ap_median": group.average_precision.median(),
                "event_recall": tp / (tp + fn),
                "event_precision": tp / (tp + fp) if tp + fp else 0.0,
                "event_f1": 2 * tp / (2 * tp + fp + fn),
                "false_alarms_per_hour": fp / group.hours.sum(),
            }
        )
    return pd.DataFrame(rows)
