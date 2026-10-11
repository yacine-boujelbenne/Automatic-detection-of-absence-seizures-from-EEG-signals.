"""Lecture contrôlée des fichiers et prédictions avec le pipeline du notebook."""

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src import eeg_project as eeg
from src.eeg_source_validation import cell_children, matrix_header
from tools import mcos_extract as mat

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 128 * 1024 * 1024
MAX_UNPACKED = 256 * 1024 * 1024
MAX_SAMPLES = 2_000_000


class InputError(ValueError):
    """Fichier incompatible ; message directement présentable dans l'interface."""


@dataclass
class Recording:
    """Un canal réel et son axe temporel mesuré."""

    time: np.ndarray
    signal: np.ndarray
    filename: str
    source: str
    fs: float
    sha256: str
    filtered: np.ndarray | None = None


def time_from_rows(row_times_ms, sample_count):
    """Reconstruit les temps des blocs réguliers en mesurant leur pas."""
    rows = np.asarray(row_times_ms, dtype=float)
    if rows.ndim != 1 or len(rows) < 2 or not np.isfinite(rows).all():
        raise InputError("Le vecteur temporel des blocs est invalide.")
    delta = np.diff(rows) / 1000
    if not (delta > 0).all() or not np.allclose(delta, np.median(delta), rtol=1e-6, atol=1e-9):
        raise InputError("Les blocs temporels doivent être réguliers et strictement croissants.")
    if sample_count % len(rows):
        raise InputError("Le nombre d'échantillons ne correspond pas aux blocs temporels.")
    block = sample_count // len(rows)
    if block < 1:
        raise InputError("Les blocs ne contiennent aucun échantillon.")
    return (
        (rows - rows[0])[:, None] / 1000 + np.arange(block)[None, :] * np.median(delta) / block
    ).ravel()


def read_mat(payload):
    """Lit les cellules numériques MCOS par leurs balises, sans balayage flottant."""
    with TemporaryDirectory(prefix="eeg-upload-") as directory:
        path = Path(directory) / "recording.mat"
        path.write_bytes(payload)
        buf = mat.load_workspace(path)
    wrappers = [
        node.fields["FileWrapper__"]
        for node in mat.walk_stream(buf)
        if "FileWrapper__" in node.fields
    ]
    if len(wrappers) != 1 or "{2}" not in wrappers[0].fields or b"millis\x00fmt" not in buf:
        raise InputError(
            "Ce fichier MAT n'a pas la structure temporelle du projet. Utilisez le fichier de signal *_d.mat ou un CSV Time,EEGF7."
        )
    rows = np.asarray(wrappers[0].fields["{2}"].data, dtype=float)
    if rows.ndim != 1 or len(rows) < 2:
        raise InputError("Le fichier MAT ne contient pas de temps exploitables.")
    channels, name_cells = [], []
    for _, _, dtype, size, offset, _ in mat._iter_elements(buf):
        if dtype != mat.MI_MATRIX or size == 0:
            continue
        end = offset + size
        try:
            cls, shape, cursor = matrix_header(buf, offset, end)
        except (ValueError, mat.ParseError):
            continue
        if cls != mat.MX_CELL:
            continue
        if shape == (len(rows), 1):
            channels.append((cursor, end))
        elif len(shape) == 2 and shape[0] == 1 and 1 <= shape[1] <= 256:
            children = cell_children(buf, cursor, end)
            if children and all(child[0] == mat.MX_CHAR for child in children):
                names = []
                for _, _, child, _ in children:
                    text_type, raw, _ = child.read_element()
                    names.append(
                        raw.decode("utf-16-le" if text_type == mat.MI_UTF16 else "utf-8").rstrip(
                            "\x00"
                        )
                    )
                name_cells.append(names)
    candidates = {
        tuple(names)
        for names in name_cells
        if len(names) == len(channels) and names.count(eeg.CHANNEL) == 1
    }
    if len(candidates) != 1:
        raise InputError("Impossible d'identifier sans ambiguïté le canal EEGF7 dans ce MAT.")
    names = list(candidates.pop())
    cursor, end = channels[names.index(eeg.CHANNEL)]
    blocks = cell_children(buf, cursor, end)
    if len(blocks) != len(rows):
        raise InputError("Les blocs EEG et les temps ne correspondent pas.")
    values, sizes = [], set()
    for cls, shape, child, _ in blocks:
        if cls != mat.MX_DOUBLE:
            raise InputError("Un bloc EEG n'est pas numérique.")
        dtype, block = child.read_numeric_array()
        if dtype != mat.MI_DOUBLE or np.prod(shape) != len(block):
            raise InputError("Un bloc EEG est incomplet.")
        sizes.add(len(block))
        # Même précision que les exports float32 utilisés pendant l'apprentissage.
        values.append(block.astype(np.float32))
    if len(sizes) != 1 or sum(map(len, values)) > MAX_SAMPLES:
        raise InputError("Longueur des blocs incohérente ou enregistrement trop long.")
    signal = np.concatenate(values).astype(float)
    return (
        time_from_rows(rows, len(signal)),
        signal,
        "Cellules numériques du fichier MATLAB original",
    )


def read_npz(payload, digest):
    """Accepte les exports source vérifiés ou un axe explicite par échantillon."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if (
            len(archive.infolist()) > 32
            or sum(entry.file_size for entry in archive.infolist()) > MAX_UNPACKED
        ):
            raise InputError("L'archive NPZ décompressée est trop volumineuse.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        if not {"signals", "channel_names"}.issubset(bundle.files):
            raise InputError("Le NPZ doit contenir signals, channel_names et un axe temporel.")
        names = bundle["channel_names"].tolist()
        signals = bundle["signals"]
        if (
            not isinstance(names, list)
            or names.count(eeg.CHANNEL) != 1
            or signals.ndim != 2
            or signals.shape[0] != len(names)
        ):
            raise InputError(
                "Le NPZ doit contenir exactement un canal EEGF7, avec signals organisé en canaux × échantillons."
            )
        if (
            signals.shape[1] > MAX_SAMPLES
            or not np.issubdtype(signals.dtype, np.number)
            or np.iscomplexobj(signals)
        ):
            raise InputError(
                "Le signal NPZ doit être réel et contenir au plus 2 millions d'échantillons."
            )
        signal = signals[names.index(eeg.CHANNEL)].astype(float)
        if "time_s" in bundle.files:
            times = np.asarray(bundle["time_s"], dtype=float)
            source = "Axe time_s fourni, un temps en secondes par échantillon"
        elif "row_times_ms" in bundle.files:
            checks = [row["export_sha256"] for row in load_catalogue().values()]
            validation = ROOT / "artifacts/final/source_validation.json"
            if validation.exists():
                checks += [
                    row["export_sha256"]
                    for row in json.loads(validation.read_text(encoding="utf-8"))
                ]
            matches = set(checks) & {digest}
            if len(matches) != 1:
                raise InputError(
                    "Cet export par blocs n'a pas été vérifié contre son fichier source. Déposez le MAT original, ou un NPZ avec time_s explicite."
                )
            times = time_from_rows(bundle["row_times_ms"], len(signal))
            source = (
                "Export du projet : empreinte identique à l'export vérifié contre le MAT source"
            )
        else:
            raise InputError(
                "Axe temporel absent : une fréquence déclarée seule ne suffit pas. Fournissez time_s."
            )
        if "fs" in bundle.files:
            stored = np.asarray(bundle["fs"])
            if (
                stored.size != 1
                or len(times) < 2
                or not np.isclose(float(stored.item()), 1 / np.median(np.diff(times)), rtol=1e-4)
            ):
                raise InputError("La fréquence déclarée dans le NPZ diffère des temps fournis.")
    return times, signal, source


def load_recording(payload, filename, expected_fs):
    """Contrôle le format, le canal, les valeurs et la fréquence réellement mesurée."""
    if not payload or len(payload) > MAX_UPLOAD:
        raise InputError("Le fichier doit être non vide et inférieur à 128 Mio.")
    digest = hashlib.sha256(payload).hexdigest()
    extension = Path(filename).suffix.lower()
    try:
        if extension == ".mat":
            times, signal, source = read_mat(payload)
        elif extension == ".npz":
            times, signal, source = read_npz(payload, digest)
        elif extension == ".csv":
            frame = pd.read_csv(
                io.BytesIO(payload),
                sep=None,
                engine="python",
                encoding="utf-8-sig",
                nrows=MAX_SAMPLES + 1,
            )
            time_column = "time_s" if "time_s" in frame else "Time"
            if time_column not in frame or eeg.CHANNEL not in frame:
                raise InputError(
                    "Le CSV doit contenir les colonnes Time (ou time_s, en secondes) et EEGF7."
                )
            times = pd.to_numeric(frame[time_column], errors="raise").to_numpy(dtype=float)
            signal = pd.to_numeric(frame[eeg.CHANNEL], errors="raise").to_numpy(dtype=float)
            source = "Colonnes numériques CSV ; temps fourni en secondes"
        else:
            raise InputError("Format non pris en charge. Choisissez un fichier .mat, .npz ou .csv.")
    except InputError:
        raise
    except Exception as exc:
        raise InputError(
            "Lecture impossible : fichier incomplet ou structure incompatible. Consultez les formats acceptés."
        ) from exc
    times, signal = np.asarray(times, dtype=float), np.asarray(signal, dtype=float)
    if (
        times.ndim != 1
        or signal.ndim != 1
        or len(times) != len(signal)
        or not 2 <= len(signal) <= MAX_SAMPLES
    ):
        raise InputError(
            "Le temps et EEGF7 doivent avoir la même longueur, entre 2 et 2 millions d'échantillons."
        )
    if not np.isfinite(times).all() or not np.isfinite(signal).all():
        raise InputError("Le signal ou les temps contiennent des valeurs manquantes ou infinies.")
    delta = np.diff(times)
    step = float(np.median(delta))
    if step <= 0 or not (delta > 0).all() or not np.allclose(delta, step, rtol=1e-4, atol=2e-6):
        raise InputError(
            "Les temps doivent être réguliers et strictement croissants, sans doublon ni trou."
        )
    fs = 1 / step
    if not np.isclose(fs, expected_fs, rtol=1e-4):
        raise InputError(
            f"Fréquence mesurée : {fs:.3f} Hz. Ce modèle exige {expected_fs:g} Hz pour retrouver les caractéristiques de l'apprentissage."
        )
    if len(signal) < round(2 * fs):
        raise InputError(
            "Il faut au moins deux secondes de signal pour former une fenêtre complète."
        )
    if np.ptp(signal) == 0:
        raise InputError(
            "Le canal EEGF7 est entièrement constant ; aucune analyse exploitable n'est possible."
        )
    return Recording(times - times[0], signal, filename, source, fs, digest)


def load_detector():
    """Charge uniquement le modèle local exporté et vérifie sa provenance."""
    directory = ROOT / "app/model"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / "detector.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise RuntimeError("Le modèle a changé. Relancez app/prepare_model.py.")
    if (
        hashlib.sha256((ROOT / "src/eeg_project.py").read_bytes()).hexdigest()
        != metadata["feature_code_sha256"]
    ):
        raise RuntimeError(
            "Le calcul des caractéristiques a changé depuis l'export du modèle. Relancez app/prepare_model.py."
        )
    pipeline = joblib.load(model_path)
    if list(pipeline.feature_names_in_) != metadata["features"]:
        raise RuntimeError("Ordre des caractéristiques incompatible avec le modèle.")
    return pipeline, metadata


def load_catalogue():
    """Les 22 exports, contrôlés contre leur .mat d'origine, avec leur statut (app/prepare_model.py)."""
    path = ROOT / "app/model/recordings.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    return {row["id"]: row for row in rows}


def load_reference():
    """Crises annotées de la cohorte retenue, pour comparer un enregistrement connu."""
    path = ROOT / "app/model/reference_events.csv"
    return (
        pd.read_csv(path)
        if path.exists()
        else pd.DataFrame(columns=["record_id", "start_s", "end_s"])
    )


def trace_points(recording, start=0, end=None, budget=5000):
    """Signal filtré affiché ; au-delà du budget, garde le min et le max de chaque bloc."""
    values = recording.filtered if recording.filtered is not None else recording.signal
    end = float(recording.time[-1]) if end is None else end
    left, right = np.searchsorted(recording.time, [start, end], side="left")
    right = min(len(recording.time), max(left + 1, right + 1))
    if right - left <= budget:
        indices = np.arange(left, right)
    else:
        edges = np.linspace(left, right, budget // 2 + 1, dtype=int)
        selected = [left, right - 1]
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            block = values[a:b]
            first, second = sorted((a + int(np.argmin(block)), a + int(np.argmax(block))))
            selected.extend([first, second])
        indices = np.unique(selected)
    return {
        "time": recording.time[indices].round(4).tolist(),
        "signal": values[indices].round(3).tolist(),
    }


def envelope(time_values, values, bins=1600):
    """Minimum et maximum par colonne pour la vue d'ensemble."""
    edges = np.linspace(0, len(values), min(bins, len(values)) + 1, dtype=int)
    lows = np.minimum.reduceat(values, edges[:-1])
    highs = np.maximum.reduceat(values, edges[:-1])
    return {
        "time": time_values[edges[:-1]].round(2).tolist(),
        "min": lows.round(2).tolist(),
        "max": highs.round(2).tolist(),
    }


def record_identity(filename):
    """Identifiant de cohorte déduit du nom de fichier (190304A_E_0000d.mat → 190304A-E)."""
    stem = Path(filename).stem.removesuffix("_0000d")
    candidate = stem.replace("_", "-")
    return (
        candidate
        if len(candidate) == 9 and candidate[6].isalpha() and candidate[7] == "-"
        else None
    )


def analyze(recording, pipeline, metadata, reference, catalogue):
    """Réutilise exactement les fonctions du notebook ; aucun apprentissage au dépôt d'un fichier."""
    frame, windows, _ = eeg.recording_features(recording.time, recording.signal, recording.fs)
    features = frame[metadata["features"]]
    if not np.isfinite(features.to_numpy()).all():
        raise InputError("L'échelle des amplitudes produit des caractéristiques non finies.")
    recording.filtered = eeg.highpass(recording.signal, recording.fs)
    scores = eeg.model_scores(pipeline, features)
    params = metadata["params"]
    labels = eeg.postprocess(
        scores, params["threshold"], params["smoothing"], params["min_windows"]
    )
    starts, ends = frame.start_s.to_numpy(), frame.end_s.to_numpy()
    intervals = eeg.intervals_from_labels(starts, ends, labels)
    smoothed = eeg.smooth_scores(scores, params["smoothing"])

    record_id = record_identity(recording.filename)
    truth_frame = reference[reference.record_id == record_id] if record_id else reference.iloc[:0]
    truth = list(truth_frame[["start_s", "end_s"]].itertuples(index=False, name=None))
    matches = eeg.match_intervals(truth, intervals)[0] if truth else []
    by_prediction = {j: (i, iou) for i, j, iou in matches}

    events = []
    for k, (start, end) in enumerate(intervals):
        inside = (starts >= start - 1e-6) & (ends <= end + 1e-6)
        match = by_prediction.get(k)
        events.append(
            {
                "id": k + 1,
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "mean_score": float(smoothed[inside].mean()),
                "peak_score": float(smoothed[inside].max()),
                "match": {"reference": match[0] + 1, "iou": match[1]} if match else None,
            }
        )
    reference_block = None
    if truth:
        matched = {i for i, _, _ in matches}
        reference_block = {
            "events": [
                {"id": i + 1, "start_s": a, "end_s": b, "matched": i in matched}
                for i, (a, b) in enumerate(truth)
            ],
            "tp": len(matches),
            "fp": len(intervals) - len(matches),
            "fn": len(truth) - len(matches),
        }
    entry = catalogue.get(record_id) if record_id else None
    membership = entry["status"] if entry else "inconnu"
    duration = len(recording.signal) / recording.fs
    covered = float(ends[-1])
    detected = sum(e["duration_s"] for e in events)
    low, high = np.quantile(smoothed, [0.01, 0.999])
    low, high = min(low, params["threshold"]), max(high, params["threshold"])
    pad = 0.08 * (high - low or 1)
    amplitude = np.subtract(*np.percentile(recording.filtered, [75, 25])) / 1.349
    return {
        "filename": recording.filename,
        "record_id": record_id if entry or truth else None,
        "membership": membership,
        "status_reason": entry["reason"] if entry else None,
        "created_at": datetime.now(UTC).isoformat(),
        "source": recording.source,
        "input_sha256": recording.sha256,
        "channel": eeg.CHANNEL,
        "fs_hz": recording.fs,
        "duration_s": duration,
        "analyzed_duration_s": covered,
        "n_samples": len(recording.signal),
        "n_windows": len(windows),
        "positive_windows": int(labels.sum()),
        "detected_duration_s": detected,
        "events": events,
        "reference": reference_block,
        "configuration": metadata["configuration"],
        "params": params,
        "amplitude_scale": float(amplitude if amplitude > 0 else np.std(recording.filtered) or 1.0),
        "overview": envelope(recording.time, recording.filtered),
        "scores": {
            "time": ((starts + ends) / 2).round(2).tolist(),
            "value": smoothed.round(4).tolist(),
            "threshold": params["threshold"],
            "range": [float(low - pad), float(high + pad)],
        },
    }
