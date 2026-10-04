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
    return ((rows - rows[0])[:, None] / 1000 + np.arange(block)[None, :] * np.median(delta) / block).ravel()


def read_mat(payload):
    """Lit les cellules numériques MCOS par leurs balises, sans balayage flottant."""
    with TemporaryDirectory(prefix="eeg-upload-") as directory:
        path = Path(directory) / "recording.mat"
        path.write_bytes(payload)
        buf = mat.load_workspace(path)
    wrappers = [node.fields["FileWrapper__"] for node in mat.walk_stream(buf) if "FileWrapper__" in node.fields]
    if len(wrappers) != 1 or "{2}" not in wrappers[0].fields or b"millis\x00fmt" not in buf:
        raise InputError("Ce fichier MAT n'a pas la structure temporelle du projet. Utilisez le fichier de signal *_d.mat ou un CSV Time,EEGF7.")
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
                    names.append(raw.decode("utf-16-le" if text_type == mat.MI_UTF16 else "utf-8").rstrip("\x00"))
                name_cells.append(names)
    candidates = {tuple(names) for names in name_cells if len(names) == len(channels) and names.count(eeg.CHANNEL) == 1}
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
    return time_from_rows(rows, len(signal)), signal, "Cellules numériques du fichier MATLAB original"


def read_npz(payload, digest):
    """Accepte les exports source vérifiés ou un axe explicite par échantillon."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 32 or sum(entry.file_size for entry in archive.infolist()) > MAX_UNPACKED:
            raise InputError("L'archive NPZ décompressée est trop volumineuse.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        if not {"signals", "channel_names"}.issubset(bundle.files):
            raise InputError("Le NPZ doit contenir signals, channel_names et un axe temporel.")
        names = bundle["channel_names"].tolist()
        signals = bundle["signals"]
        if not isinstance(names, list) or names.count(eeg.CHANNEL) != 1 or signals.ndim != 2 or signals.shape[0] != len(names):
            raise InputError("Le NPZ doit contenir exactement un canal EEGF7, avec signals organisé en canaux × échantillons.")
        if signals.shape[1] > MAX_SAMPLES or not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
            raise InputError("Le signal NPZ doit être réel et contenir au plus 2 millions d'échantillons.")
        signal = signals[names.index(eeg.CHANNEL)].astype(float)
        if "time_s" in bundle.files:
            times = np.asarray(bundle["time_s"], dtype=float)
            source = "Axe time_s fourni, un temps en secondes par échantillon"
        elif "row_times_ms" in bundle.files:
            checks_path = ROOT / "artifacts/final/source_validation.json"
            checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else []
            matches = [row for row in checks if row["export_sha256"] == digest]
            if len(matches) != 1:
                raise InputError("Cet export par blocs n'a pas été vérifié contre son fichier source. Déposez le MAT original, ou un NPZ avec time_s explicite.")
            times = time_from_rows(bundle["row_times_ms"], len(signal))
            source = "Export du projet : empreinte identique à l'export vérifié contre le MAT source"
        else:
            raise InputError("Axe temporel absent : une fréquence déclarée seule ne suffit pas. Fournissez time_s.")
        if "fs" in bundle.files:
            stored = np.asarray(bundle["fs"])
            if stored.size != 1 or len(times) < 2 or not np.isclose(float(stored.item()), 1 / np.median(np.diff(times)), rtol=1e-4):
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
            frame = pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding="utf-8-sig", nrows=MAX_SAMPLES + 1)
            time_column = "time_s" if "time_s" in frame else "Time"
            if time_column not in frame or eeg.CHANNEL not in frame:
                raise InputError("Le CSV doit contenir les colonnes Time (ou time_s, en secondes) et EEGF7.")
            times = pd.to_numeric(frame[time_column], errors="raise").to_numpy(dtype=float)
            signal = pd.to_numeric(frame[eeg.CHANNEL], errors="raise").to_numpy(dtype=float)
            source = "Colonnes numériques CSV ; temps fourni en secondes"
        else:
            raise InputError("Format non pris en charge. Choisissez un fichier .mat, .npz ou .csv.")
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Lecture impossible : fichier incomplet ou structure incompatible. Consultez les formats acceptés.") from exc
    times, signal = np.asarray(times, dtype=float), np.asarray(signal, dtype=float)
    if times.ndim != 1 or signal.ndim != 1 or len(times) != len(signal) or not 2 <= len(signal) <= MAX_SAMPLES:
        raise InputError("Le temps et EEGF7 doivent avoir la même longueur, entre 2 et 2 millions d'échantillons.")
    if not np.isfinite(times).all() or not np.isfinite(signal).all():
        raise InputError("Le signal ou les temps contiennent des valeurs manquantes ou infinies.")
    delta = np.diff(times)
    step = float(np.median(delta))
    if step <= 0 or not (delta > 0).all() or not np.allclose(delta, step, rtol=1e-4, atol=1e-8):
        raise InputError("Les temps doivent être réguliers et strictement croissants, sans doublon ni trou.")
    fs = 1 / step
    if not np.isclose(fs, expected_fs, rtol=1e-4):
        raise InputError(f"Fréquence mesurée : {fs:.3f} Hz. Ce modèle exige {expected_fs:g} Hz pour retrouver les caractéristiques de l'apprentissage.")
    if len(signal) < round(2 * fs):
        raise InputError("Il faut au moins deux secondes de signal pour former une fenêtre complète.")
    if np.ptp(signal) == 0:
        raise InputError("Le canal EEGF7 est entièrement constant ; aucune analyse exploitable n'est possible.")
    return Recording(times - times[0], signal, filename, source, fs, digest)


def load_detector():
    """Charge uniquement le modèle local exporté et vérifie sa provenance."""
    directory = ROOT / "app/model"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / "detector.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise RuntimeError("Le modèle a changé. Relancez app/prepare_model.py.")
    if hashlib.sha256((ROOT / "src/eeg_project.py").read_bytes()).hexdigest() != metadata["feature_code_sha256"]:
        raise RuntimeError("Le calcul des caractéristiques a changé depuis l'export du modèle.")
    pipeline = joblib.load(model_path)
    if list(pipeline.feature_names_in_) != eeg.FEATURES:
        raise RuntimeError("Ordre des caractéristiques incompatible avec le modèle.")
    return pipeline, metadata


def trace_points(recording, start=0, end=None, budget=5000):
    """Réduit seulement le tracé affiché ; conserve les extrema de chaque petit bloc."""
    end = float(recording.time[-1]) if end is None else end
    left, right = np.searchsorted(recording.time, [start, end], side="left")
    right = min(len(recording.time), max(left + 1, right + 1))
    if right - left <= budget:
        indices = np.arange(left, right)
    else:
        edges = np.linspace(left, right, budget // 2 + 1, dtype=int)
        selected = [left, right - 1]
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            values = recording.signal[a:b]
            selected.extend([a + int(np.argmin(values)), a + int(np.argmax(values))])
        indices = np.unique(selected)
    return {"time": recording.time[indices].round(6).tolist(), "signal": recording.signal[indices].round(6).tolist()}


def analyze(recording, pipeline, metadata):
    """Réutilise les fonctions du notebook ; aucun apprentissage au dépôt d'un fichier."""
    windows, starts, _, _, _ = eeg.segment_signal(recording.time, recording.signal, [])
    predictions, probabilities = [], []
    for offset in range(0, len(windows), 2048):
        features = eeg.extract_features(windows[offset : offset + 2048], recording.fs)
        if not np.isfinite(features.to_numpy()).all():
            raise InputError("L'échelle des amplitudes produit des caractéristiques non finies.")
        predictions.extend(pipeline.predict(features).astype(int).tolist())
        probabilities.extend(pipeline.predict_proba(features)[:, list(pipeline.classes_).index(1)].tolist())
    frame = pd.DataFrame({"record_id": recording.filename, "start_s": recording.time[starts], "end_s": recording.time[starts] + windows.shape[1] / recording.fs})
    intervals = eeg.reconstruct_intervals(frame, predictions)
    labels, scores = np.asarray(predictions), np.asarray(probabilities)
    events = []
    for i, (start, end) in enumerate(intervals, 1):
        selected = (labels == 1) & (frame.start_s.to_numpy() >= start - 1e-6) & (frame.end_s.to_numpy() <= end + 1e-6)
        events.append({"id": i, "start_s": start, "end_s": end, "duration_s": end - start,
                       "positive_windows": int(selected.sum()), "mean_score": float(scores[selected].mean()), "peak_score": float(scores[selected].max())})
    duration = len(recording.signal) / recording.fs
    covered = float(frame.end_s.iloc[-1])
    detected = sum(row["duration_s"] for row in events)
    known_id = recording.filename.removesuffix(Path(recording.filename).suffix).removesuffix("_0000d").replace("_", "-")
    membership = "apprentissage" if known_id in metadata["train_records"] else "test" if known_id in metadata["test_records"] else "inconnu"
    # Le nom du fichier ne prouve pas son identité : l'information est indicative.
    return {
        "filename": recording.filename, "created_at": datetime.now(UTC).isoformat(),
        "source": recording.source, "input_sha256": recording.sha256, "channel": eeg.CHANNEL,
        "fs_hz": recording.fs, "duration_s": duration, "analyzed_duration_s": covered,
        "uncovered_tail_s": max(0, duration - covered), "n_samples": len(recording.signal),
        "n_windows": len(windows), "positive_windows": int(labels.sum()),
        "detected_duration_s": detected, "detected_fraction": detected / covered,
        "longest_event_s": max((row["duration_s"] for row in events), default=0),
        "events": events, "overview": trace_points(recording), "model": metadata,
        "record_name_membership": membership,
        "interpretation": "Épisodes signalés par le modèle, à vérifier sur le tracé. Les scores ne sont pas des probabilités cliniques calibrées. Sans annotations de référence, la précision et le rappel de ce fichier ne sont pas calculables.",
    }
