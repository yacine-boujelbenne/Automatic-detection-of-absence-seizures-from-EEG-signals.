"""Contrôle des exports par lecture des cellules numériques MATLAB originales."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import mcos_extract as mat


def file_hash(path):
    """Calcule une empreinte pour invalider les contrôles après un changement."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_header(buf, start, end):
    """Lit les attributs d'une matrice à partir de ses balises, sans balayage flottant."""
    cursor = mat.Stream(buf, end)
    cursor.pos = start
    flag_type, flags, _ = cursor.read_element()
    if flag_type != mat.MI_UINT32 or len(flags) < 8:
        raise ValueError("Drapeaux MATLAB invalides.")
    cls = int.from_bytes(flags[:4], "little") & 255
    _, dims = cursor.read_numeric_array()
    shape = tuple(map(int, dims))
    cursor.read_name()
    return cls, shape, cursor


def cell_children(buf, cursor, end):
    """Lit les enfants directs d'une cellule en respectant les tailles déclarées."""
    children = []
    while cursor.pos < end:
        dtype, _, payload, following = cursor.peek_element()
        if dtype != mat.MI_MATRIX or following > end:
            raise ValueError("Cellule MATLAB incomplète.")
        cls, shape, child_cursor = matrix_header(buf, payload, following)
        children.append((cls, shape, child_cursor, following))
        cursor.pos = following
    return children


def validate_record(root, record_id):
    """Compare chaque bloc et nom de canal à son export, puis mesure le pas temporel."""
    stem = record_id.replace("-", "_")
    source_path = root / "data/CRISES ABSENCES" / f"{stem}_0000d.mat"
    export_path = root / "data/exports" / f"{stem}.npz"
    with np.load(export_path, allow_pickle=False) as archive:
        signals = archive["signals"]
        names = archive["channel_names"].tolist()
        exported_times = archive["row_times_ms"]
        stored_fs = float(archive["fs"])
    if signals.ndim != 2 or signals.shape[0] != len(names):
        raise ValueError(f"{record_id} : dimensions de l'export invalides.")
    if not np.isfinite(signals).all():
        raise ValueError(f"{record_id} : échantillons non finis.")
    buf = mat.load_workspace(source_path)
    wrappers = [
        node.fields["FileWrapper__"]
        for node in mat.walk_stream(buf)
        if "FileWrapper__" in node.fields
    ]
    if len(wrappers) != 1:
        raise ValueError("Le conteneur temporel original n'est pas identifié.")
    wrapper = wrappers[0]
    row_times = wrapper.fields["{2}"].data
    if not np.array_equal(row_times, exported_times):
        raise ValueError("Le vecteur temporel exporté diffère du vecteur source.")
    # Le conteneur comprend la propriété duration « millis », et non des secondes.
    if b"millis\x00fmt" not in buf:
        raise ValueError("La propriété MATLAB de durée en millisecondes est absente.")
    differences = np.diff(row_times) / 1000
    if not (differences > 0).all() or not np.allclose(differences, np.median(differences)):
        raise ValueError("Temps source non monotone ou irrégulier.")
    channel_cells = []
    name_candidates = []
    for _, _, dtype, nbytes, payload, _ in mat._iter_elements(buf):
        if dtype != mat.MI_MATRIX or nbytes == 0:
            continue
        end = payload + nbytes
        try:
            cls, shape, cursor = matrix_header(buf, payload, end)
        except (ValueError, mat.ParseError):
            continue
        if cls != mat.MX_CELL:
            continue
        if shape == (len(row_times), 1):
            channel_cells.append(cell_children(buf, cursor, end))
        elif shape == (1, len(names)):
            children = cell_children(buf, cursor, end)
            if all(child[0] == mat.MX_CHAR for child in children):
                candidate = []
                for _, _, char_cursor, _ in children:
                    dtype, raw, _ = char_cursor.read_element()
                    encoding = "utf-16-le" if dtype == mat.MI_UTF16 else "utf-8"
                    candidate.append(raw.decode(encoding).rstrip("\x00"))
                name_candidates.append(candidate)
    if len(channel_cells) != len(names) or names not in name_candidates:
        raise ValueError("Les cellules sources ou les noms de canaux ne concordent pas.")
    block_size = None
    for channel, blocks in enumerate(channel_cells):
        if len(blocks) != len(row_times):
            raise ValueError("Nombre de lignes temporelles et de blocs incohérent.")
        position = 0
        for cls, shape, cursor, _ in blocks:
            if cls != mat.MX_DOUBLE:
                raise ValueError("Un bloc d'échantillons n'est pas une matrice double.")
            dtype, values = cursor.read_numeric_array()
            if dtype != mat.MI_DOUBLE or np.prod(shape) != len(values):
                raise ValueError("Taille du bloc numérique invalide.")
            if block_size is None:
                block_size = len(values)
            if len(values) != block_size:
                raise ValueError("Les blocs n'ont pas une longueur constante.")
            reference = values.astype(signals.dtype)
            if not np.array_equal(reference, signals[channel, position : position + block_size]):
                raise ValueError(f"{record_id} : export différent du canal {names[channel]}.")
            position += block_size
        if position != signals.shape[1]:
            raise ValueError("Des échantillons manquent dans l'export.")
    fs = block_size / float(np.median(differences))
    if not np.isclose(fs, stored_fs):
        raise ValueError("Fréquence stockée différente de celle mesurée sur les blocs source.")
    return {
        "record_id": record_id,
        "n_channels": len(names),
        "n_samples": signals.shape[1],
        "n_time_rows": len(row_times),
        "samples_per_time_row": block_size,
        "time_step_s": float(np.median(differences)),
        "fs_hz": fs,
        "duration_s": signals.shape[1] / fs,
        "source_values_match": True,
        "source_names_match": True,
        "source_times_match": True,
        "source_sha256": file_hash(source_path),
        "export_sha256": file_hash(export_path),
    }


def validate_exports(root, record_ids, output_dir):
    """Réutilise les vérifications uniquement si les fichiers ont la même empreinte."""
    manifest_path = output_dir / "source_validation.json"
    cache = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    cached = {row["record_id"]: row for row in cache}
    rows = []
    for record_id in sorted(record_ids):
        stem = record_id.replace("-", "_")
        source_path = root / "data/CRISES ABSENCES" / f"{stem}_0000d.mat"
        export_path = root / "data/exports" / f"{stem}.npz"
        old = cached.get(record_id)
        if (
            old
            and old["source_sha256"] == file_hash(source_path)
            and old["export_sha256"] == file_hash(export_path)
        ):
            rows.append(old)
        else:
            rows.append(validate_record(root, record_id))
        print(f"Source vérifiée : {record_id}", flush=True)
        manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "source_validation.csv", index=False)
    return frame
