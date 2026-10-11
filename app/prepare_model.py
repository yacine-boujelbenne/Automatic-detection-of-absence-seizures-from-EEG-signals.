"""Exporte pour l'application le modèle choisi en validation interne, avec son post-traitement."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import eeg_project as eeg  # noqa: E402
from src.eeg_source_validation import file_hash, validate_record  # noqa: E402

STATUS_ORDER = {"test": 0, "apprentissage": 1, "sans annotation": 2, "exclu": 3}


def main():
    """Le choix n'utilise jamais le test : AP en validation interne, puis réglages hors pli."""
    source = ROOT / "artifacts/final"
    destination = ROOT / "app/model"
    destination.mkdir(parents=True, exist_ok=True)
    selected = json.loads((source / "selected_model.json").read_text(encoding="utf-8"))
    name = selected["configuration"]
    experiment = joblib.load(source / "trained_experiment.joblib")
    pipeline = experiment["models"][name]
    pipeline.named_steps["model"].set_params(
        **({"n_jobs": 1} if "n_jobs" in pipeline.named_steps["model"].get_params() else {})
    )
    model_path = destination / "detector.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    frame = pd.read_csv(source / "features.csv", usecols=["record_id"])
    events = pd.read_csv(source / "annotations_seconds.csv")
    events[["record_id", "start_s", "end_s"]].to_csv(
        destination / "reference_events.csv", index=False
    )
    comparison = pd.read_csv(source / "temporal_comparison.csv")
    test = comparison[
        (comparison.stage == "Test réservé") & (comparison.postprocessing == "Post-traitement")
    ].iloc[0]
    cv = experiment["cv_summary"].set_index("configuration").loc[name]
    loro = eeg.loro_summary(pd.read_csv(source / "loro_results.csv"))
    loro_row = loro[(loro.configuration == name) & (loro.postprocessing == "Post-traitement")].iloc[
        0
    ]
    test_records = sorted(frame.iloc[experiment["test_indices"]].record_id.unique().tolist())
    source_checks = json.loads((source / "source_validation.json").read_text(encoding="utf-8"))
    rates = {float(row["fs_hz"]) for row in source_checks}
    if len(rates) != 1:
        raise ValueError("La fréquence de référence n'est pas unique.")
    feature_set, strategy, model = eeg.parse_config(name)
    metadata = {
        "configuration": name,
        "label": f"{eeg.MODELS[model]} · {eeg.STRATEGIES[strategy].lower()}",
        "feature_set": feature_set,
        "features": eeg.FEATURE_SETS[feature_set],
        "params": selected["postprocessing"],
        "selection": selected["selection"],
        "channel": eeg.CHANNEL,
        "fs_hz": rates.pop(),
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "metrics": {
            "cv": {"average_precision": float(cv.average_precision_mean), "f1": float(cv.f1_mean)},
            "test": {
                key: float(test[key])
                for key in [
                    "precision",
                    "recall",
                    "f1",
                    "event_TP",
                    "event_FN",
                    "event_FP",
                    "false_alarms_per_hour",
                    "hours",
                ]
            },
            "loro": {
                "records": int(frame.record_id.nunique()),
                **{
                    key: float(loro_row[key])
                    for key in [
                        "event_recall",
                        "event_precision",
                        "false_alarms_per_hour",
                        "f1_median",
                        "f1_q1",
                        "f1_q3",
                    ]
                },
            },
        },
        "test_records": test_records,
        "train_records": sorted(
            frame.iloc[experiment["train_indices"]].record_id.unique().tolist()
        ),
        "reference_counts": {rid: int((events.record_id == rid).sum()) for rid in test_records},
        "feature_code_sha256": hashlib.sha256(
            (ROOT / "src/eeg_project.py").read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    for key in ("event_TP", "event_FN", "event_FP"):
        metadata["metrics"]["test"][key] = int(metadata["metrics"]["test"][key])
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    recordings = build_recordings(destination, metadata, events)
    print(f"Modèle exporté : {name}, post-traitement {selected['postprocessing']}")
    print(f"{len(recordings)} enregistrements disponibles par identifiant.")


def build_recordings(destination, metadata, events):
    """Liste de tous les exports, chacun contrôlé contre son .mat d'origine, avec son statut."""
    path = destination / "recordings.json"
    previous = (
        {row["id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
        if path.exists()
        else {}
    )
    audit = pd.read_csv(ROOT / "artifacts/final/cohort_audit.csv")
    rows = []
    for export in sorted((ROOT / "data/exports").glob("*.npz")):
        rid = export.stem.replace("_", "-")
        source = ROOT / "data/CRISES ABSENCES" / f"{export.stem}_0000d.mat"
        hashes = {"source_sha256": file_hash(source), "export_sha256": file_hash(export)}
        old = previous.get(rid)
        if old and all(old.get(key) == value for key, value in hashes.items()):
            check = {"duration_s": old["duration_s"]}
        else:
            check = validate_record(ROOT, rid)
            print(f"Source vérifiée : {rid}", flush=True)
        lines = audit[audit.record_id == rid]
        if rid in metadata["test_records"]:
            status, reason = "test", "Jamais vu pendant l'entraînement"
        elif rid in metadata["train_records"]:
            status, reason = "apprentissage", "A servi à l'entraînement"
        elif len(lines):
            status, reason = "exclu", lines.reason.iloc[0]
        else:
            status, reason = "sans annotation", "Absent du classeur d'annotations mis à jour"
        with np.load(export, allow_pickle=False) as bundle:
            channels = bundle["channel_names"].tolist()
        available = eeg.CHANNEL in channels
        if not available:
            reason = f"Pas de canal F7 : montage réduit à {sum(c.startswith('EEG') for c in channels)} électrodes"
        ages = sorted({int(a) for a in lines.age.dropna()})
        sexes = sorted(set(lines.sex.dropna()))
        rows.append(
            {
                "id": rid,
                "status": status,
                "available": available,
                "reason": reason,
                "age": " / ".join(map(str, ages)) or None,
                "sex": " / ".join(sexes) or None,
                "annotated_seizures": int((events.record_id == rid).sum())
                if status in ("test", "apprentissage")
                else None,
                "duration_s": float(check["duration_s"]),
                **hashes,
            }
        )
    rows.sort(key=lambda row: (STATUS_ORDER[row["status"]], row["id"]))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
