"""Exporte pour la démonstration le meilleur F1 observé sur le test du projet."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eeg_project import CHANNEL, FEATURES, MODELS, STRATEGIES  # noqa: E402


def main():
    """Conserve les poids entraînés et documente le choix effectué après test."""
    source = ROOT / "artifacts/final"
    destination = ROOT / "app/model"
    destination.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(source / "test_results.csv")
    best = results.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    experiment = joblib.load(source / "trained_experiment.joblib")
    pipeline = experiment["models"][best.configuration]
    pipeline.named_steps["model"].n_jobs = 1
    model_path = destination / "detector.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    frame = pd.read_csv(source / "features.csv", usecols=["record_id"])
    cv_choice = json.loads((source / "selected_model.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "training_provenance.json").read_text(encoding="utf-8"))
    source_checks = json.loads((source / "source_validation.json").read_text(encoding="utf-8"))
    rates = {float(row["fs_hz"]) for row in source_checks}
    if len(rates) != 1:
        raise ValueError("La fréquence de référence n'est pas unique.")
    metadata = {
        "configuration": best.configuration,
        "name": MODELS[best.model],
        "strategy": STRATEGIES[best.strategy],
        "selection": "Meilleur F1 observé sur le test réservé, choisi pour cette démonstration à la demande de l'utilisateur.",
        "validation_choice": cv_choice["configuration"],
        "selection_note": "Le notebook conserve son choix par validation interne. Les scores de test ci-dessous ont servi au choix de démonstration ; une nouvelle validation indépendante reste nécessaire.",
        "metrics": {name: float(best[name]) for name in ["precision", "recall", "f1", "accuracy"]},
        "confusion": {name: int(best[name]) for name in ["TP", "FP", "TN", "FN"]},
        "channel": CHANNEL,
        "fs_hz": rates.pop(),
        "features": FEATURES,
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "amplitude_unit": "unité du fichier source",
        "test_records": sorted(frame.iloc[experiment["test_indices"]].record_id.unique().tolist()),
        "train_records": sorted(frame.iloc[experiment["train_indices"]].record_id.unique().tolist()),
        "test_windows": int(sum(int(best[name]) for name in ["TP", "FP", "TN", "FN"])),
        "versions": provenance["versions"],
        "training_fingerprint": provenance["fingerprint"],
        "feature_code_sha256": hashlib.sha256((ROOT / "src/eeg_project.py").read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Modèle exporté : {best.configuration}, F1 de test = {best.f1:.3f}")


if __name__ == "__main__":
    main()
