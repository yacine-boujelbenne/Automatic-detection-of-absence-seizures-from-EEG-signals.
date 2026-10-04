"""Entraîne et conserve la comparaison utilisée par le notebook final."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import eeg_project as eeg  # noqa: E402
from src.eeg_source_validation import validate_exports  # noqa: E402


def main():
    """Exécute la chaîne sur les données réelles avec leurs contrôles."""
    sys.stdout.reconfigure(encoding="utf-8")
    out = ROOT / "artifacts/final"
    out.mkdir(parents=True, exist_ok=True)
    cohort, events, audit, all_events, unannotated = eeg.load_annotations(ROOT)
    validation = validate_exports(ROOT, cohort.record_id, out)
    audit.to_csv(out / "cohort_audit.csv", index=False)
    events.to_csv(out / "annotations_seconds.csv", index=False)
    all_events.to_csv(out / "annotations_all_rows.csv", index=False)
    frame, counts = eeg.build_features(ROOT, cohort, events, validation)
    frame.to_csv(out / "features.csv", index=False)
    counts.to_csv(out / "class_counts.csv", index=False)
    train, test, folds = eeg.choose_split(frame)
    print("Fenêtres", len(frame), "positives", int(frame.Class.sum()), flush=True)
    print("Test", sorted(frame.iloc[test].record_id.unique()), len(test), flush=True)
    result = eeg.run_training(frame, train, test, folds, out)
    temporal = eeg.temporal_analysis(frame, events, result, out)
    print("Choix en validation interne :", result["best"])
    print(result["results"].to_string(index=False))
    print(temporal["events"].to_string(index=False))


if __name__ == "__main__":
    main()
