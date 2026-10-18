"""Entraîne le CNN 1D sur les 19 voies, avec la même séparation par enregistrement que le notebook.

1. Validation interne : 3 plis groupés sur les 12 tracés d'apprentissage. Dans chaque pli,
   un tracé de l'apprentissage sert à l'arrêt précoce ; le pli de validation n'est jamais
   vu avant la prédiction. Ses scores hors pli règlent le post-traitement.
2. Modèle final : les 12 tracés, nombre d'époques = médiane des époques retenues.
3. Test réservé : lu une seule fois, à la fin.

Sorties dans app/model/ : cnn.pt, cnn_metadata.json, cnn_test_scores.csv.
Lancement : .venv/bin/python scripts/train_cnn.py  (environ 15 minutes sur 8 cœurs).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import inference  # noqa: E402
from src import eeg_deep as deep  # noqa: E402
from src import eeg_project as eeg  # noqa: E402

MAX_EPOCHS = 25
NAME = "deep__original__cnn"  # seuil par défaut 0,5 (probabilité)
OUT = ROOT / "app/model"


def load_records(record_ids, catalogue):
    """Exports dont l'empreinte est celle contrôlée contre le MAT d'origine, préparés à 128 Hz."""
    prepared = {}
    for rid in record_ids:
        path = ROOT / "data/exports" / f"{rid.replace('-', '_')}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != catalogue[rid]["export_sha256"]:
            raise RuntimeError(f"{path.name} diffère de l'export vérifié.")
        with np.load(path, allow_pickle=False) as bundle:
            names = bundle["channel_names"].tolist()
            signals = deep.select_channels(bundle["signals"], names)
            times = inference.time_from_rows(bundle["row_times_ms"], signals.shape[1])
        fs = 1 / float(np.median(np.diff(times)))
        prepared[rid] = deep.prepare(signals, fs)
        print(f"Préparé : {rid} ({signals.shape[1] / fs / 60:.1f} min, {fs:g} Hz)", flush=True)
    return prepared


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    started = time.time()
    meta = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    catalogue = inference.load_catalogue()
    frame = pd.read_csv(
        ROOT / "artifacts/final/features.csv",
        usecols=["record_id", "start_sample", "start_s", "end_s", "Class"],
    )
    events = pd.read_csv(ROOT / "artifacts/final/annotations_seconds.csv")
    record_ids = sorted(frame.record_id.unique())
    records = load_records(record_ids, catalogue)
    index_of = {rid: k for k, rid in enumerate(record_ids)}
    order = [records[rid] for rid in record_ids]
    items = np.c_[frame.record_id.map(index_of).to_numpy(), frame.start_sample.to_numpy() // 2]
    y = frame.Class.to_numpy()

    train_mask = frame.record_id.isin(meta["train_records"]).to_numpy()
    test_mask = frame.record_id.isin(meta["test_records"]).to_numpy()
    assert not (train_mask & test_mask).any()
    train_frame = frame[train_mask].reset_index(drop=True)
    train_index = np.flatnonzero(train_mask)

    # 1. Validation interne groupée.
    oof = np.full(len(train_frame), np.nan)
    epochs, fold_rows = [], []
    rng = np.random.default_rng(eeg.SEED)
    for k, (fit, val) in enumerate(eeg.grouped_folds(train_frame), 1):
        fit_records = sorted(train_frame.record_id.iloc[fit].unique())
        positive = [r for r in fit_records if train_frame.Class[train_frame.record_id == r].sum()]
        stop_record = rng.choice(positive)
        stop = fit[train_frame.record_id.iloc[fit].to_numpy() == stop_record]
        learn = fit[train_frame.record_id.iloc[fit].to_numpy() != stop_record]
        print(
            f"Pli {k} : arrêt précoce sur {stop_record}, validation sur "
            f"{', '.join(sorted(train_frame.record_id.iloc[val].unique()))}",
            flush=True,
        )
        model, best_epoch, _ = deep.train(
            order, items[train_index[learn]], y[train_index[learn]], MAX_EPOCHS,
            val=(items[train_index[stop]], y[train_index[stop]]),
        )  # fmt: skip
        oof[val] = deep.predict(model, order, items[train_index[val]])
        epochs.append(best_epoch)
        fold_rows.append({
            "fold": k,
            "epoch": best_epoch,
            "average_precision": float(average_precision_score(y[train_index[val]], oof[val])),
        })  # fmt: skip
        print(f"  AP du pli : {fold_rows[-1]['average_precision']:.3f}", flush=True)
    assert np.isfinite(oof).all()
    recordings = eeg.Recordings(train_frame, events)
    params, _ = eeg.tune_postprocessing(recordings, oof, NAME)
    cv_metrics, _, _ = recordings.evaluate(oof, params)

    # 2. Modèle final sur les 12 tracés d'apprentissage.
    final_epochs = int(np.median(epochs))
    print(f"Modèle final : {final_epochs} époques ; post-traitement {params}", flush=True)
    model, _, history = deep.train(order, items[train_index], y[train_index], final_epochs)

    # 3. Test réservé, lu une seule fois.
    test_frame = frame[test_mask].reset_index(drop=True)
    scores = deep.predict(model, order, items[test_mask])
    test_recordings = eeg.Recordings(test_frame, events)
    raw, _, _ = test_recordings.evaluate(scores, eeg.default_params(NAME))
    post, _, _ = test_recordings.evaluate(scores, params)
    ap = float(average_precision_score(test_frame.Class, scores))

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "cnn.pt")
    test_frame[["record_id", "start_s", "end_s", "Class"]].assign(cnn=scores).to_csv(
        OUT / "cnn_test_scores.csv", index=False, float_format="%.6g"
    )

    def clean(metrics):
        return {
            k: (float(v) if isinstance(v, float | np.floating) else int(v))
            for k, v in metrics.items()
        }

    metadata = {
        "label": "CNN 1D · 19 voies brutes",
        "channels": deep.CHANNELS,
        "fs_hz": deep.DEEP_FS,
        "source_fs_hz": meta["fs_hz"],
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "params": params,
        "epochs": final_epochs,
        "cv": {
            "folds": fold_rows,
            "average_precision_mean": float(np.mean([r["average_precision"] for r in fold_rows])),
            "postprocessed": clean(cv_metrics),
        },
        "test": {"average_precision": ap, "raw": clean(raw), "postprocessed": clean(post)},
        "history": history,
        "train_records": meta["train_records"],
        "test_records": meta["test_records"],
        "model_sha256": hashlib.sha256((OUT / "cnn.pt").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256((ROOT / "src/eeg_deep.py").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "seed": eeg.SEED,
        "training_minutes": (time.time() - started) / 60,
    }
    (OUT / "cnn_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Test réservé : AP {ap:.3f} ; fenêtres F1 brut {raw['f1']:.3f}, post-traité {post['f1']:.3f} ; "
        f"crises {post['event_TP']}/{post['event_TP'] + post['event_FN']}, "
        f"{post['false_alarms_per_hour']:.1f} fausses alarmes/h ; {metadata['training_minutes']:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
