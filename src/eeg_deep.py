"""Réseau convolutif 1D sur les 19 voies EEG brutes, sans caractéristiques calculées à la main.

Mêmes fenêtres (2 s, pas de 1 s) et mêmes étiquettes que le pipeline classique ; mêmes
enregistrements d'apprentissage et de test. Le signal est filtré (passe-haut 0,5 Hz),
rééchantillonné à 128 Hz après un filtre anti-repliement, puis normalisé voie par voie par
la médiane et l'IQR de l'enregistrement entier : aucune étiquette n'est utilisée.
"""

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score
from torch import nn

from src import eeg_project as eeg

CHANNELS = [
    "EEGFp1", "EEGFp2", "EEGF7", "EEGF3", "EEGFz", "EEGF4", "EEGF8", "EEGT3", "EEGC3", "EEGCz",
    "EEGC4", "EEGT4", "EEGT5", "EEGP3", "EEGPz", "EEGP4", "EEGT6", "EEGO1", "EEGO2",
]  # fmt: skip
DEEP_FS = 128
WINDOW = 2 * DEEP_FS
JITTER = DEEP_FS // 8  # décalage aléatoire de ±0,125 s pendant l'apprentissage
CLIP = 20.0


class CNN1D(nn.Module):
    """Trois blocs convolution-normalisation-ReLU, moyenne globale, dropout, sortie logit."""

    def __init__(self, n_channels=len(CHANNELS), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
        )  # fmt: skip
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def select_channels(signals, names):
    """Les 19 voies dans l'ordre fixe du modèle ; erreur explicite si l'une manque."""
    missing = [c for c in CHANNELS if c not in names]
    if missing:
        raise ValueError(f"Voies absentes : {', '.join(c.removeprefix('EEG') for c in missing)}.")
    return np.stack([signals[names.index(c)] for c in CHANNELS]).astype(float)


def prepare(signals, fs):
    """Passe-haut 0,5 Hz, passage à 128 Hz, normalisation robuste par voie, bornée à ±20."""
    factor = fs / DEEP_FS
    if not np.isclose(factor, round(factor)) or round(factor) < 1:
        raise ValueError(f"Fréquence {fs:g} Hz incompatible avec un passage à {DEEP_FS} Hz.")
    x = resample_poly(eeg.highpass(signals, fs), 1, round(factor), axis=1)
    median = np.median(x, axis=1, keepdims=True)
    q1, q3 = np.percentile(x, [25, 75], axis=1, keepdims=True)
    iqr = np.where(q3 - q1 > 1e-12, q3 - q1, 1.0)
    return np.clip((x - median) / iqr, -CLIP, CLIP).astype(np.float32)


def window_starts(n_samples, fs):
    """Débuts des fenêtres en échantillons à `fs`, identiques à `segment_signal`."""
    size, hop = round(2 * fs), round(fs)
    return np.arange(0, n_samples - size + 1, hop)


def crop(records, items, jitter=0, rng=None):
    """Extrait un lot de fenêtres (record, début à 128 Hz), avec décalage aléatoire éventuel."""
    batch = np.empty((len(items), len(CHANNELS), WINDOW), dtype=np.float32)
    for k, (r, start) in enumerate(items):
        x = records[r]
        if jitter:
            start = int(np.clip(start + rng.integers(-jitter, jitter + 1), 0, x.shape[1] - WINDOW))
        batch[k] = x[:, start : start + WINDOW]
    return batch


def augment(batch, rng):
    """Amplitude ±10 % par fenêtre et bruit gaussien léger."""
    scale = rng.uniform(0.9, 1.1, size=(len(batch), 1, 1)).astype(np.float32)
    return batch * scale + rng.normal(0, 0.05, size=batch.shape).astype(np.float32)


def predict(model, records, items, batch_size=512):
    """Probabilités de crise, réseau en mode évaluation."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            x = torch.from_numpy(crop(records, items[i : i + batch_size]))
            out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out) if out else np.empty(0)


def predict_mc(model, records, items, passes=30, batch_size=512, seed=eeg.SEED):
    """Monte-Carlo dropout : dropout actif, normalisation figée ; moyenne et écart-type."""
    torch.manual_seed(seed)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    draws = np.empty((passes, len(items)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            # Les convolutions ne dépendent pas du dropout : calculées une seule fois par lot.
            features = model.features(torch.from_numpy(crop(records, items[i : i + batch_size])))
            for p in range(passes):
                draws[p, i : i + len(features)] = torch.sigmoid(
                    model.head(features).squeeze(-1)
                ).numpy()
    model.eval()
    return draws.mean(axis=0), draws.std(axis=0)


def saliency(model, window):
    """Gradient × entrée du logit, en valeur absolue, pour une fenêtre (voies × échantillons)."""
    model.eval()
    x = torch.from_numpy(window[None].astype(np.float32)).requires_grad_(True)
    model(x).sum().backward()
    return (x.grad[0] * x[0]).abs().detach().numpy(), float(torch.sigmoid(model(x)).item())


def train(records, items, labels, epochs, seed=eeg.SEED, val=None, patience=4, log=print):
    """Apprentissage pondéré (pos_weight) avec augmentation ; arrêt précoce si `val` est fourni.

    `val` = (items, labels) : l'époque retenue maximise l'AP sur ces fenêtres.
    Renvoie le modèle, l'époque retenue et l'historique.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = CNN1D()
    labels = np.asarray(labels, dtype=np.float32)
    pos_weight = torch.tensor((labels == 0).sum() / max(1, (labels == 1).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_state, best_epoch, history = -1.0, None, epochs, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(items))
        total = 0.0
        for i in range(0, len(order), 128):
            index = order[i : i + 128]
            x = torch.from_numpy(augment(crop(records, items[index], JITTER, rng), rng))
            y = torch.from_numpy(labels[index])
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        row = {"epoch": epoch, "loss": total / len(items)}
        if val is not None:
            row["val_ap"] = float(average_precision_score(val[1], predict(model, records, val[0])))
            if row["val_ap"] > best:
                best, best_epoch = row["val_ap"], epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append(row)
        log(
            "  époque {epoch:2d}  perte {loss:.4f}".format(**row)
            + (f"  AP val {row['val_ap']:.3f}" if val else "")
        )
        if val is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history
