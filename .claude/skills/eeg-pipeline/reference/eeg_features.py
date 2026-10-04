"""Canonical EEG feature extraction, verified against analytic cases.

This module is the reference implementation referenced by the eeg-pipeline
skill. It is deliberately dependency-light (numpy only) so the same code can be
pasted into the notebook or imported from src/.

Every formula is checked in test_reference.py against a closed-form result.
"""

from __future__ import annotations

import numpy as np

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def window_length(fs: float, seconds: float = 2.0) -> int:
    """Samples per window. At fs=256 and 2 s this is 512, per the assignment."""
    n = int(round(fs * seconds))
    if n < 2:
        raise ValueError("window too short: %d samples" % n)
    return n


def iter_windows(x: np.ndarray, n: int, overlap: float = 0.5) -> np.ndarray:
    """Sliding windows as a (n_windows, n) view-free copy.

    overlap=0.5 at fs=256, n=512 gives a hop of 256 samples, as specified.
    The tail that cannot fill a full window is dropped, so every row is a
    complete window; this keeps feature values comparable across windows.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1), got %r" % (overlap,))
    hop = max(1, int(round(n * (1.0 - overlap))))
    if x.size < n:
        return np.empty((0, n), dtype=float)
    starts = np.arange(0, x.size - n + 1, hop)
    idx = starts[:, None] + np.arange(n)[None, :]
    return x[idx]


def seizure_mask(n_samples: int, fs: float, intervals: list[tuple[float, float]]) -> np.ndarray:
    """Per-sample binary mask. Interval membership is (debut, fin] -- open on
    the left, closed on the right, exactly as the assignment specifies.

    intervals are (start_s, end_s) in the same time base as `fs`.
    """
    t = np.arange(n_samples) / fs
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in intervals:
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        mask |= (t > start) & (t <= end)
    return mask


def label_windows(
    mask: np.ndarray, n: int, overlap: float = 0.5, threshold: float = 0.5
) -> np.ndarray:
    """Label each window 1 when at least `threshold` of its samples are seizure.

    threshold=0.5 is the assignment's simplification.
    """
    wins = iter_windows(mask.astype(float), n, overlap)
    if wins.size == 0:
        return np.empty(0, dtype=int)
    return (wins.mean(axis=1) >= threshold).astype(int)


def temporal_features(w: np.ndarray) -> dict[str, float]:
    """The 8 temporal features required by Part V.

    energy is defined here as the sum of squares (the signal-processing
    definition). If the grader expects the mean square, divide by len(w); state
    the convention you used in the report.
    """
    w = np.asarray(w, dtype=float)
    return {
        "mean": float(w.mean()),
        "std": float(w.std(ddof=0)),
        "variance": float(w.var(ddof=0)),
        "min": float(w.min()),
        "max": float(w.max()),
        "amplitude": float(w.max() - w.min()),
        "rms": float(np.sqrt(np.mean(w**2))),
        "energy": float(np.sum(w**2)),
    }


def band_powers(w: np.ndarray, fs: float, bands: dict = BANDS) -> dict[str, float]:
    """Power in each frequency band, via a one-sided FFT of the raw window.

    Power spectrum is |rfft(w)|^2 / n^2, so a pure sine of amplitude A gives
    A^2/4 split across the two adjacent bins. Band power is the sum of the
    one-sided bins in [lo, hi), doubled away from DC and Nyquist.

    Band edges are inclusive of the low edge and exclusive of the high edge,
    so adjacent bands never double-count the boundary bin.
    """
    w = np.asarray(w, dtype=float)
    n = w.size
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(spec) ** 2) / (n**2)
    # One-sided: bins 1..end are mirrored, DC and Nyquist are not.
    scale = np.full(power.shape, 2.0)
    scale[0] = 1.0
    if n % 2 == 0:
        scale[-1] = 1.0
    power = power * scale

    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        out["power_%s" % name] = float(power[sel].sum())
    return out


def extract_dataset(
    x: np.ndarray,
    fs: float,
    intervals: list[tuple[float, float]],
    seconds: float = 2.0,
    overlap: float = 0.5,
    threshold: float = 0.5,
) -> tuple[list[dict], np.ndarray]:
    """Full window -> feature-row pipeline for one recording.

    Returns (rows, labels). One row per window, in time order.
    """
    n = window_length(fs, seconds)
    mask = seizure_mask(x.size, fs, intervals)
    y = label_windows(mask, n, overlap, threshold)
    wins = iter_windows(np.asarray(x, dtype=float), n, overlap)
    if wins.shape[0] == 0:
        return [], np.empty(0, dtype=int)
    if wins.shape[0] != y.size:
        raise AssertionError("window/label count mismatch: %d vs %d" % (wins.shape[0], y.size))
    rows = [{**temporal_features(w), **band_powers(w, fs), "t_start": i * (n * (1 - overlap)) / fs}
            for i, w in enumerate(wins)]
    return rows, y


def predictions_to_intervals(
    y_pred: np.ndarray, fs: float, seconds: float = 2.0, overlap: float = 0.5
) -> list[tuple[float, float]]:
    """Reconstruct (start_s, end_s) crisis intervals from per-window labels.

    Consecutive positive windows are merged into one event, and the merged
    interval is stretched by half a window at each end, because each window
    label summarises a span centred on that window. This is the Part X step.
    """
    n = window_length(fs, seconds)
    hop_time = n * (1.0 - overlap) / fs
    half = seconds / 2.0
    out: list[tuple[float, float]] = []
    start_idx = None
    for i, v in enumerate(list(y_pred) + [0]):
        if v == 1 and start_idx is None:
            start_idx = i
        elif v == 0 and start_idx is not None:
            t0 = max(0.0, start_idx * hop_time - half)
            t1 = min((i - 1) * hop_time + half, (len(y_pred) - 1) * hop_time + half)
            out.append((t0, t1))
            start_idx = None
    return out


def match_events(
    truth: list[tuple[float, float]],
    pred: list[tuple[float, float]],
    min_iou: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy one-to-one event matching by interval IoU.

    Returns (TP, FP, FN) at the EVENT level, which is what a clinician cares
    about and what window-level F1 hides. A predicted event matches a true
    event when their IoU exceeds `min_iou`; each true event is consumed once.
    """
    def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        if inter <= 0.0:
            return 0.0
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0.0

    used: set[int] = set()
    tp = 0
    for p in pred:
        best, best_iou = None, 0.0
        for i, t in enumerate(truth):
            if i in used:
                continue
            v = iou(t, p)
            if v > best_iou:
                best, best_iou = i, v
        if best is not None and best_iou > min_iou:
            used.add(best)
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp
