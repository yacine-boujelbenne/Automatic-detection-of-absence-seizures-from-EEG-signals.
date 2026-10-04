---
name: eeg-pipeline
description: Use when implementing the assignment's 12 parts for absence-seizure detection - windowing the EEG signal, the 8 temporal features, the Delta/Theta/Alpha/Beta band powers, the 50% seizure-overlap window label, stratified 80/20 split, the Decision Tree / Random Forest / KNN / SVM models, the metrics table, or reconstructing crisis intervals from window predictions. Contains a numerically verified reference implementation.
---

# EEG absence-seizure pipeline

The assignment's pipeline, with the exact conventions that the reference code
uses. `reference/eeg_features.py` is verified by `reference/test_reference.py`
(76 closed-form checks, all passing) — prefer it over re-deriving the maths.

```bash
cd .claude/skills/eeg-pipeline/reference && python test_reference.py
```

## Spec, part by part

| Part | Requirement | Convention used |
| --- | --- | --- |
| IV | 2 s windows, 50% overlap | `n = round(fs*2)` = 512 at 256 Hz; hop = `n*(1-overlap)` = 256 |
| IV | Window is class 1 if ≥50% of its samples are seizure | `wins.mean(axis=1) >= 0.5`, inclusive |
| II/VI | Interval membership is `(debut, fin]` | mask is `t > start & t <= end` |
| V | 8 temporal features | mean, std (ddof=0), variance, min, max, amplitude = max−min, RMS, energy = Σx² |
| VI | 4 band powers | one-sided FFT, see below |
| VIII | Stratified 80/20 | `train_test_split(..., stratify=y, random_state=42)` |
| IX | Accuracy, Precision, Recall, F1, confusion matrix | `zero_division=0` always |
| X | Reconstruct intervals from predictions | merge consecutive 1s, pad ±½ window |
| XI | TP/FP/FN error analysis | both window-level **and** event-level |

### Band powers

```python
spec  = np.fft.rfft(w)
power = np.abs(spec)**2 / n**2
power[1:-1] *= 2                      # one-sided; DC and Nyquist unmirrored
band_power = power[(freqs >= lo) & (freqs < hi)].sum()
```

- Bands are `[lo, hi)` — low edge **included**, high edge **excluded**, so
  4 Hz lands in theta not delta and 13 Hz lands in beta not alpha. No bin is
  counted twice.
- A real sinusoid of amplitude `A` inside a band contributes exactly `A²/2`,
  and the power spectrum over all bins sums to `mean(w²)` (Parseval). Both
  properties are asserted in the test file.
- The four bands cover 0.5–30 Hz. A 2 s window at 256 Hz resolves 0–128 Hz, so
  the bands capture only ~23% of total power. That is expected, not a bug.

### `energy` — the one genuinely ambiguous spec item

The assignment lists "énergie" without a formula. The reference uses
`Σx²` (signal-processing convention). If your course defines it as the mean
square, use `np.mean(w**2)`. **State which you used in the report** — it
changes the feature scale but not the ranking of models.

## The four models

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
}
```

**SVM and KNN need scaling; the trees do not.** Putting the scaler inside a
`Pipeline` is what guarantees it is fitted on the training split only — see
`eeg-validation`. `random_state=42` everywhere so the notebook is reproducible.

## Metrics table (Part IX)

```python
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

def score_all(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-score":  f1_score(y_true, y_pred, zero_division=0),
    }
```

`zero_division=0` is not optional. Without it, a model that never predicts a
crisis raises instead of scoring 0 — which is exactly the degenerate case
Part IX/Question 13 is about.

## Part X — from window labels back to time

`predictions_to_intervals()` merges runs of positive windows and extends each
by half a window at both ends, because a window label summarises a span centred
on that window. At 256 Hz with 2 s / 50% the hop is 1.0 s.

Then compare against the true annotations, and additionally report **event-level**
TP/FP/FN via `match_events()`. Window-level F1 flatters the model badly: one
real 8 s crisis becomes ~5 overlapping positive windows, so a single missed
crisis costs 5 false negatives. Event-level counting is what a clinician cares
about and is the strongest thing you can add to the report.

## Two questions the assignment does not ask but you should

1. **Per-record metrics.** With 21 usable records, a pooled score hides that one
   hard record may be failing entirely. Report mean ± std of per-record F1.
2. **The 2 s window is longer than most of these seizures.** The workbook's
   median annotated duration is `1.3e-4` in its own units — once converted, if
   seizures are genuinely shorter than 2 s, a single window can straddle seizure
   and non-seizure and the 50% rule becomes very sensitive. Sweep the window
   length (1 s, 2 s, 4 s) as the Part XII improvement and report the effect.

## Order of operations (do not reorder)

load → audit → segment → label → extract features → **split** → scale → train →
evaluate → reconstruct intervals → error analysis

The split must come after feature extraction but before any fitting, and the
scaler must live inside the pipeline. See `eeg-validation` for the leakage
rules and the group-split problem specific to this dataset.
