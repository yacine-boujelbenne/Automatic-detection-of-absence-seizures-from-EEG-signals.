---
name: eeg-validation
description: Use when splitting data, training or evaluating the seizure classifiers, choosing which metric to report, or reviewing results for leakage and class imbalance. Covers the 80/20 stratified split, why overlapping 2 s windows leak across it, why the scaler must sit inside a Pipeline, the 95/5 always-negative baseline from Question 13, and the 21-record sample-size problem.
---

# Validation and leakage guardrails

The single most likely way to lose marks on this project is to report a
flattering number that a sharp reader can dismantle. These are the specific
traps in this dataset.

## Trap 1 — overlapping windows leak across the train/test split

This is the big one, and it is specific to a windowed time series. With 2 s
windows at 50% overlap, **consecutive windows share 50% of their samples**. A
naive `train_test_split` over the window table puts window *i* in train and
window *i+1* in test. Those two windows are not independent observations — they
are nearly the same 2 s of EEG. The model can effectively memorise the test
signal, and every reported metric is inflated.

Worse, the split ignores the patient entirely: all windows from one recording
land on both sides.

**Fix — split by recording, then by time.** Hold out whole records:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["record_id"]))
```

Do the assignment's stratified 80/20 as asked, using
`train_test_split(..., stratify=y)` on **record ids** so both sides keep a
similar class proportion, then assign all windows of a record to the same side.

State clearly in the report which you used and why. Doing the record-level split
and reporting it as a deliberate methodological choice is worth marks; doing the
naive window split without noticing is the failure mode.

**Verify the fix, do not assume it:**

```python
assert not (set(train_records) & set(test_records)), "records leak across the split"
```

## Trap 2 — the scaler fitted on everything

`StandardScaler().fit_transform(X)` before the split leaks test-set statistics
into training. It also silently rescales the tree models' inputs, changing
their behaviour. Always:

```python
pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X_tr, y_tr)      # scaler fitted here, on training data only
```

## Trap 3 — Accuracy is meaningless here (Question 13)

The assignment's own example: if 95% of windows are non-crisis, a model that
**always** answers "no crisis" scores 95% accuracy and detects nothing.

For this dataset the imbalance is expected to be severe — 103 seizure events
across 21 records, and each event is a minority of the samples. So:

- **Report all four metrics plus the confusion matrix.** Never accuracy alone.
- **Print the majority-class baseline** next to your scores:

```python
baseline = y_te.value_counts(normalize=True).max()
print("always-negative baseline accuracy: %.3f" % baseline)
```

If your model's accuracy is not clearly above that baseline, you have learned
nothing — say so rather than hiding it.

- **Justify your preferred metric in writing.** The defensible answer for
  seizure detection: **recall (sensitivity) is primary**, because a missed
  crisis has a clinical cost and a false alarm does not. Report F1 as the
  balance, and precision to show the false-alarm rate. Argue it; do not just
  list the numbers.
- Recall the asymmetry explicitly: with a 95/5 split, precision 0.95 and recall
  0.05 gives F1 ≈ 0.09 despite accuracy ≈ 0.95.

## Trap 4 — 21 records is not many

After removing the defective rows you have ~21 usable records. Consequences:

- A test split of 20% is **~4 records**. One record can swing recall by tens of
  percentage points. Always report the count, and prefer **cross-validation**
  (`StratifiedGroupKFold` with `groups=record_id`) for model selection, keeping
  the held-out split as a final check.
- Do not tune hyperparameters against the test set. Tune with CV, then touch
  the test set once.
- Report variability (mean ± std across folds), not a single number.

## Trap 5 — class imbalance handling

For Part XII, if you rebalance (`class_weight="balanced"` for SVM/RF/DT,
`SMOTE` from `imbalanced-learn`, or threshold tuning), you change the
precision/recall trade-off deliberately. Report the before/after of both
metrics, and never claim the accuracy improved — it usually drops while recall
rises, which is the point.

## Pre-flight checklist

Before writing results into the report:

- [ ] No record appears in both train and test (`assert` above).
- [ ] Scaler inside a `Pipeline`, fitted on train only.
- [ ] Accuracy, precision, recall, F1 and the confusion matrix all reported.
- [ ] Majority-class baseline printed alongside.
- [ ] `random_state` fixed; notebook runs top-to-bottom with no hidden state.
- [ ] Preferred metric chosen and **justified in prose**.
- [ ] Variability reported (CV folds or repeated runs), not one bare number.
- [ ] Event-level TP/FP/FN in addition to window-level, per `eeg-pipeline`.
