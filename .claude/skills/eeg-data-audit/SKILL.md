---
name: eeg-data-audit
description: Use when loading, parsing or trusting the CRISES ABSENCES dataset, the "Crises absences.xlsx" annotation workbook, the DebCE/FinCE/Intercrit columns, or any .mat EEG file in data/CRISES ABSENCES. Encodes the verified defects in this specific dataset (Excel padding rows, duplicate record 230515B-G, missing 211222B-G signal, empty 200625A-F, NombreCE mismatch, misspelled columns DebutCE5/DebtCE11, non-second time units) and provides a runnable audit script. Read BEFORE writing any data-loading code.
---

# EEG data audit (CRISES ABSENCES)

The delivered data is dirty in specific, verified ways. Every one of these
silently produces wrong numbers rather than an error, so audit first.

## Always run the audit before any other work

```bash
python .claude/skills/eeg-data-audit/scripts/audit_annotations.py
```

It exits `1` when it finds blocking defects and writes
`artifacts/annotations_audit.csv`. Do not start modelling while it exits `1`;
either fix the input or write the defect down explicitly in the report.

## Verified defects in this dataset

Read `artifacts/annotations_audit.csv` for the authoritative per-record numbers.
Summary of what is actually wrong:

| Defect | Detail | Consequence |
| --- | --- | --- |
| Excel padding | The sheet has 1,048,574 rows; only 24 are real records. Two padding rows even carry a `NombreCE` value (98, 196) with no ID. | `read_excel` without filtering gives a garbage-shaped frame. |
| ID formatting | IDs appear as `190304A-E` in the workbook but `190304A_E_0000d.mat` in filenames. Some cells carry stray whitespace. | Naive `merge`/`map` on the string joins nothing. |
| Duplicate record | `230515B-G` appears **twice** with conflicting data (age 9 vs 8, different annotation values). | Duplicates leak across a train/test split and inflate scores. |
| Annotated, no signal | `211222B-G` has 8 annotated seizures but **no `.mat` file at all**. | Cannot be used; drops out of the joined dataset. |
| Signal, no annotation | `200625A-F` has a `.mat` file but **zero** Deb/Fin pairs (declared `NombreCE=2`). | Unsupervised only; cannot be a classification target. |
| Declared count lies | `210914B-A` declares `NombreCE=6` but has **11** Deb/Fin pairs. It is also the only row with `TypeAbs='atypiques'`. | Trust the actual pairs, not the declared count. |
| Misspelled columns | `DebutCE5` (not `DebCE5`) and `DebtCE11` (typo, not `DebCE11`). | `df['DebCE5']` raises `KeyError`; positional indexing mispairs columns. |
| Fewer interictal cols | Only `Intercrit1..Intercrit7` for up to 12 seizures. | Index out of range if assumed 12. |
| **Time units are not seconds** | Start times sit in `[0.34, 0.64]`; durations have median `1.3e-4`, max `0.125`. At 256 Hz one sample is `0.0039 s`, so a median duration of `1.3e-4` is **thirty times smaller than a single sample** — physically impossible if the column were seconds. | The absolute scale factor is unknown. Do not hardcode a conversion. |

### 22 usable records

24 workbook rows → 23 unique IDs → **22** have a matching `.mat` file. After
dropping `200625A-F` (no annotation), the supervised cohort is **21 records /
103 annotated seizure events**. Report this number explicitly; it is small and
it is the single biggest limitation of the study.

## Rules

1. **Filter by ID regex, never by row count.** `^\d{6}[A-H]-[A-H]$` after
   `.str.strip()`. Then `reset_index(drop=True)`.
2. **Never index Deb/Fin positionally.** Detect by prefix and pair by the
   trailing integer, then report unpaired halves as orphans. `pair_columns()`
   in the audit script does this.
3. **Resolve the duplicate before splitting.** Pick one `230515B-G` row
   explicitly and say which, or drop the record. Do not let it straddle the
   train/test boundary.
4. **Reconcile units empirically, after the signal is readable.** Compute the
   `Time` vector from the `.mat`, then check whether any annotation start
   exceeds the recording duration. If all 103 events fall inside a window far
   shorter than the annotations imply, the column is scaled — report the
   inferred factor and its evidence. Never assume.
5. **Keep the audit CSV.** It is the evidence that exploration was done
   properly, and the report can cite it.

## What can be answered right now, without the signal

- Q3 / Q4 (crisis start, end, count, duration stats) — from the workbook, once
  units are pinned down.
- The dataset-defect discussion and the class-imbalance percentage (Q7) can be
  computed per window only after segmentation, so it still needs the signal.

## What is blocked

Parts I, IV, V, VI and X all need the actual EEG samples, which are unreadable
in Python. See the `eeg-mat-reader` skill. Do not synthesise substitute data.
