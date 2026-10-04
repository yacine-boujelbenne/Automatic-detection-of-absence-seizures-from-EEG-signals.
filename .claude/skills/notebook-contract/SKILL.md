---
name: notebook-contract
description: Use when creating, structuring, reviewing or cleaning the deliverable Python notebook for this EEG seizure-detection project, or when a figure needs axis labels, titles and units. Enforces the 12 mandated sections from the assignment, French-language output, deterministic execution and figure quality rules. Apply before submitting.
---

# Notebook contract

The assignment mandates **one Python notebook** containing twelve specific
things. Graders check for them by section. Build the skeleton in this order and
never merge sections.

## Mandated sections, in order

| # | Section | Assignment part | Must contain |
| --- | --- | --- | --- |
| 1 | Chargement des données | — | imports, config, `random_state`, `fs` |
| 2 | Exploration | I | shape, dtypes, missing values, fs, duration per record |
| 3 | Visualisations | I, Q2 | signal vs time, labelled |
| 4 | Traitement des annotations | II | Deb/Fin → interval list, per-record table |
| 5 | Segmentation | IV | windowing, 50% overlap, window count |
| 6 | Extraction des caractéristiques | V, VI | 8 temporal + 4 band powers |
| 7 | Entraînement des modèles | VIII | DT, RF, KNN, SVM |
| 8 | Évaluation | IX | 4 metrics + confusion matrices |
| 9 | Comparaison | IX | the results table |
| 10 | Détection temporelle | X | intervals reconstructed vs annotations |
| 11 | Analyse des erreurs | XI | TP/FP/FN/TN |
| 12 | Améliorations | XII | ≥2 improvements, measured |

Keep the whole thing **in French** — the assignment, the data and the report
are French. Code identifiers in English are fine; prose, comments, plot labels
and markdown in French.

## Figure rules (Question 2 is graded on this explicitly)

Every figure needs: **x-axis label with unit, y-axis label with unit, and a
title.** A plot without them loses the point of the question.

```python
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Amplitude EEG (µV)")
ax.set_title("Signal EEG — enregistrement 190304A-E")
```

Also: distinguish the two classes visually in the Part III figure (shaded spans
or colour, not just a legend), show real vs predicted crisis periods in Part X,
and always label which record and which channel a figure shows.

## Execution hygiene

- **Restart the kernel and Run All before submitting.** No stale outputs, no
  `In[ ]` prompt numbers, no stray `print` debugging.
- **No hidden state.** Every step must run top-to-bottom from a clean kernel.
  Do not rely on a variable defined two cells above and then edited by hand.
- **Seed everything.** `random_state=42` on every estimator and split.
- **Guard the long steps.** The audit and the feature extraction take time; make
  the expensive part idempotent and cache to `artifacts/` rather than
  recomputing silently on every run.
- **Keep outputs clean.** Use `display(...)` or a trailing expression, not
  `print(df)` for wide frames.

## Code style that keeps the notebook readable

- Small functions with names in French or clear English (`extraire_features`,
  `reconstruire_intervalles`), not 200-line inline cells.
- Docstring on each function: what it takes, what it returns.
- Comment the *why*, never the obvious *what*.
- No commented-out dead code left in the final notebook.

## Review pass before submitting

Read the notebook once as if you were the grader:

- [ ] All twelve sections present, in order, with French headings.
- [ ] Every figure has title + both axis labels + units.
- [ ] Class 0 / class 1 visually distinguished in the Part III figure.
- [ ] Q3/Q4 answered with an actual printed table of crises.
- [ ] Q7 class percentages actually computed and printed.
- [ ] Q11 correlation matrix shown, with the three sub-answers in prose.
- [ ] Q12 all four models trained; Q13 answered in prose with justification.
- [ ] Q14 reconstructed intervals compared to annotations side by side.
- [ ] Q15 error analysis present, both window- and event-level.
- [ ] Part XII: at least two improvements, each with a measured before/after.
- [ ] Kernel restarted, Run All clean, no errors, no stale outputs.
- [ ] Data defects documented (see `eeg-data-audit`) rather than hidden.
