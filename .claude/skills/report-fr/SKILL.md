---
name: report-fr
description: Use when writing, structuring or reviewing the French report (rapport) deliverable for the EEG absence-seizure data-mining project, or when drafting the introduction, problématique, méthodologie, discussion, conclusion, limites or bibliography sections.
---

# Rapport (French deliverable)

The assignment requires a report with a fixed set of sections. Write it in
**French**, in academic register, past tense for what was done and present tense
for what the results show.

## Required structure

1. **Introduction** — the clinical context (EEG, crises d'absence), why
   automatic detection matters, and the objective. One or two paragraphs.
2. **Problématique** — the precise question. E.g. *« Peut-on détecter
   automatiquement les crises d'absence à partir de signaux EEG, et avec quelles
   caractéristiques et quel modèle ? »* Keep it to one question.
3. **Méthodologie** — the full pipeline, in order, with the real numbers:
   sampling rate, window length, overlap, the 12 features, the label rule, the
   split strategy. Include a pipeline diagram (the assignment asks for one).
4. **Données** — the cohort: how many records, how many seizure events, and
   **the defects you found and how you handled each**. This section is where the
   `eeg-data-audit` output belongs. Do not hide it.
5. **Résultats** — the metrics table for the four models, plus the correlation
   analysis (Q11) and the class-comparison findings (Q10).
6. **Discussion / Analyse des performances et des erreurs** — which model won and
   *why*; TP/FP/FN; where the failures come from; window-level vs event-level
   disagreement.
7. **Conclusion** — direct answer to the problématique.
8. **Limites et perspectives** — be specific and honest (see below).
9. **Bibliographie** — real, formatted references. Not invented ones.

## Style rules

- French punctuation: non-breaking space before `: ; ! ?`, and inside
  guillemets `« … »`.
- Decimal comma in prose: `0,95` — but keep the dot in code and tables.
- Define the acronym once: *crises d'absence (CE)*, then CE.
- No bullet-point fragments in the discussion — write prose.
- Every figure referenced as « Figure N » in the text, numbered in order of
  first mention. Same for tables.
- Never claim a result you did not measure. If a number is missing, run it.

## The parts that carry the most marks

**Question 13 — which metric to prefer.** Do not just list metrics. Argue it:

> L'accuracy est ici trompeuse : avec un déséquilibre aussi marqué, un modèle
> prédisant systématiquement « hors crise » atteint déjà une accuracy élevée sans
> détecter aucune crise. Nous privilégions donc le **rappel (sensibilité)**, car
> une crise manquée a un coût clinique alors qu'une fausse alarme est moins
> grave. Le F1-score est reported comme compromis, et la précision traduit le
> taux de fausses alarmes.

**Limites.** The honest ones, which are strong because they are specific:

- The cohort is tiny — ~21 usable records, ~103 events — so all metrics carry
  wide uncertainty; a test split is only a handful of records.
- The signal files could not be read in Python (MATLAB `timetable` objects); the
  exact impact on the results is stated.
- The annotation time units are not those stated in the assignment; the inferred
  scaling is documented, and the consequences for window labelling are discussed.
- Windows overlap, so observations are not independent; the split is
  record-level for this reason.
- Some records were excluded (`200625A-F` unannotated, `211222B-G` no signal) —
  say so and justify.
- Absence-seizure detection in the real clinic uses scalp EEG with known
  electrode montage; this dataset is a simplified teaching extract.

**Perspectives** — pick from the Part XII list you actually measured, plus:
cross-validation instead of a single split, event-level evaluation as the primary
metric, and interictal-vs-seizure band-power ratios as features.

## Numbers to keep consistent between report and notebook

Copy them from actual output, never from memory: record count, event count,
window count, class percentages, the four metrics per model, and the
reconstructed intervals. A mismatch between the report and the notebook is the
easiest mistake for a grader to catch.
