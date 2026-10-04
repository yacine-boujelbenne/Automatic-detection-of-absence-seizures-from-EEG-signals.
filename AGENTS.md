# AGENTS.md — DM-project (crises d'absence EEG)

Project instructions for any AI assistant or contributor working in this repo.
Read this before touching code.

## What this project is

A French data-mining mini-project: **automatic detection of absence seizures
from EEG signals**. The specification is `mini-prjet-data-mining.pdf`
(12 parts, 14 questions, 6–8 h, pair work).

Two deliverables:

1. `notebooks/` — one Python notebook with 12 mandated sections.
2. `reports/` — one French report (introduction, problématique, méthodologie,
   données, résultats, discussion, conclusion, limites, bibliographie).

## Non-negotiables

1. **Never fabricate data.** Les `.mat` contiennent des objets MATLAB MCOS
   `timetable` qui ne sont pas lisibles directement par `scipy.io` ou `h5py`.
   Utiliser les exports numériques seulement après vérification contre les
   cellules sources par `src/eeg_source_validation.py`. Si ce contrôle échoue,
   signaler le blocage ; ne jamais inventer de signal ou de métrique.
2. **Never assert the sampling rate.** The PDF mentions 256 Hz as an example.
   Measure it from the `Time` vector.
3. **Never assume the annotation units.** Le classeur actuel est
   `Annotations MAJ.xlsx` : `DebutTrace`, les débuts et les fins sont des horaires
   Excel. Appliquer la correction de l'enseignant, puis contrôler les bornes
   contre la durée réelle du signal.
4. **Never report accuracy alone.** With a heavy class imbalance a
   always-negative model scores ~95%. Report precision, recall, F1 and the
   confusion matrix, and print the majority-class baseline.
5. **No leakage.** Split by record, not by window — 50%-overlapping windows
   share half their samples, so a window-level split leaks. Scalers go inside a
   `Pipeline`.
6. **Prose, comments, plots and markdown in French.** Code identifiers in
   English are fine.

## Environment

Use the project interpreter, never the system Python:

```bash
venv/Scripts/python.exe <script>
```

Les versions nécessaires au notebook final sont dans `requirements.txt`.
`venv/` dépend du Python local de `.runtime/python/` : conserver les deux.
Certaines descriptions dans les skills décrivent l'audit initial ; les résultats
courants vérifiés sont ceux du notebook final et de `artifacts/final/`.

## Data facts (verified — do not re-derive)

- Les 44 `.mat` originaux et les 22 exports NPZ sont conservés dans `data/`.
- Le classeur utilisé est `Annotations MAJ.xlsx` ; le classeur initial dans
  `data/CRISES ABSENCES/` est conservé comme source historique.
- Cohorte finale : **16 enregistrements, 65 crises, 23 444 fenêtres**.
- Exclure les doublons conflictuels `230406B-A` et `230515B-G`, ainsi que
  `210504B-C` dont une annotation dépasse la durée du signal.
- Les trois exports sans annotation dans le classeur mis à jour ne sont pas
  considérés automatiquement comme hors crise.
- `210914B-A` déclare 6 crises, mais possède 11 paires réelles ; conserver les
  paires renseignées. Colonnes mal orthographiées : `DebutCE5`, `DebtCE11`.
- Exemple de l'enseignant : `190304A-E`, origine 10:13:35, crise 10:35:06–10:35:13,
  soit 1291–1298 secondes depuis le début et une durée de 7 secondes.
- Les contrôles des valeurs, temps et canaux sources sont dans
  `artifacts/final/source_validation.json` ; les exclusions dans `cohort_audit.csv`.

## Pipeline order (do not reorder)

load → audit → segment → label → extract features → **split** → scale → train
→ evaluate → reconstruct intervals → error analysis

Window: 2 s, 50% overlap (512 samples / hop 256 at 256 Hz). Label a window
crisis when ≥50% of its samples fall in `[debut, fin]`, conformément au PDF.
L'ancienne référence utilise `(debut, fin]` ; cette différence est documentée
dans le notebook final. Features: 8 temporal
(mean, std, var, min, max, amplitude, RMS, energy) + 4 band powers
(Delta 0.5–4, Theta 4–8, Alpha 8–13, Beta 13–30 Hz) via one-sided FFT.

Verified reference implementation and its 76-check test suite:
`.claude/skills/eeg-pipeline/reference/`.

## Quality gates

```bash
venv/Scripts/python.exe -m ruff check .
venv/Scripts/python.exe -m ruff format .
cd .claude/skills/eeg-pipeline/reference && ../../../../venv/Scripts/python.exe test_reference.py
```

## Skills

`.claude/skills/` is shared by Claude Code and opencode. Read the relevant
skill before starting a section:

| Skill | Use for |
| --- | --- |
| `eeg-data-audit` | loading/parsing the workbook, the Deb/Fin columns |
| `eeg-mat-reader` | opening the `.mat` files, `MatlabOpaque` errors |
| `eeg-pipeline` | windowing, features, models, metrics, intervals |
| `eeg-validation` | splitting, leakage, imbalance, metric choice |
| `notebook-contract` | the notebook deliverable and figure rules |
| `report-fr` | the French report |
| `project-env` | venv, tooling, linting, repo hygiene |
