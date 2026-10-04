# CLAUDE.md

Ce projet est pilote par **[AGENTS.md](AGENTS.md)**. Lis ce fichier avant toute
modification.

En bref :

- Specification : `mini-prjet-data-mining.pdf` (12 parties, 14 questions).
- Livrables : `notebooks/` (12 sections) + `reports/` (rapport en francais).
- Les `.mat` contiennent des objets MATLAB MCOS `timetable`. Les exports
  numériques sont contrôlés contre les valeurs sources par
  `src/eeg_source_validation.py`. Ne jamais fabriquer de données EEG.
- Toujours utiliser `venv/Scripts/python.exe`, jamais le Python systeme.
- Les skills de `.claude/skills/` sont partages avec opencode via
  `opencode.json` (`skills.paths`). Commence par `eeg-data-audit` et
  `eeg-mat-reader`.
