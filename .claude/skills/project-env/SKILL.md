---
name: project-env
description: Use when setting up, repairing or standardising the DM-project working environment - the venv, the missing Jupyter/papermill tooling, ruff/pytest linting, the pandas 3.x and numpy 2.x migration gotchas, the 800 MB zip that must not be committed, or the absence of version control. Read before installing anything or writing the first cell of the notebook.
---

# Project environment

## Current state (measured, not assumed)

- Interpreter: `venv/Scripts/python.exe`, **Python 3.13.7**
- Installed and sufficient: `numpy 2.5.3`, `pandas 3.0.6`, `scipy 1.18.1`,
  `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `seaborn 0.13.2`,
  `openpyxl 3.1.5`, `h5py 3.16.0`
- **Missing and needed:** `jupyter`, `notebook`/`jupyterlab`, `ipykernel` — the
  primary deliverable is a notebook and there is currently no way to open one.
  Also missing: `imbalanced-learn` (Part XII rebalancing), `pytest`,
  `ruff`.
- **Not installed, check before planning around them:** MATLAB, GNU Octave,
  PyWavelets, torch/tensorflow. No MATLAB means the `.mat` route in
  `eeg-mat-reader` is not currently available.
- **Not a git repository.** No version control, so no history and no undo.
- `CRISES ABSENCES.zip` is **801 MB** and still present next to the extracted
  `data/` tree. It must never enter version control.
- `data/CRISES ABSENCES/~$Crises absences.xlsx` is an Excel lock file, safe to
  delete.

## Always use the venv interpreter

```bash
venv/Scripts/python.exe <script>          # Windows, Git Bash
```

Do not `pip install` into the system Python, and do not mix interpreters — that
is the usual cause of "import works in my notebook but not in the script".

## Install the missing essentials

```bash
venv/Scripts/python.exe -m pip install jupyterlab ipykernel imbalanced-learn pytest ruff
venv/Scripts/python.exe -m ipykernel install --user --name dm-project --display-name "Python (dm-project)"
```

Pin what you install into `requirements.txt` so the environment is reproducible
for the grader. Keep `pandas 3.x` / `numpy 2.x` in the pin — do not "fix" them
by downgrading unless a specific error forces it.

## pandas 3.x / numpy 2.x gotchas that will bite this project

- **Copy-on-Write is the default.** Chained assignment
  `df[df.a > 1]["b"] = 0` no longer works. Use `df.loc[mask, "b"] = 0`.
- **`.map` vs `.applymap`**: element-wise `applymap` is gone; use
  `df.map(...)` or `Series.map(...)`.
- **String dtype is PyArrow-backed by default.** `read_excel` string columns
  behave differently around missing values than the old object dtype — this is
  why the audit script coerces IDs with `.astype(str).str.strip()` explicitly.
- Removed kwargs such as `errors="ignore"` in `read_csv` raise, do not warn.
- In numpy 2.x, `np.float_`/`np.int_` aliases are gone; use `np.float64`/
  `np.int64`. `np.NaN` is `np.nan`.
- Always build test cases for array code — several genuine bugs in this project
  (subtracting column names instead of values) were caught only by
  `test_reference.py`, not by running the pipeline.

## Quality gates

```bash
venv/Scripts/python.exe -m ruff check .          # lint
venv/Scripts/python.exe -m ruff format .         # format
venv/Scripts/python.exe -m pytest -q             # tests
```

`ruff` doubles as the formatter and linter and is what `opencode.json` wires up
for Python LSP, so keeping the code ruff-clean makes editor diagnostics useful
instead of noise.

## Repository hygiene

If you initialise git, `.gitignore` must cover:

```gitignore
venv/
data/
CRISES ABSENCES.zip
artifacts/
.ipynb_checkpoints/
~$*.xlsx
__pycache__/
```

`data/` and the zip together are ~1.6 GB of binaries. Keep them out of history;
reference them by path in the report instead.

## Working layout

```
notebooks/    the deliverable notebook
src/          reusable modules (feature extraction lives in the skill's reference/)
artifacts/    audit CSVs, generated tables, exported figures
reports/      the French report
```

`artifacts/` is regenerable — treat it as a cache, not a source of truth, and
make every script that writes into it idempotent.
