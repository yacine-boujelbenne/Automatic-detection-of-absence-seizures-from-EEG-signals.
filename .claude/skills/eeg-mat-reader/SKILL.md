---
name: eeg-mat-reader
description: Use when opening, converting or debugging the .mat EEG files in data/CRISES ABSENCES (190304A_E_0000d.mat etc.), when scipy.io.loadmat returns a MatlabOpaque or empty array, when h5py reports "file signature not found", or when asked to load the EEG signal. Documents that these are MCOS-serialized MATLAB timetable objects that Python cannot read, and gives the exact conversion routes to actually recover the samples.
---

# Reading the `.mat` EEG files

## The situation

Every file in `data/CRISES ABSENCES/` is a **MATLAB v5 container holding an
MCOS-serialized `timetable` class object**, deflate-compressed.

Confirm it yourself:

```bash
python .claude/skills/eeg-mat-reader/scripts/probe_mat.py
```

Verified output for a signal file:

```
container : v5 (version=0x0100, endian=IM)
payload   : deflate-compressed (miCOMPRESSED)
h5py      : cannot open: OSError (file signature not found)
scipy.io  : output_d: MatlabOpaque class='timetable' typesystem='MCOS' metadata=object len=1
VERDICT   : NOT extractable with Python alone
```

`scipy.io.loadmat` does not error. It returns a `MatlabOpaque` placeholder
whose `_ObjectMetadata` field is a one-element descriptor array, not the
samples. Iterating it, `.ravel()`-ing it, or casting it will hand you garbage
that *looks* like data. The file naming is also misleading:

- `*_0000d.mat` — 24–53 MB, the signal payload.
- `*_0000h.mat` — ~1.8 KB, header/descriptor only, no samples.

Neither is readable. There is no Python-only fix, and no amount of debugging
the loader will change that: the object graph is MATLAB class metadata that
only MATLAB's own runtime can instantiate.

## Routes to actually get the data, best first

### 1. Ask for a re-export (recommended, cheapest)

Request either format from whoever supplied the data — this is a small ask and
unblocks the entire project:

- one CSV or Parquet per record, or
- a v7.3 (HDF5) `.mat`, which `h5py` reads natively, or
- a `.mat` containing a plain numeric matrix rather than a `timetable`.

State the blocker plainly: *the delivered `.mat` files store the signal inside a
MATLAB `timetable` object, which `scipy`/`h5py` cannot decode; please re-export
as CSV or as a numeric-matrix `.mat`.* Also ask **which unit the `DebCE`/`FinCE`
columns use** — see the unit warning in `eeg-data-audit`.

### 2. Convert with MATLAB

```matlab
tt = load('190304A_E_0000d.mat');       % returns the timetable
writetable(tt, '190304A_E_0000d.csv');  % Time + EEG canal columns
```

From Python, drive it with the official engine:

```python
import matlab.engine
eng = matlab.engine.start_matlab()
eng.writetable(eng.load('190304A_E_0000d.mat'), '190304A_E_0000d.csv')
eng.quit()
```

### 3. GNU Octave

```bash
octave --eval "tt = load('190304A_E_0000d.mat'); writetable(tt, '190304A_E_0000d.csv');"
```

Neither MATLAB nor Octave is installed on this machine, so route 2/3 needs
setup first. Check before planning around it.

### 4. Not viable

Scanning raw bytes for float64 patterns cannot work here: the body is
deflate-compressed, so any "EEG-looking" range you find is a coincidence. Do
not spend time on it.

## Rules

1. **Never fabricate, simulate or reuse another record's samples** to fill the
   gap. Invented EEG produces confident, meaningless metrics — strictly worse
   than an honest blocker, and it would misrepresent the work in the report.
2. **Never assert a sampling rate.** The assignment mentions 256 Hz as an
   example, not as a property of these files. Read `fs` from the `Time` vector
   (`fs = 1/median(diff(Time))`) and report what you measured.
3. **Validate any recovered signal before using it.** Check row count, that
   `Time` is monotonic with a constant `dt`, the amplitude range, and that the
   number of EEG channels matches the assignment's table.
4. **State the blocker in the report** if it is still unresolved at submission,
   with the concrete request you made. That is a defensible result; invented
   data is not.

## Downstream

Once samples are available, the pipeline, windowing and feature definitions are
in the `eeg-pipeline` skill, and the leakage rules are in `eeg-validation`.
