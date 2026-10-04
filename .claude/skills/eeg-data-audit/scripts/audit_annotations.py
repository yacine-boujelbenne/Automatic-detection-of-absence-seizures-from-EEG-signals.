#!/usr/bin/env python
"""Audit the CRISES ABSENCES annotations before any modelling.

Run:  python .claude/skills/eeg-data-audit/scripts/audit_annotations.py
      python .claude/skills/eeg-data-audit/scripts/audit_annotations.py --xlsx PATH --matdir PATH

Prints a deterministic report and writes artifacts/annotations_audit.csv.
Exits 1 if any blocking data defect is found.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

ID_RE = re.compile(r"^\d{6}[A-H]-[A-H]$")
MAT_RE = re.compile(r"^(\d{6}[A-H]_[A-H])_(\d{4})([a-zA-Z])\.mat$")

# Column names in the delivered workbook are NOT consistent. Observed defects:
#   DebCE1..DebCE4, then "DebutCE5", then DebCE10, then "DebtCE11" (typo).
# Match loosely by prefix + trailing digit. NEVER index these positionally.
DEB_PREFIXES = ("debce", "debutce", "debtce")
FIN_PREFIXES = ("fince", "findebutce", "findebtce")


def _norm(col: str) -> str:
    return col.lower().replace(" ", "")


def _num_suffix(col: str) -> int:
    digits = re.sub(r"\D", "", col)
    return int(digits) if digits else 0


def _matches(col: str, prefixes: tuple[str, ...]) -> bool:
    c = _norm(col)
    return any(c.startswith(p) and c[len(p) :].isdigit() for p in prefixes)


def deb_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if _matches(c, DEB_PREFIXES)), key=_num_suffix)


def fin_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if _matches(c, FIN_PREFIXES)), key=_num_suffix)


def intercrit_cols(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df.columns if _norm(c).startswith("intercrit")), key=_num_suffix)


def pair_columns(deb: list[str], fin: list[str]) -> list[tuple[str, str]]:
    """Pair Deb_i with Fin_i by their numeric suffix, not by position.

    Positional zip() silently mispairs whenever a Deb column exists without its
    Fin partner. Suffix matching keeps the pairing honest and exposes orphans.
    """
    fmap = {_num_suffix(c): c for c in fin}
    pairs: list[tuple[str, str]] = []
    for d in deb:
        f = fmap.get(_num_suffix(d))
        if f is not None:
            pairs.append((d, f))
    return pairs


def load_annotations(path: str) -> tuple[pd.DataFrame, int, int]:
    raw = pd.read_excel(path)
    n_raw = len(raw)
    idcol = raw.columns[0]
    raw[idcol] = raw[idcol].astype(str).str.strip()
    df = raw[raw[idcol].str.match(ID_RE)].reset_index(drop=True)
    return df, n_raw, int((raw.shape[0] - df.shape[0]))


def mat_index(matdir: str) -> dict[str, dict[str, list[str]]]:
    idx: dict[str, dict[str, list[str]]] = {}
    for name in sorted(os.listdir(matdir)):
        m = MAT_RE.match(name)
        if not m:
            continue
        rid = m.group(1).replace("_", "-")
        idx.setdefault(rid, {}).setdefault(m.group(3), []).append(name)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/CRISES ABSENCES/Crises absences.xlsx")
    ap.add_argument("--matdir", default="data/CRISES ABSENCES")
    ap.add_argument("--out", default="artifacts/annotations_audit.csv")
    args = ap.parse_args()

    df, n_raw, n_dropped = load_annotations(args.xlsx)
    idcol = df.columns[0]
    deb, fin, inter = deb_cols(df), fin_cols(df), intercrit_cols(df)
    pairs = pair_columns(deb, fin)
    mats = mat_index(args.matdir)

    problems: list[str] = []
    print("=" * 78)
    print("ANNOTATION AUDIT  --  %s" % args.xlsx)
    print("=" * 78)
    print("rows in sheet            : %d   (Excel pads to 1,048,576)" % n_raw)
    print("rows dropped (no id)     : %d" % n_dropped)
    print("usable records           : %d" % len(df))
    print("unique record ids        : %d" % df[idcol].nunique())
    print("Deb columns detected     : %d  %s" % (len(deb), deb))
    print("Fin columns detected     : %d  %s" % (len(fin), fin))
    print("paired Deb/Fin columns   : %d" % len(pairs))
    print("Intercrit columns        : %d  %s" % (len(inter), inter))
    print()

    dups = df[idcol].value_counts()
    dups = {k: int(v) for k, v in dups[dups > 1].items()}
    if dups:
        problems.append("duplicate ids: %s" % dups)
        print("!! DUPLICATE IDS (conflicting annotation rows): %s" % dups)
        for rid in dups:
            sub = df[df[idcol] == rid]
            print("   %s -> %d rows, ages %s" % (rid, len(sub), list(sub.get("Age", []))))
        print()

    ids_x, ids_m = set(df[idcol]), set(mats)
    only_x, only_m = sorted(ids_x - ids_m), sorted(ids_m - ids_x)
    if only_x:
        problems.append("annotations with no .mat: %s" % only_x)
        print("!! ANNOTATED BUT NO SIGNAL FILE: %s" % only_x)
    if only_m:
        problems.append("signal files with no annotations: %s" % only_m)
        print("!! SIGNAL FILE BUT NO ANNOTATIONS: %s" % only_m)
    print()

    rows = []
    for _, r in df.iterrows():
        ev = [(r[d], r[f]) for d, f in pairs if pd.notna(r[d]) and pd.notna(r[f])]
        half = [(d, f) for d, f in pairs if pd.notna(r[d]) != pd.notna(r[f])]
        durs = [f - d for d, f in ev]
        declared = r.get("NombreCE", float("nan"))
        rows.append(
            dict(
                id=r[idcol],
                n_declared=declared,
                n_pairs=len(ev),
                n_orphan_halfpairs=len(half),
                n_intercrit=int(sum(pd.notna(r[c]) for c in inter)),
                t_start_min=min((d for d, _ in ev), default=float("nan")),
                t_start_max=max((d for d, _ in ev), default=float("nan")),
                dur_mean=sum(durs) / len(durs) if durs else float("nan"),
                dur_total=sum(durs),
                has_signal=r[idcol] in ids_m,
            )
        )
    audit = pd.DataFrame(rows)

    print("-" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(audit.to_string(index=False))
    print("-" * 78)
    print()

    bad = audit[audit.n_declared.notna() & (audit.n_declared != audit.n_pairs)]
    if len(bad):
        problems.append("NombreCE != actual Deb/Fin pairs for: %s" % list(bad.id))
        print("!! NombreCE DISAGREES WITH ACTUAL PAIRS (%d records):" % len(bad))
        for _, b in bad.iterrows():
            print("   %-12s declares %-4s  has %d pairs" % (b.id, b.n_declared, b.n_pairs))
        print()

    noann = audit[audit.n_pairs == 0]
    if len(noann):
        problems.append("zero annotated seizures: %s" % list(noann.id))
        print("!! ZERO ANNOTATED SEIZURES -> unusable for supervised training: %s" % list(noann.id))
        print()

    orph = audit[audit.n_orphan_halfpairs > 0]
    if len(orph):
        problems.append("orphan Deb/Fin halves: %s" % list(orph.id))
        print("!! Deb present without Fin (or vice versa): %s" % list(orph.id))
        print()

    alld = [
        r[f] - r[d]
        for _, r in df.iterrows()
        for d, f in pairs
        if pd.notna(r[d]) and pd.notna(r[f])
    ]
    if alld:
        s = pd.Series(alld)
        print("seizure events total    : %d" % len(s))
        print("duration mean/median    : %.6g / %.6g" % (s.mean(), s.median()))
        print("duration min/max        : %.6g / %.6g" % (s.min(), s.max()))
        print("negative durations      : %d" % int((s < 0).sum()))
        print()
        print("!! UNIT CHECK: the PDF states these are SECONDS. If the median duration")
        print("   is far below 1/fs, the column is NOT seconds -- do not assume. Read the")
        print("   signal Time vector first (see eeg-mat-reader skill).")
        if s.median() < 0.01:
            problems.append("annotation durations implausibly small for seconds (median %.3g)" % s.median())
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    audit.to_csv(args.out, index=False)
    print("wrote %s" % args.out)

    if problems:
        print("\nBLOCKING DEFECTS (%d):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        print("\nResolve or explicitly document each one before modelling.")
        return 1
    print("\nNo blocking defects found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
