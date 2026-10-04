#!/usr/bin/env python
"""Probe CRISES ABSENCES .mat files and report whether the signal is readable.

Run:  python .claude/skills/eeg-mat-reader/scripts/probe_mat.py
      python .claude/skills/eeg-mat-reader/scripts/probe_mat.py --file "data/.../190304A_E_0000d.mat"

The delivered .mat files are MATLAB v5 containers holding an MCOS-serialized
`timetable` class object. scipy.io cannot reconstruct MCOS objects: it returns a
MatlabOpaque placeholder whose _ObjectMetadata is a raw uint32 descriptor, not
the numeric payload. This script proves that, so nobody wastes hours
"debugging" the loader or, worse, fabricates signal data.

Exit 0 = signal extracted. Exit 2 = signal NOT extractable (needs MATLAB/Octave
or a re-export from the data provider).
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys

HEADER = 116
HDF5_SIG = b"\x89HDF\r\n\x1a\n"
ZLIB_MAGIC = b"\x78\x9c"


def describe_container(path: str) -> tuple[str, int, bool]:
    """Return (kind, size_bytes, is_zlib_compressed).

    MATLAB v5 header layout: 116 bytes text, 8 bytes subsystem offset,
    2 bytes version (0x0100), 2 bytes endian ("IM" little / "MI" big).
    MATLAB v7.3 is HDF5: the 8-byte HDF5 signature sits at offset 116.
    """
    with open(path, "rb") as fh:
        head = fh.read(512)
    size = os.path.getsize(path)
    if not head.startswith(b"MATLAB"):
        return "not a MATLAB file", size, False
    if head[HEADER : HEADER + 8] == HDF5_SIG:
        return "v7.3 (HDF5) - h5py can read this", size, False
    version = int.from_bytes(head[124:126], "little")
    endian = head[126:128]
    kind = "v5 (version=0x%04x, endian=%s)" % (version, endian.decode("latin-1", "replace"))
    # First data element tag starts at 128; miCOMPRESSED = type 15 (0x0f).
    compressed = len(head) >= 132 and head[128:130] == b"\x0f\x00"
    return kind, size, compressed


def try_h5py(path: str) -> str | None:
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as fh:
            found: list[str] = []

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    found.append("%s%s %s" % (name, obj.shape, obj.dtype))

            fh.visititems(visit)
            return "h5py found %d datasets: %s" % (len(found), found[:6])
    except Exception as exc:  # noqa: BLE001
        return "h5py cannot open: %s: %s" % (type(exc).__name__, exc)


def try_scipy(path: str) -> tuple[bool, str]:
    import numpy as np
    import scipy.io as sio

    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data if not k.startswith("__")]
    if not keys:
        return False, "no variables"
    report = []
    extracted = False
    for k in keys:
        v = data[k]
        kind = type(v).__name__
        names = getattr(getattr(v, "dtype", None), "names", None)
        if kind == "MatlabOpaque" and names:
            cls = ""
            try:
                cls = str(np.asarray(v["_Class"]).ravel()[0])
                ts = str(np.asarray(v["_TypeSystem"]).ravel()[0])
            except Exception:  # noqa: BLE001
                ts = "?"
            meta = np.asarray(v["_ObjectMetadata"])
            report.append(
                "%s: MatlabOpaque class=%r typesystem=%r metadata=%s len=%d"
                % (k, cls, ts, meta.dtype, meta.size)
            )
        elif isinstance(v, np.ndarray) and v.dtype.names:
            report.append("%s: struct with fields %s" % (k, list(v.dtype.names)))
            extracted = True
        elif isinstance(v, np.ndarray) and v.dtype != object:
            report.append("%s: %s %s" % (k, v.dtype, v.shape))
            extracted = True
        else:
            report.append("%s: %s" % (k, kind))
    return extracted, "; ".join(report)


def search_plain_arrays(path: str, compressed: bool) -> str:
    """Last-ditch scan for an uncompressed float64 run in the raw bytes.

    For miCOMPRESSED (deflate) v5 files this is expected to find nothing: the
    payload is not stored as plain little-endian doubles, so a "plausible EEG
    range" here would be a coincidence, not a recovery. Do not trust it.
    """
    import numpy as np

    size = os.path.getsize(path)
    if size < 4096:
        return "file too small to contain signal"
    with open(path, "rb") as fh:
        fh.seek(HEADER)
        raw = fh.read(min(size - HEADER, 8 * 1024 * 1024))
    body = raw[: len(raw) - (len(raw) % 8)]
    arr = np.frombuffer(body, dtype="<f8")
    if arr.size < 512:
        return "no float64-aligned run in first 8 MiB"
    finite = arr[np.isfinite(arr)]
    if finite.size < 512:
        return "float64-aligned data present but mostly non-finite"
    lo, hi = float(finite.min()), float(finite.max())
    verdict = ""
    if compressed:
        verdict = " -- FILE IS DEFLATE-COMPRESSED, so this is a false positive; ignore it"
    return "float64-aligned run n=%d range=[%.4g, %.4g]%s" % (finite.size, lo, hi, verdict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="single .mat file to probe")
    ap.add_argument("--matdir", default="data/CRISES ABSENCES")
    ap.add_argument("--full", action="store_true", help="probe every .mat file")
    args = ap.parse_args()

    if args.file:
        targets = [args.file]
    else:
        targets = sorted(glob.glob(os.path.join(args.matdir, "*.mat")))
        if not args.full:
            d = [t for t in targets if t.endswith("d.mat")]
            h = [t for t in targets if t.endswith("h.mat")]
            targets = (d[:1] + h[:1]) or targets

    print("=" * 78)
    print("MATLAB .mat READABILITY PROBE")
    print("=" * 78)

    any_ok = False
    for t in targets:
        if not os.path.exists(t):
            print("\nMISSING: %s" % t)
            continue
        kind, size, compressed = describe_container(t)
        print("\n--- %s" % os.path.basename(t))
        print("    container : %s" % kind)
        print("    size      : %.2f MB" % (size / 1e6))
        if compressed:
            print("    payload   : deflate-compressed (miCOMPRESSED)")
        h5 = try_h5py(t)
        if h5:
            print("    h5py      : %s" % h5)
        try:
            ok, detail = try_scipy(t)
            print("    scipy.io  : %s" % detail)
            if ok:
                any_ok = True
                print("    VERDICT   : signal is readable")
                continue
        except Exception as exc:  # noqa: BLE001
            print("    scipy.io  : raised %s: %s" % (type(exc).__name__, exc))
        print("    raw scan  : %s" % search_plain_arrays(t, compressed))
        print("    VERDICT   : NOT extractable with Python alone")

    print("\n" + "=" * 78)
    if any_ok:
        print("At least one file yielded numeric data. Read it, do not guess it.")
        return 0
    print("RESULT: no .mat file in this dataset is readable via scipy/h5py.")
    print("")
    print("The payload is an MCOS-serialized MATLAB `timetable` object inside a")
    print("deflate-compressed v5 container. Options, in order of preference:")
    print("  1. BEST: ask the data provider to re-export each timetable as CSV or")
    print("     parquet, or as a v7.3 .mat, or as a .mat holding a numeric matrix.")
    print("     This is a small ask and unblocks the whole project.")
    print("  2. Use MATLAB (matlab.engine for Python) or GNU Octave to convert:")
    print("       tt = load('f.mat'); writetable(tt, 'f.csv');")
    print("  3. NOT VIABLE for these files: the body is deflate-compressed, so no")
    print("     raw-byte scan can recover the samples. Do not attempt it.")
    print("")
    print("Until then Parts I, IV, V, VI and X of the assignment cannot be answered.")
    print("Do not fabricate EEG samples to fill the gap -- a fabricated signal makes")
    print("every downstream metric meaningless and is worse than an honest blocker.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
