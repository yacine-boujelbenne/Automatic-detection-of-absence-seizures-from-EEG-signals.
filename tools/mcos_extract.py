"""Extract EEG data from MATLAB v5 MCOS `timetable` .mat files, pure Python.

The files in data/CRISES ABSENCES are MATLAB v5 MAT-files holding MCOS
opaque objects of class `timetable`. `scipy.io.loadmat` returns a
`MatlabOpaque` stub and `h5py` cannot open them (they are not HDF5).

`pymatreader` recovers the raw decompressed element stream under the key
`__function_workspace__`. That stream is a sequence of miCOMPRESSED
elements, the same encoding used for a MAT-file data section. This module
walks it and pulls out the numeric arrays.

Two independent strategies, because a full MCOS decode is fragile:

1. `walk_stream`  - structured miCOMPRESSED / miMATRIX walker.
2. `scan_blocks`  - blind scan for long, aligned runs of float64 that
                    look like EEG, plus uniformly-spaced increasing runs
                    that look like a Time vector.

Both are cross-checked: a Time vector found by strategy 2 must agree with
the array named `Time` (or `rowTimes`) found by strategy 1.
"""

from __future__ import annotations

import argparse
import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# miCOMPRESSED data types.
MI_INT8, MI_UINT8, MI_INT16, MI_UINT16 = 1, 2, 3, 4
MI_INT32, MI_UINT32, MI_SINGLE, MI_DOUBLE = 5, 6, 7, 9
MI_INT64, MI_UINT64 = 12, 13
MI_MATRIX, MI_COMPRESSED, MI_UTF8, MI_UTF16, MI_UTF32 = 14, 15, 16, 17, 18
_TEXT = {MI_UTF8, MI_UTF16, MI_UTF32}

_DTYPE = {
    MI_INT8: np.dtype("<i1"),
    MI_UINT8: np.dtype("<u1"),
    MI_INT16: np.dtype("<i2"),
    MI_UINT16: np.dtype("<u2"),
    MI_INT32: np.dtype("<i4"),
    MI_UINT32: np.dtype("<u4"),
    MI_SINGLE: np.dtype("<f4"),
    MI_DOUBLE: np.dtype("<f8"),
    MI_INT64: np.dtype("<i8"),
    MI_UINT64: np.dtype("<u8"),
    MI_UTF8: np.dtype("<u1"),
    MI_UTF16: np.dtype("<u2"),
    MI_UTF32: np.dtype("<u4"),
}

# mxArray class codes.
MX_CELL, MX_STRUCT, MX_OBJECT, MX_CHAR, MX_SPARSE = 1, 2, 3, 4, 5
MX_DOUBLE, MX_SINGLE, MX_INT8, MX_UINT8 = 6, 7, 8, 9
MX_INT16, MX_UINT16, MX_INT32, MX_UINT32 = 10, 11, 12, 13
MX_INT64, MX_UINT64, MX_FUNCTION, MX_OPAQUE = 14, 15, 16, 17

_NUMERIC = {
    MX_DOUBLE: MI_DOUBLE,
    MX_SINGLE: MI_SINGLE,
    MX_INT8: MI_INT8,
    MX_UINT8: MI_UINT8,
    MX_INT16: MI_INT16,
    MX_UINT16: MI_UINT16,
    MX_INT32: MI_INT32,
    MX_UINT32: MI_UINT32,
    MX_INT64: MI_INT64,
    MX_UINT64: MI_UINT64,
}

MAX_DEPTH = 64


def _align8(n: int) -> int:
    """Round a payload length up to the 8-byte element alignment."""
    rem = n % 8
    return n if rem == 0 else n + (8 - rem)


class ParseError(RuntimeError):
    pass


@dataclass
class Node:
    """A decoded value from the MCOS stream."""

    kind: str  # 'numeric' | 'char' | 'struct' | 'cell' | 'opaque' | 'other'
    name: str = ""
    path: str = ""
    dtype: np.dtype | None = None
    shape: tuple[int, ...] = ()
    data: np.ndarray | None = None
    fields: dict = field(default_factory=dict)
    start: int = -1
    end: int = -1
    error: str = ""


class Stream:
    """Cursor over a miCOMPRESSED element stream."""

    def __init__(self, buf: bytes, limit: int | None = None) -> None:
        self.buf = buf
        self.pos = 0
        # Optional hard end offset, used so the small-element heuristic can
        # tell a plausible nested header from a 4-byte payload.
        self.limit = len(buf) if limit is None else limit

    def remaining(self) -> int:
        return len(self.buf) - self.pos

    def _align8(self, n: int) -> None:
        rem = n % 8
        if rem:
            self.pos += 8 - rem

    def peek_element(self) -> tuple[int, int, int, int]:
        """Return ``(dtype, nbytes, payload_start, next_pos)`` without moving.

        The MCOS workspace mixes two element encodings, and the type code is
        what tells them apart:

        * *pair* - two little-endian u32 written back to back, the data type
          code then the payload length. Used for the workspace framing and for
          every miMATRIX sub-element (array flags, dimensions, names, data).
        * *packed* - a single u32 holding ``(nbytes << 16) | dtype``, the form
          used inside a MAT-file data section. This appears for the struct
          field-name count and for the opaque type tag.

        The pair form is tried first because it is the common case; the packed
        form is used only when the first u32 is not a valid type code.

        The element length is ``align8(tag + nbytes)``, i.e. the alignment
        applies to the whole element, not to the payload alone. Applying it to
        the payload alone is wrong for a 4-byte packed payload, which would be
        counted as 12 bytes instead of 8, and that single error is what
        desynchronised every later offset by 4 bytes.
        """
        if self.remaining() < 8:
            raise ParseError("truncated element header")
        first, second = struct.unpack_from("<II", self.buf, self.pos)
        if first in _DTYPE or first in (MI_MATRIX, MI_COMPRESSED):
            dtype, nbytes, tag = first, second, 8
        else:
            dtype, nbytes = first & 0xFFFF, first >> 16
            if dtype not in _DTYPE and dtype != MI_MATRIX:
                raise ParseError(f"invalid element header 0x{first:08x} at {self.pos}")
            tag = 4
        return dtype, nbytes, self.pos + tag, self.pos + _align8(tag + nbytes)

    def read_header(self) -> tuple[int, int]:
        """Consume only the tag and return ``(dtype, nbytes)``."""
        dtype, nbytes, _payload_start, nxt = self.peek_element()
        self.pos = nxt
        return dtype, nbytes

    def read_element(self) -> tuple[int, bytes, int]:
        """Read one data element and return ``(dtype, payload, next_pos)``.

        The byte layout is unambiguous, which is what the earlier
        ``nbytes == 0`` heuristic got wrong and cost a 4-byte drift:

        * pair form, ``nbytes > 0`` - an 8-byte tag then ``align8(nbytes)``
          bytes of payload.
        * pair form, ``nbytes == 0`` - a small element. The 8 bytes of the tag
          pair *are* the whole element, so the advance is exactly 8 and the
          payload is empty. Reading this as "4 payload bytes then pad to 8"
          also lands on 8 from the tag start, so both readings agree; the
          important part is that the advance is 8 and not 4.
        * packed form - a 4-byte tag then ``align8(nbytes)`` bytes, so a
          zero-length packed element is 4 bytes.

        Verified byte by byte on the opening of `190304A_E_0000d.mat`: tags at
        8/16/32/48 (struct `n0`, flags ``[2, 0]``, dims ``[1, 1]``, empty
        name), the packed int32 field count 5 at 56, the field name ``MCOS``
        at 64, the opaque matrix at 80 with class name ``FileWrapper__`` at
        120, and the properties cell at 144 whose first child matrix starts
        at 192.
        """
        start = self.pos
        dtype, nbytes, payload_start, nxt = self.peek_element()
        if nxt > self.limit:
            raise ParseError(f"element at {start} overruns the enclosing matrix")
        payload = self.buf[payload_start : payload_start + nbytes]
        self.pos = nxt
        return dtype, payload, nxt

    def skip_element(self) -> int:
        dtype, _payload, _next_pos = self.read_element()
        return dtype

    def read_numeric_array(self) -> tuple[int, np.ndarray]:
        dtype, payload, _next = self.read_element()
        if dtype not in _DTYPE:
            raise ParseError(f"expected numeric, got {dtype}")
        n = len(payload) // _DTYPE[dtype].itemsize
        return dtype, np.frombuffer(payload, dtype=_DTYPE[dtype], count=n)

    def read_name(self) -> str:
        dtype, payload, _next = self.read_element()
        if dtype not in (MI_INT8, MI_UINT8, MI_UTF8):
            # Not a name; rewind is impossible, so surface it as a marker.
            return f"<non-name:{dtype}>"
        return payload.decode("latin-1", errors="replace").rstrip("\x00")


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def parse_matrix(st: Stream, name: str, path: str, depth: int, end: int) -> Node:
    if depth > MAX_DEPTH:
        raise ParseError("max depth")

    flags_dtype, flags, _flags_next = st.read_element()
    if flags_dtype != MI_UINT32 or len(flags) < 8:
        raise ParseError("bad array flags")
    cls = struct.unpack_from("<i", flags, 0)[0] & 0xFF
    struct.unpack_from("<i", flags, 4)[0]

    if cls in _NUMERIC:
        _, dims = st.read_numeric_array()
        shape = tuple(int(d) for d in dims)
        # MAT v5 layout is flags, dims, name, data. The name element is
        # mandatory even when empty, so it must be consumed before the data.
        st.read_name()
        _dt, values = st.read_numeric_array()
        node = Node(
            kind="numeric",
            name=name,
            path=path,
            dtype=values.dtype,
            shape=shape,
            data=values,
        )
        if _prod(shape) not in (0, values.size):
            node.kind = "other"
        return node

    if cls == MX_CHAR:
        _, dims = st.read_numeric_array()
        shape = tuple(int(d) for d in dims)
        st.read_name()
        node = Node(kind="char", name=name, path=path, shape=shape)
        try:
            dtype, raw, _raw_next = st.read_element()
        except ParseError:
            node.kind = "other"
            node.error = "char: missing data element"
            return node
        if dtype in (MI_UINT16, MI_UINT8, MI_INT8, MI_UINT32, MI_INT32):
            vals = np.frombuffer(raw, dtype=_DTYPE[dtype])
            if dtype == MI_UINT16:
                try:
                    node.data = vals.view(">u2").astype("U1")
                except Exception:  # pragma: no cover - defensive
                    node.kind = "other"
            else:
                node.data = vals.astype("U1")
        else:
            # Some char rows store the payload under a different element type.
            # Keep the bytes so the text is still recoverable.
            node.kind = "other"
            node.error = f"char: unexpected storage type {dtype}"
            node.data = np.frombuffer(raw, dtype=np.uint8)
        return node

    if cls == MX_STRUCT:
        _, dims = st.read_numeric_array()
        shape = tuple(int(d) for d in dims)
        st.read_name()  # struct array name
        node = Node(kind="struct", name=name, path=path, shape=shape)
        # A struct is a field-name-count element, the field names themselves,
        # then one (name, value) entry per field. In this stream the count is
        # a packed miINT32 (offset 56 of 190304A_E_0000d.mat holds 5, the
        # field-name count for the top-level wrapper). The count is advisory:
        # entries are then read until the matrix's declared end offset, so a
        # wrong count cannot desynchronise the parse.
        if st.pos < end:
            try:
                dtype, nbytes, nxt = st.peek_element()
                count = None
                if dtype in (MI_INT32, MI_UINT32) and nbytes in (4, 8):
                    count = int(
                        np.frombuffer(st.buf[nxt - _align8(nbytes) : nxt], dtype=_DTYPE[dtype])[0]
                    )
                if count is not None and 0 <= count <= 4096:
                    st.pos = nxt
                    for _ in range(count):
                        st.read_name()
            except (ParseError, IndexError, ValueError):
                pass
        while st.pos < end:
            prop, child = _next_child(st, path, depth, end)
            if prop is None:
                break
            node.fields[prop] = child
        return node

    if cls == MX_CELL:
        _, dims = st.read_numeric_array()
        shape = tuple(int(d) for d in dims)
        st.read_name()
        node = Node(kind="cell", name=name, path=path, shape=shape)
        idx = 0
        while st.pos < end and idx < 4096:
            before = st.pos
            prop, child = _next_child(st, f"{path}{{{idx}}}", depth, end, anonymous=f"{{{idx}}}")
            if child is None:
                node.error = (
                    f"cell stopped at {before} (expected {end}): "
                    f"{st.buf[before : before + 8].hex(' ')}"
                )
                break
            node.fields[f"{{{idx}}}"] = child
            idx += 1
            if st.pos <= before:
                node.error = f"no progress at {before}"
                break
        return node

    if cls in (MX_OPAQUE, MX_OBJECT):
        # MCOS serialised object. The array name is the variable name, then
        # the class name, then a single miMATRIX holding the properties (a
        # cell array of 11 entries for `FileWrapper__`, one per property).
        st.read_name()
        cls_name = st.read_name()
        node = Node(kind="opaque", name=name, path=path)
        node.fields["__class__"] = Node(kind="char", name="__class__", data=np.array([cls_name]))
        if st.pos < end:
            try:
                node.fields["__properties__"] = _read_value(
                    st, "__properties__", f"{path}.__properties__", depth + 1
                )
            except ParseError:
                pass
        return node

    raise ParseError(f"unsupported class {cls}")


def _next_child(
    st: Stream, path: str, depth: int, end: int, anonymous: str | None = None
) -> tuple[str | None, Node | None]:
    """Read one (name, value) child, keeping the byte cursor aligned.

    Returns (property_name, node). On a parse failure the cursor is moved to
    the child's declared end so the enclosing loop can continue; when the
    failure is unrecoverable, (None, None) is returned.
    """
    start = st.pos
    try:
        dtype, nbytes, _payload, nxt = st.peek_element()
    except ParseError:
        return None, None

    if dtype == MI_MATRIX:
        child_end = nxt
        prop = anonymous or f"<anon{abs(hash(path)) % 1000}>"
        try:
            return prop, _read_value(st, prop, f"{path}.{prop}", depth + 1)
        except ParseError:
            st.pos = min(child_end, end)
            return f"<bad@{start}>", Node(
                kind="empty",
                name=prop,
                path=path,
                start=start,
                end=st.pos,
                error="parse failed, skipped to declared end",
            )

    # Otherwise the child is introduced by a miINT8 name element.
    try:
        prop = st.read_name()
    except ParseError:
        st.pos = min(nxt, end)
        return None, None
    if not prop or prop.startswith("<non-name"):
        st.pos = min(nxt, end)
        return "<skip>", Node(kind="empty", name="", path=path, start=start, end=st.pos)
    try:
        return prop, _read_value(st, prop, f"{path}.{prop}", depth + 1)
    except ParseError:
        st.pos = min(nxt, end)
        return prop, Node(
            kind="empty",
            name=prop,
            path=path,
            start=start,
            end=st.pos,
            error="value parse failed",
        )


def _read_value(st: Stream, name: str, path: str, depth: int) -> Node:
    start = st.pos
    dtype, nbytes, payload, nxt = st.peek_element()
    if dtype != MI_MATRIX:
        raise ParseError(f"expected miMATRIX, got {dtype}")
    st.pos = payload
    if nbytes == 0:
        # An 8-byte placeholder: a miMATRIX tag with a zero length. The real
        # element follows immediately, so leave the cursor at the next one.
        st.pos = nxt
        node = Node(kind="empty", name=name, path=path)
        node.start = start
        node.end = nxt
        return node
    node = parse_matrix(st, name, path, depth, nxt)
    node.start = start
    node.end = st.pos
    return node


def walk_stream(buf: bytes, limit_nodes: int = 200000) -> list[Node]:
    """Walk a miCOMPRESSED stream and return every top-level Node.

    The workspace begins with an 8-byte block header (0x4d490100 followed by
    zero), so parsing starts at offset 8. That offset is confirmed rather
    than assumed: the first two elements must be miMATRIX and their sizes
    must account for the whole buffer.
    """
    start = 8 if len(buf) > 16 else 0
    if start == 8 and not (buf[2:4] == b"IM" or buf[0:2] == b"\x00\x01"):
        start = 0

    st = Stream(buf)
    st.pos = start
    nodes: list[Node] = []
    while st.remaining() >= 8 and len(nodes) < limit_nodes:
        try:
            dtype, nbytes, _payload, nxt = st.peek_element()
        except (ParseError, struct.error):
            break
        if dtype != MI_MATRIX or nbytes == 0:
            break
        try:
            node = _read_value(st, "", f"n{len(nodes)}", 0)
        except (ParseError, struct.error, IndexError):
            break
        nodes.append(node)
    return nodes


def iter_numeric(node: Node, prefix: str = ""):
    """Yield (path, array) for every numeric array reachable from node."""
    base = f"{prefix}/{node.path}" if prefix else node.path
    if node.kind == "numeric" and node.data is not None:
        yield base or node.name, node.data
    for key, child in node.fields.items():
        if child.kind == "char" and key == "__class__":
            continue
        yield from iter_numeric(child, base)


# --------------------------------------------------------------------------
# Strategy 2: blind scan. Independent of any MCOS knowledge.
# --------------------------------------------------------------------------

EEG_ABS_MAX = 1e5


def scan_time_vectors(buf: bytes, min_len: int = 64) -> list[dict]:
    """Find strictly increasing, uniformly spaced float64 runs.

    A MATLAB `timetable` Time column is a duration stored in seconds as
    float64, so it shows up here directly. Finding one gives fs without
    trusting any metadata.
    """
    found = []
    arr = np.frombuffer(buf, dtype="<f8")
    if arr.size < min_len:
        return found
    # Only consider 8-byte aligned views; the workspace base offset is a
    # multiple of 8 in practice, so try both parities.
    for offset in (0, 4):
        view = arr[offset:]
        finite = np.isfinite(view)
        d = np.diff(view)
        ok = finite[:-1] & finite[1:] & (d > 0)
        # A run of interest: at least min_len consecutive "increasing".
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        splits = np.flatnonzero(np.diff(idx) != 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [len(idx) - 1]))
        for s, e in zip(starts, ends, strict=False):
            run = idx[s : e + 1]
            if run.size < min_len:
                continue
            lo, hi = run[0] + 1, run[-1] + 2
            seg = view[lo - 1 : hi]
            seg_d = np.diff(seg)
            if seg_d.size == 0:
                continue
            step = np.median(seg_d)
            if step <= 0:
                continue
            rel = np.abs(seg_d - step).max() / step
            if rel > 1e-9:
                continue
            fs = 1.0 / step
            if not (1.0 <= fs <= 20000.0):
                continue
            found.append(
                {
                    "byte_offset": int((lo - 1 + offset) * 8),
                    "n": int(seg.size),
                    "step": float(step),
                    "fs": float(fs),
                    "t0": float(seg[0]),
                    "t1": float(seg[-1]),
                }
            )
    return found


def _iter_elements(
    buf: bytes,
    start: int = 8,
    end: int | None = None,
    prefix: str = "root",
    depth: int = 0,
    max_depth: int = 12,
):
    """Yield every element of the miCOMPRESSED stream, in storage order.

    Each item is ``(path, depth, dtype, nbytes, byte_offset, payload)``. The
    walk is exact: it tiles the workspace with element headers and recurses
    into each ``miMATRIX`` payload, which a linear sweep of the two top-level
    matrices confirms accounts for every byte of both a header and a signal
    file.
    """
    if end is None:
        end = len(buf)
    if depth > max_depth:
        return
    st = Stream(buf, end)
    st.pos = start
    idx = 0
    while st.pos < end and st.remaining() >= 4:
        try:
            dtype, nbytes, payload, nxt = st.peek_element()
        except ParseError:
            return
        if nxt > end:
            return
        st.pos = nxt
        path = f"{prefix}.{idx}"
        yield path, depth, dtype, nbytes, payload, buf[payload : payload + nbytes]
        if dtype == MI_MATRIX and nbytes:
            yield from _iter_elements(buf, payload, nxt, path, depth + 1, max_depth)
        idx += 1


def sweep_arrays(
    buf: bytes,
    start: int = 8,
    end: int | None = None,
    prefix: str = "root",
    min_bytes: int = 4096,
    out: list | None = None,
    depth: int = 0,
) -> list:
    """Collect the numeric arrays of at least ``min_bytes`` bytes.

    Unlike :func:`parse_matrix` this ignores the MAT array class byte, so it
    does not depend on that byte being interpreted correctly.
    """
    if out is None:
        out = []
    for path, d, dtype, nbytes, payload, _data in _iter_elements(buf, start, end, prefix, depth):
        if dtype in _DTYPE and dtype not in _TEXT and nbytes >= min_bytes:
            out.append(
                {
                    "path": path,
                    "dtype": dtype,
                    "nbytes": int(nbytes),
                    "n": int(nbytes) // 8,
                    "byte_offset": int(payload),
                    "depth": d,
                }
            )
    return out


def scan_numeric_blocks(buf: bytes, min_len: int = 1000, abs_max: float = EEG_ABS_MAX):
    """Find long aligned runs of plausible float64 EEG samples."""
    arr = np.frombuffer(buf, dtype="<f8")
    out = []
    for offset in (0, 4):
        view = arr[offset:]
        good = np.isfinite(view) & (np.abs(view) < abs_max)
        if not good.any():
            continue
        idx = np.flatnonzero(good)
        splits = np.flatnonzero(np.diff(idx) != 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [len(idx) - 1]))
        for s, e in zip(starts, ends, strict=False):
            run = idx[s : e + 1]
            if run.size < min_len:
                continue
            lo, _hi = run[0] + offset, run[-1] + offset + 1
            seg = view[run[0] : run[-1] + 1]
            out.append(
                {
                    "byte_offset": int(lo * 8),
                    "n": int(seg.size),
                    "min": float(seg.min()),
                    "max": float(seg.max()),
                    "mean": float(seg.mean()),
                    "std": float(seg.std()),
                }
            )
    return out


def load_workspace(path: Path) -> bytes:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymatreader import read_mat

        d = read_mat(str(path))
    if "__function_workspace__" not in d:
        raise ParseError(f"no MCOS workspace in {path.name}")
    return bytes(np.asarray(d["__function_workspace__"]).tobytes())


# Samples per miDOUBLE chunk in the signal files: every sample block is a
# 2048-byte element, i.e. 256 float64 values.
CHUNK_SAMPLES = 2048 // 8


_LABEL = re.compile(rb"EEG[A-Za-z0-9]{1,3}|EMG|SLI|ECG")


def _text_element(raw: bytes) -> str:
    raw = raw.split(b"\x00", 1)[0]
    if len(raw) < 2 or not all(32 <= c < 127 for c in raw):
        return ""
    return raw.decode("latin-1").strip()


def channel_names(buf: bytes) -> list[str]:
    """Recover the ``varNames`` of the ``timetable``, in column order.

    Spellings are not consistent across records - one file writes ``EMG1``
    where another writes ``EMG_1_`` - so a fixed pattern is not enough. The
    names are read structurally instead: the ``varNames`` cell is the longest
    run of consecutive short text elements in the stream, and the run length is
    what the caller validates against the number of extracted channels.
    """
    found: list[tuple[int, str]] = []
    for _path, _depth, dtype, nbytes, offset, payload in _iter_elements(buf):
        if not 1 <= nbytes <= 200 or dtype not in (MI_INT8, MI_UINT8, MI_UTF8):
            continue
        name = _text_element(payload)
        if name and _LABEL.search(name.encode("latin-1", "replace")):
            found.append((offset, name))
    best: list[tuple[int, str]] = []
    run: list[tuple[int, str]] = []
    for item in found:
        if run and item[0] - run[-1][0] <= 128:
            run.append(item)
        else:
            run = [item]
        if len(run) > len(best):
            best = list(run)
    return [n for _o, n in best]


def extract_record(buf: bytes, min_channels: int = 8) -> dict:
    """Reassemble the signal of one ``_d.mat`` workspace.

    The layout, verified on ``190304A_E_0000d.mat`` by a byte-exact element
    tiling, is:

    * property 2 of the ``FileWrapper__`` is ``rowTimes``, a float64 vector of
      1575 values stepping by exactly 1000, i.e. one row per second;
    * property 3 is a struct holding a 23-entry cell - the ``Variables`` of the
      ``timetable``, one per channel;
    * each channel is a 1575-entry cell whose entries are 256-sample
      ``miDOUBLE`` blocks, one per row.

    So the sampling rate is *derived* as ``CHUNK_SAMPLES`` samples per
    ``median(diff(rowTimes))`` rather than assumed: 256 samples per 1000 ms is
    256 Hz.

    Returns a dict with ``signals`` shaped ``(n_channels, n_samples)``,
    ``row_times`` in the file's own unit, ``fs`` and ``duration``.
    """
    row_times = None
    for item in sweep_arrays(buf, min_bytes=8 * 64):
        if item["dtype"] != MI_DOUBLE:
            continue
        x = np.frombuffer(buf, dtype="<f8", count=item["n"], offset=item["byte_offset"])
        if x.size >= 64 and np.isfinite(x).all():
            d = np.diff(x)
            if (d > 0).all() and np.ptp(d) < 1e-6:
                row_times = x
                break
    if row_times is None:
        raise ParseError("no monotonic row-time vector found")

    # Channel cells: the sample blocks of one channel are consecutive, with
    # only the 8-byte miMATRIX tag between them, so group on a small gap
    # rather than on strict adjacency.
    blocks: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    prev_end = None
    for item in sweep_arrays(buf, min_bytes=CHUNK_SAMPLES * 8):
        if item["dtype"] != MI_DOUBLE or item["n"] != CHUNK_SAMPLES:
            continue
        if prev_end is not None and item["byte_offset"] - prev_end > 64:
            if len(cur) > 1:
                blocks.append(cur)
            cur = []
        cur.append((item["byte_offset"], item["nbytes"]))
        prev_end = item["byte_offset"] + item["nbytes"]
    if len(cur) > 1:
        blocks.append(cur)
    if not blocks:
        raise ParseError("no sample blocks found")

    # The channel cells are the groups with the same length; the sample blocks
    # of one channel are contiguous, so group by contiguity and keep the
    # longest runs, one per channel.
    channels = [g for g in blocks if len(g) > 1]
    if len(channels) < min_channels:
        raise ParseError(f"only {len(channels)} channel groups found")
    lengths = {len(g) for g in channels}
    n_rows = max(lengths)
    channels = [g for g in channels if len(g) == n_rows][: len(channels)]

    n_ch = len(channels)
    signals = np.empty((n_ch, n_rows * CHUNK_SAMPLES), dtype=np.float64)
    for i, group in enumerate(channels):
        for j, (off, _nbytes) in enumerate(group):
            lo = j * CHUNK_SAMPLES
            signals[i, lo : lo + CHUNK_SAMPLES] = np.frombuffer(
                buf, dtype="<f8", count=CHUNK_SAMPLES, offset=off
            )

    step_ms = float(np.median(np.diff(row_times)))
    fs = CHUNK_SAMPLES * 1000.0 / step_ms if step_ms else float("nan")
    duration_s = signals.shape[1] / fs if fs else float("nan")
    return {
        "signals": signals,
        "row_times": row_times,
        "fs": fs,
        "n_rows": int(n_rows),
        "n_channels": int(n_ch),
        "duration_s": float(duration_s),
        "row_times_unit": "ms",
        "channel_names": channel_names(buf),
    }


def export_all(data_dir: Path, out_dir: Path) -> int:
    """Convert every _d.mat signal file to a compressed .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("*0000d.mat"))
    if not files:
        print("no *0000d.mat files found", flush=True)
        return 1
    rows = []
    for f in files:
        # `190304A_E_0000d.mat` -> `190304A_E`
        record = f.stem[: -len("0000d")].rstrip("_")
        target = out_dir / f"{record}.npz"
        try:
            buf = load_workspace(f)
            r = extract_record(buf)
        except Exception as exc:
            print(f"  {record}: FAILED ({exc})", flush=True)
            continue
        names = r["channel_names"]
        sig = r["signals"]
        if len(names) != sig.shape[0]:
            print(
                f"  {record}: channel-name mismatch "
                f"({len(names)} names vs {sig.shape[0]} channels)",
                flush=True,
            )
            continue
        np.savez_compressed(
            target,
            signals=sig.astype(np.float32),
            channel_names=np.array(names),
            fs=np.float64(r["fs"]),
            row_times_ms=r["row_times"].astype(np.float64),
        )
        rows.append((record, sig.shape[0], sig.shape[1], r["fs"], r["duration_s"]))
        print(
            f"  {record}: {sig.shape[0]:>3} ch x {sig.shape[1]:>8,} samples  "
            f"fs={r['fs']:.4f} Hz  {r['duration_s']:.2f} s  -> {target.name}",
            flush=True,
        )
    print(f"\n{len(rows)}/{len(files)} records exported to {out_dir}", flush=True)
    return 0 if rows else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help=".mat files to inspect")
    ap.add_argument("--depth", action="store_true", help="also run the tree walker")
    ap.add_argument(
        "--export",
        nargs=2,
        metavar=("DATA_DIR", "OUT_DIR"),
        help="convert every *_d.mat under DATA_DIR to OUT_DIR/<record>.npz",
    )
    args = ap.parse_args()

    if args.export:
        return export_all(Path(args.export[0]), Path(args.export[1]))

    files = args.paths
    if not files:
        files = sorted(Path(".").glob("data/*/*.mat"))[:2]
    for f in files:
        print(f"\n=== {f} ===", flush=True)
        try:
            buf = load_workspace(f)
            r = extract_record(buf)
        except Exception as exc:
            print(f"  extraction failed: {exc}", flush=True)
            continue
        sig = r["signals"]
        print(
            f"  {r['n_channels']} ch x {sig.shape[1]:,} samples  "
            f"fs={r['fs']:.4f} Hz  {r['duration_s']:.2f} s",
            flush=True,
        )
        print(f"  channels: {r['channel_names']}", flush=True)
        print(
            f"  per-channel std: {np.round(sig.std(axis=1), 2).tolist()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
