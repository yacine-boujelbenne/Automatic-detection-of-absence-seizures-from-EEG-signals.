"""Closed-form checks for eeg_features. Run: python test_reference.py

Band-power convention used by eeg_features.band_powers:
    P1[k] = |rfft(w)[k]|^2 / n^2   for k = 0 and k = n/2
    P1[k] = 2 |rfft(w)[k]|^2 / n^2 for 0 < k < n/2
    band power = sum of P1 over bins with lo <= f < hi
This is the standard one-sided power spectrum, so a real sinusoid of amplitude
A contributes exactly A^2/2 to whichever band contains it, and
sum over all bands + DC = mean(w^2) (Parseval).
"""

import numpy as np

import eeg_features as F

FS = 256.0
N = 512
fails = []
n_checks = 0


def check(name, got, want, tol=1e-9):
    global n_checks
    n_checks += 1
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print("%-48s %-16s want=%-14g %s" % (name, ("%.10g" % got), want, "ok" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def check_eq(name, got, want):
    global n_checks
    n_checks += 1
    g = tuple(got) if hasattr(got, "__iter__") else (got,)
    w = tuple(want) if hasattr(want, "__iter__") else (want,)
    ok = g == w
    print("%-48s %-16s want=%-14s %s" % (name, g, w, "ok" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def sine(freq, amp=2.0, n=N, fs=FS):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs)


print("== window geometry ==")
check("window_length(256, 2.0) == 512", F.window_length(FS, 2.0), 512)
x = np.arange(1024.0)
wins = F.iter_windows(x, 512, 0.5)
check("n windows @50% overlap (1024 samp)", wins.shape[0], 3)
check("window 0 is samples 0..511", wins[0, 0], 0.0)
check("window 1 starts at hop 256", wins[1, 0], 256.0)
check("window 2 starts at 512", wins[2, 0], 512.0)
check("window 2 ends at 1023", wins[2, -1], 1023.0)
check("n windows, no overlap", F.iter_windows(x, 512, 0.0).shape[0], 2)
check("n windows, 75% overlap (hop 128)", F.iter_windows(x, 512, 0.75).shape[0], 5)
check("short signal -> 0 windows", F.iter_windows(np.arange(100.0), 512, 0.5).shape[0], 0)
check("exact fit -> 1 window", F.iter_windows(np.arange(512.0), 512, 0.5).shape[0], 1)

print("\n== temporal features vs closed form ==")
w = np.array([1.0, 2.0, 3.0, 4.0])
tf = F.temporal_features(w)
check("mean", tf["mean"], 2.5)
check("std (ddof=0)", tf["std"], np.sqrt(1.25))
check("variance", tf["variance"], 1.25)
check("min", tf["min"], 1.0)
check("max", tf["max"], 4.0)
check("amplitude = max-min", tf["amplitude"], 3.0)
check("rms", tf["rms"], np.sqrt(7.5))
check("energy = sum of squares", tf["energy"], 30.0)
check("all-zero rms", F.temporal_features(np.zeros(10))["rms"], 0.0)
check("all-zero energy", F.temporal_features(np.zeros(10))["energy"], 0.0)
check("single sample -> zero std", F.temporal_features(np.array([5.0]))["std"], 0.0)

print("\n== band powers vs analytic ==")
p10 = F.band_powers(sine(10.0), FS)
check("10 Hz lands in alpha", p10["power_alpha"], 2.0)  # A^2/2 = 4/2
check("10 Hz absent from beta", p10["power_beta"], 0.0, 1e-12)
check("10 Hz absent from theta", p10["power_theta"], 0.0, 1e-12)
check("10 Hz absent from delta", p10["power_delta"], 0.0, 1e-12)
check("sum of band powers == mean(w^2)", sum(p10.values()), np.mean(sine(10.0) ** 2), 1e-12)

for freq, band in [(1.0, "delta"), (6.0, "theta"), (10.0, "alpha"), (20.0, "beta")]:
    p = F.band_powers(sine(freq), FS)
    check("%g Hz -> power_%s == A^2/2" % (freq, band), p["power_%s" % band], 2.0, 1e-9)

print("\n== band edges: low inclusive, high exclusive ==")
p4 = F.band_powers(sine(4.0), FS)
check("4 Hz excluded from delta (0.5-4)", p4["power_delta"], 0.0, 1e-12)
check("4 Hz included in theta (4-8)", p4["power_theta"], 2.0, 1e-9)
p13 = F.band_powers(sine(13.0), FS)
check("13 Hz excluded from alpha (8-13)", p13["power_alpha"], 0.0, 1e-12)
check("13 Hz included in beta (13-30)", p13["power_beta"], 2.0, 1e-9)
# Band edges are [lo, hi), so the upper edge 30 Hz belongs to no band.
check("30 Hz excluded from beta (high edge exclusive)", F.band_powers(sine(30.0), FS)["power_beta"], 0.0, 1e-12)
check("29 Hz included in beta", F.band_powers(sine(29.0), FS)["power_beta"], 2.0, 1e-9)
# Nyquist is 128 Hz here, so everything the bands can see is below it.
check("no energy above Nyquist", np.fft.rfftfreq(N, 1 / FS)[-1], FS / 2)

print("\n== DC handling ==")
dc = np.full(N, 3.0)
pdc = F.band_powers(dc, FS)
check("DC excluded from every band (delta starts 0.5 Hz)", sum(pdc.values()), 0.0, 1e-12)
white = np.ones(N)
pdcw = F.band_powers(white, FS)
check("DC term does not leak into delta", pdcw["power_delta"], 0.0, 1e-12)

print("\n== seizure mask uses (debut, fin] ==")
m = F.seizure_mask(8, FS, [(0.0, 3 / FS)])
check("samples flagged", int(m.sum()), 3)
check("t=0 excluded (open left)", bool(m[0]), False)
check("t=1/fs included", bool(m[1]), True)
check("t=3/fs included (closed right)", bool(m[3]), True)
check("t=4/fs excluded", bool(m[4]), False)
check("reversed interval handled", int(F.seizure_mask(8, FS, [(3 / FS, 0.0)]).sum()), 3)
check("NaN interval ignored", int(F.seizure_mask(8, FS, [(np.nan, 1.0)]).sum()), 0)
check("disjoint intervals union", int(F.seizure_mask(8, FS, [(0.0, 1 / FS), (4 / FS, 5 / FS)]).sum()), 2)

print("\n== window labelling at the 50% rule ==")
# 8 samples, window 4, hop 2 -> windows [0,1,1,1] [1,1,0,0] [0,0,0,0]
y = F.label_windows(np.array([0, 1, 1, 1, 0, 0, 0, 0], dtype=float), 4, 0.5, 0.5)
check_eq("labels", y, [1, 1, 0])
check("exactly 50% counts as seizure (>=)", F.label_windows(np.array([0, 1, 1, 0.0]), 4, 0.0, 0.5)[0], 1)
check("just under 50% -> 0", F.label_windows(np.array([0, 1, 0, 0.0]), 4, 0.0, 0.5)[0], 0)
check("full window seizure -> 1", F.label_windows(np.ones(4), 4, 0.0, 0.5)[0], 1)
check("empty window seizure -> 0", F.label_windows(np.zeros(4), 4, 0.0, 0.5)[0], 0)

print("\n== interval reconstruction (Part X) ==")
pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])
iv = F.predictions_to_intervals(pred, FS, 2.0, 0.5)
check_eq("two intervals recovered", len(iv), 2)
check("interval 1 start (idx2*1.0 - 1.0)", iv[0][0], 1.0)
check("interval 1 end (idx4*1.0 + 1.0)", iv[0][1], 5.0)
check("interval 2 start", iv[1][0], 6.0)
check("interval 2 end", iv[1][1], 8.0)
check("all-zero -> none", len(F.predictions_to_intervals(np.zeros(5, dtype=int), FS)), 0)
check_eq("all-one -> one", len(F.predictions_to_intervals(np.ones(5, dtype=int), FS)), 1)
check("trailing run not dropped", F.predictions_to_intervals(np.array([0, 1, 1]), FS)[0][1], 3.0)

print("\n== event matching (event-level TP/FP/FN) ==")
truth = [(10.0, 20.0), (40.0, 50.0)]
check_eq("perfect prediction", F.match_events(truth, [(10.0, 20.0), (40.0, 50.0)]), (2, 0, 0))
check_eq("one missed", F.match_events(truth, [(10.0, 20.0)]), (1, 0, 1))
check_eq("one spurious", F.match_events(truth, [(10.0, 20.0), (40.0, 50.0), (70.0, 80.0)]), (2, 1, 0))
check_eq("completely disjoint", F.match_events(truth, [(25.0, 35.0)]), (0, 1, 2))
check_eq("no truth", F.match_events([], [(1.0, 2.0)]), (0, 1, 0))
check_eq("no predictions", F.match_events(truth, []), (0, 0, 2))
check_eq("duplicate predictions not double-counted", F.match_events(truth, [(10.0, 20.0), (10.0, 20.0)]), (1, 1, 1))
check_eq("partial overlap below IoU 0.5", F.match_events([(0.0, 10.0)], [(0.0, 20.0)]), (0, 1, 1))

print("\n== end-to-end on a synthetic recording ==")
rng = np.random.default_rng(0)
dur = 20.0
tt = np.arange(int(dur * FS)) / FS
sig = 0.5 * np.sin(2 * np.pi * 10 * tt) + 0.05 * rng.normal(size=tt.size)
rows, yy = F.extract_dataset(sig, FS, [(5.0, 9.0)], 2.0, 0.5, 0.5)
check("window count over 20 s", len(rows), (int(dur * FS) - 512) // 256 + 1)
check("12 features + t_start", len(rows[0]), 13)
check("labels align with windows", len(rows), len(yy))
check("at least one positive window", int(yy.sum() > 0), 1)
check("alpha power dominates for a 10 Hz signal", rows[0]["power_alpha"] > rows[0]["power_beta"], 1)
print("  sample row:", {k: round(v, 5) for k, v in list(rows[0].items())})

print("\n== Parseval / band coverage ==")
# Exact Parseval: the one-sided power spectrum over ALL bins equals mean(w^2).
spec = np.fft.rfft(sine(10.0))
allbins = (np.abs(spec) ** 2) / N**2
allbins[1:-1] *= 2
check("one-sided power over all bins == mean(w^2)", allbins.sum(), np.mean(sine(10.0) ** 2), 1e-12)

# The 4 bands span 0.5-30 Hz, but a 2 s / 256 Hz window resolves 0-128 Hz.
# So the bands must capture only the low-frequency share, NOT everything.
noise = rng.normal(size=N)
p = F.band_powers(noise, FS)
share = sum(p.values()) / np.mean(noise**2)
n_band_bins = int(((np.fft.rfftfreq(N, 1 / FS) >= 0.5) & (np.fft.rfftfreq(N, 1 / FS) < 30)).sum())
expected = n_band_bins / (N // 2 + 1)
print("  bands cover %d of %d bins -> expected power share ~%.3f, got %.3f"
      % (n_band_bins, N // 2 + 1, expected, share))
check("band share within 25%% of bin fraction", abs(share - expected) < 0.25 * expected, 1)
check("bands do NOT capture the whole spectrum", share < 0.5, 1)

print("\n" + ("ALL PASS (%d checks)" % n_checks if not fails
             else "FAILURES %d of %d: %s" % (len(fails), n_checks, fails)))
raise SystemExit(1 if fails else 0)
