# Requires: data/Baseline_HPC_Probe.mat
# Place your own .mat recording in the data/ folder and update the
# channel number (ch) and file path to match your recording.

from rippledetect.nrem import classify_nrem, gate_by_nrem
from rippledetect.detection import detect_ripples
import os
import sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))


# ── USER SETTINGS ────────────────────────────────────────────────
DATA_PATH = "data/Baseline_HPC_Probe.mat"
CHANNEL = 8      # str. pyramidale channel (identified via CSD)
# ─────────────────────────────────────────────────────────────────

m = loadmat(DATA_PATH)
ch = CHANNEL
i0 = int(m["datastart"][ch-1, 0]) - 1
i1 = int(m["dataend"][ch-1, 0])
hc = m["data"].ravel()[i0:i1] * 1000.0
hc = hc - hc.mean()
fs = float(m["samplerate"][ch-1, 0])
print(f"Loaded {len(hc)/fs/60:.1f} min at {fs:.0f} Hz")

starts, ends, peaks = detect_ripples(
    hc, fs, min_duration_ms=30, max_duration_ms=100)
print(f"Detected {len(starts)} ripples total")

is_nrem, ratio, thr = classify_nrem(hc, fs)
print(
    f"NREM-like: {is_nrem.sum()/fs/60:.1f} min ({100*is_nrem.mean():.1f}% of recording)")

starts, ends, peaks = gate_by_nrem(starts, ends, peaks, is_nrem)
print(f"Ripples during NREM: {len(starts)}")
