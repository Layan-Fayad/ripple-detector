from rippledetect.plotting import (plot_detection, plot_event_grid,
                                   plot_average_ripple, plot_duration_histogram)
from rippledetect.nrem import classify_nrem, gate_by_nrem
from rippledetect.detection import detect_ripples
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Load your recording
m = loadmat("data/Baseline_HPC_Probe.mat")
ch = 8
i0 = int(m["datastart"][ch-1, 0]) - 1
i1 = int(m["dataend"][ch-1, 0])
hc = m["data"].ravel()[i0:i1] * 1000.0
hc = hc - hc.mean()
fs = float(m["samplerate"][ch-1, 0])
print(f"Loaded {len(hc)/fs/60:.1f} min at {fs:.0f} Hz, ch={ch}")

starts, ends, peaks = detect_ripples(
    hc, fs, min_duration_ms=30, max_duration_ms=100)
print(f"Detected {len(starts)} ripples before NREM filter")

is_nrem, ratio, thr = classify_nrem(hc, fs)
print(f"NREM: {is_nrem.sum()/fs/60:.1f} min ({100*is_nrem.mean():.1f}%)")

starts, ends, peaks = gate_by_nrem(starts, ends, peaks, is_nrem)
print(f"Ripples during NREM: {len(starts)}")

plot_detection(hc, fs, starts, ends, peaks, save_path="figures/hero.png")
plot_event_grid(hc, fs, starts, ends, peaks)
plot_average_ripple(hc, fs, peaks, save_path="figures/average_ripple.png")
plot_duration_histogram(starts, ends, fs)

plt.show()
