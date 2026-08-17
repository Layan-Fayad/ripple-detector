"""End-to-end demo: synthetic LFP -> detection -> validation -> figure.

Run from the repo root:  python notebooks/demo.py

This uses *synthetic* data with known injected ripples so the pipeline runs
out of the box. Replace this with a loader for your own recordings (or a
public dataset such as CRCNS hc-11) to produce the real README figures and
validation numbers.
"""

from rippledetect import detect_ripples, detect_ripples_array, plot_detection, precision_recall_f1
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_synthetic_lfp(fs=1250, duration=20.0, n_ripples=15, seed=0):
    """Generate a synthetic LFP with pink-ish noise and injected ripples."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    lfp = 0.5 * np.sin(2 * np.pi * 7 * t)
    lfp += rng.standard_normal(n) * 0.4

    truth = []
    for _ in range(n_ripples):
        center = rng.uniform(0.5, duration - 0.5)
        dur = rng.uniform(0.04, 0.09)
        freq = rng.uniform(140, 200)
        i0 = int((center - dur / 2) * fs)
        i1 = int((center + dur / 2) * fs)
        tt = np.arange(i1 - i0) / fs
        window = np.hanning(len(tt))
        lfp[i0:i1] += 1.6 * window * np.sin(2 * np.pi * freq * tt)
        truth.append([center - dur / 2, center + dur / 2])

    return lfp, fs, np.array(sorted(truth))


def main():
    lfp, fs, ground_truth = make_synthetic_lfp()

    ripples = detect_ripples_array(lfp, fs)
    print(f"Detected {len(ripples)} ripples (injected {len(ground_truth)}).")

    scores = precision_recall_f1(ripples, ground_truth)
    print(
        f"Precision {scores['precision']:.2f} | "
        f"Recall {scores['recall']:.2f} | "
        f"F1 {scores['f1']:.2f} "
        f"(TP {scores['tp']}, FP {scores['fp']}, FN {scores['fn']})"
    )

    starts = (ripples[:, 0] * fs).astype(int)
    ends = (ripples[:, 1] * fs).astype(int)
    peaks = ((ripples[:, 0] + ripples[:, 1]) / 2 * fs).astype(int)

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_detection(
        lfp, fs, starts, ends, peaks, t_start=0, t_end=6,
        save_path=os.path.join(fig_dir, "hero.png"),
    )
    print("Saved figures/hero.png")
    plt.show()


if __name__ == "__main__":
    main()
