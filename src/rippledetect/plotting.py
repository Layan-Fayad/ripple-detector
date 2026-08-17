"""Visualization of detected ripples. Ported from NREMripple.m."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from .filters import amplitude_envelope, bandpass_filter, moving_average


def _zscored_envelope(lfp, fs, low=100.0, high=300.0, smooth_ms=10.0):
    filtered = bandpass_filter(lfp, fs, low, high)
    env = moving_average(amplitude_envelope(filtered), fs, smooth_ms)
    z = (env - env.mean()) / env.std()
    return filtered, z


def plot_detection(lfp, fs, starts, ends, peaks, low=100.0, high=300.0,
                   edge_sd=0.5, t_start=None, t_end=None, window_s=3.0, save_path=None):
    """Raw LFP, ripple-band signal, and z-envelope; detected ripples shaded.
    With no time window, auto-centers on the strongest detected ripple."""
    filtered, z = _zscored_envelope(lfp, fs, low, high)
    if t_start is None or t_end is None:
        center = (peaks[np.argmax(z[peaks])] /
                  fs) if len(peaks) else window_s / 2
        t_start = max(0, center - window_s / 2)
        t_end = min(len(lfp) / fs, center + window_s / 2)
    i0, i1 = int(t_start * fs), int(t_end * fs)
    t = np.arange(i0, i1) / fs
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, lfp[i0:i1], color="#333333", lw=0.6)
    axes[0].set_ylabel("Raw LFP (mV)")
    axes[1].plot(t, filtered[i0:i1], color="#1F3A5F", lw=0.6)
    axes[1].set_ylabel("100-300 Hz")
    axes[2].plot(t, z[i0:i1], color="#1F3A5F", lw=0.8)
    axes[2].axhline(edge_sd, color="crimson", ls="--", lw=0.8)
    axes[2].set_ylabel("Envelope (z)")
    axes[2].set_xlabel("Time (s)")
    for s, e in zip(starts, ends):
        if e / fs < t_start or s / fs > t_end:
            continue
        for ax in axes:
            ax.axvspan(s / fs, e / fs, color="orange", alpha=0.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_event_grid(lfp, fs, starts, ends, peaks, low=100.0, high=300.0,
                    n=16, win_ms=150.0, seed=0, save_path=None):
    """Grid of example detected events: raw (black) + filtered (red)."""
    if len(peaks) == 0:
        return None
    filtered = bandpass_filter(lfp, fs, low, high)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(
        len(peaks), size=min(n, len(peaks)), replace=False))
    win = int(win_ms / 1000 * fs)
    ncols = 4
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for slot, ev in enumerate(idx):
        pk = peaks[ev]
        a, b = pk - win, pk + win
        if a < 0 or b > len(lfp):
            continue
        tt = (np.arange(a, b) - pk) / fs * 1000
        ax = axes[slot]
        ax.axis("on")
        ax.plot(tt, lfp[a:b], color="k", lw=0.6)
        ax2 = ax.twinx()
        ax2.plot(tt, filtered[a:b], color="crimson", lw=0.6)
        dur = (ends[ev] - starts[ev]) / fs * 1000
        ax.set_title(f"#{ev} | {dur:.0f} ms", fontsize=8)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax.set_xlabel("ms", fontsize=7)
    fig.suptitle("Example detected ripples  (raw = black, filtered = red)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_average_ripple(lfp, fs, peaks, low=100.0, high=300.0, win_ms=100.0, save_path=None):
    """Mean raw waveform and mean filtered envelope, aligned on ripple peaks."""
    if len(peaks) == 0:
        return None
    filtered = bandpass_filter(lfp, fs, low, high)
    win = int(win_ms / 1000 * fs)
    raw, filt = [], []
    for pk in peaks:
        a, b = pk - win, pk + win + 1
        if a < 0 or b > len(lfp):
            continue
        raw.append(lfp[a:b])
        filt.append(filtered[a:b])
    raw = np.array(raw)
    filt = np.array(filt)
    tt = np.arange(-win, win + 1) / fs * 1000
    env_mean = np.abs(hilbert(filt, axis=1)).mean(0)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(tt, raw.mean(0), color="k", lw=2)
    axes[0].set_ylabel("Mean raw (mV)")
    axes[0].set_title(f"Average across {len(raw)} ripples")
    axes[1].plot(tt, env_mean, color="crimson", lw=2)
    axes[1].set_ylabel("Mean filtered envelope")
    axes[1].set_xlabel("Time from peak (ms)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_duration_histogram(starts, ends, fs, min_ms=15.0, max_ms=100.0, save_path=None):
    """Histogram of ripple durations with the min/max cutoffs marked."""
    dur = (ends - starts) / fs * 1000
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dur, bins=np.arange(0, dur.max() + 5, 5),
            color="#4d7fbf", edgecolor="white")
    ax.axvline(min_ms, color="crimson", ls="--")
    ax.axvline(max_ms, color="crimson", ls="--")
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Number of events")
    ax.set_title("Distribution of ripple durations")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
