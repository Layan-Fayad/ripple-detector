"""Core sharp-wave ripple detection, ported from NREMripple.m.
Dual-threshold detection on the z-scored envelope, with peak tracking and a
duration filter. Returns sample indices (start, end, peak) per ripple."""
from __future__ import annotations
import numpy as np
from .filters import amplitude_envelope, bandpass_filter, moving_average


def detect_ripples(lfp, fs, low=100.0, high=300.0, edge_sd=0.5, peak_sd=3.0,
                   smooth_ms=10.0, min_duration_ms=30.0, max_duration_ms=100.0):
    """Detect sharp-wave ripples. Returns (starts, ends, peaks) as sample indices."""
    # 1. Bandpass to the ripple band.
    filtered = bandpass_filter(lfp, fs, low, high)

    # 2. Envelope -> smooth -> z-score.
    env = moving_average(amplitude_envelope(filtered), fs, smooth_ms)
    z = (env - env.mean()) / env.std()

    # 3. Find contiguous runs above the edge threshold.
    #    MATLAB: diff([0; above; 0]); starts=find(==1); ends=find(==-1)-1
    above_edge = z > edge_sd
    padded = np.concatenate(([0], above_edge.astype(int), [0]))
    d = np.diff(padded)
    edge_starts = np.where(d == 1)[0]
    edge_ends = np.where(d == -1)[0] - 1

    # 4. Keep events whose peak exceeds the peak threshold; record the peak.
    starts, ends, peaks = [], [], []
    for s, e in zip(edge_starts, edge_ends):
        seg = z[s: e + 1]
        offset = int(np.argmax(seg))
        if seg[offset] >= peak_sd:
            starts.append(s)
            ends.append(e)
            peaks.append(s + offset)

    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    peaks = np.asarray(peaks, dtype=int)

    # 5. Duration filter.
    if starts.size:
        durations_ms = (ends - starts) / fs * 1000.0
        keep = (durations_ms >= min_duration_ms) & (
            durations_ms <= max_duration_ms)
        starts, ends, peaks = starts[keep], ends[keep], peaks[keep]

    return starts, ends, peaks

#


def detect_ripples_array(lfp, fs, **kwargs):
    """Convenience wrapper — returns (N, 2) array of [start, end] in seconds.
    Suitable for use with metrics.precision_recall_f1 and demo.py."""
    starts, ends, peaks = detect_ripples(lfp, fs, **kwargs)
    if len(starts) == 0:
        return np.empty((0, 2))
    return np.column_stack([starts / fs, ends / fs])
