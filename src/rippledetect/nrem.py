"""NREM-like state classification from the theta/delta power ratio.
Ported from NREMripple.m."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d


def _bandpass(signal, fs, low, high, order=3):
    """Zero-phase Butterworth bandpass (MATLAB butter + filtfilt)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def classify_nrem(lfp, fs, k=0.5, power_smooth_sec=5.0, state_smooth_sec=10.0):
    """Classify NREM-like samples from the theta/delta power ratio.
    Returns (is_nrem, ratio, threshold)."""
    # Band-power proxies: filter, square, smooth.
    delta = _bandpass(lfp, fs, 0.5, 4.0) ** 2
    theta = _bandpass(lfp, fs, 4.0, 10.0) ** 2
    win = max(1, round(power_smooth_sec * fs))
    delta_s = uniform_filter1d(delta, size=win)
    theta_s = uniform_filter1d(theta, size=win)
    ratio = theta_s / delta_s

    # Threshold at mean - k*SD, then require the state to persist.
    threshold = ratio.mean() - k * ratio.std()
    is_nrem_raw = ratio < threshold
    win2 = max(1, round(state_smooth_sec * fs))
    is_nrem = uniform_filter1d(is_nrem_raw.astype(float), size=win2) > 0.5
    return is_nrem, ratio, threshold


def gate_by_nrem(starts, ends, peaks, is_nrem):
    """Keep only ripples whose peak sample falls in an NREM-like window."""
    if len(peaks) == 0:
        return starts, ends, peaks
    mask = is_nrem[peaks]
    return starts[mask], ends[mask], peaks[mask]
