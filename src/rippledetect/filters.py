"""Signal-processing helpers: bandpass filtering, envelope, smoothing.
Ported from NREMripple.m."""
from __future__ import annotations
import numpy as np
from scipy.signal import cheby1, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d


def bandpass_filter(signal, fs, low=100.0, high=300.0, order=4, ripple_db=0.1):
    """Zero-phase Chebyshev Type I bandpass filter (MATLAB cheby1 + filtfilt)."""
    nyq = 0.5 * fs
    b, a = cheby1(order, ripple_db, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def amplitude_envelope(signal):
    """Amplitude envelope via the Hilbert transform -> abs(hilbert(x))."""
    return np.abs(hilbert(signal))


def moving_average(signal, fs, window_ms=10.0):
    """Centered moving-average smoothing over a window in ms (MATLAB movmean)."""
    win = max(1, round(window_ms / 1000.0 * fs))
    return uniform_filter1d(signal, size=win)