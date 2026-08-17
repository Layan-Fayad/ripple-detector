"""Basic unit tests for the ripple detector.

Run with:  pytest
"""

from rippledetect import detect_ripples, detect_ripples_array, precision_recall_f1
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _signal_with_one_ripple(fs=1250, duration=4.0, center=2.0):
    """A quiet signal with a single clear ripple injected at ``center``."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    lfp = 0.1 * np.random.default_rng(0).standard_normal(n)
    dur = 0.06
    i0, i1 = int((center - dur / 2) * fs), int((center + dur / 2) * fs)
    tt = np.arange(i1 - i0) / fs
    lfp[i0:i1] += 2.0 * np.hanning(len(tt)) * np.sin(2 * np.pi * 170 * tt)
    return lfp, fs


def test_detects_a_clear_ripple():
    lfp, fs = _signal_with_one_ripple()
    ripples = detect_ripples_array(lfp, fs)
    assert len(ripples) >= 1


def test_returns_seconds_within_bounds():
    lfp, fs = _signal_with_one_ripple(duration=4.0)
    ripples = detect_ripples(lfp, fs)
    assert np.all(ripples >= 0)
    assert np.all(ripples <= 4.0)
    # Each event: start < end.
    assert np.all(ripples[:, 0] < ripples[:, 1])


def test_quiet_signal_has_few_detections():
    rng = np.random.default_rng(1)
    lfp = 0.1 * rng.standard_normal(1250 * 4)
    ripples = detect_ripples(lfp, fs=1250)
    assert len(ripples) <= 2  # allow the rare noise blip


def test_metrics_perfect_match():
    events = np.array([[1.0, 1.05], [2.0, 2.06]])
    scores = precision_recall_f1(events, events)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
