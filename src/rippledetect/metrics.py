"""Evaluate detected events against ground-truth (expert-labeled) events."""

from __future__ import annotations

import numpy as np


def _overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    """True if intervals a=[s,e] and b=[s,e] overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


def match_events(
    detected: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[int, int, int]:
    """Match detected events to ground-truth events by temporal overlap.

    Each ground-truth event can be matched by at most one detection and vice
    versa (greedy one-to-one matching).

    Parameters
    ----------
    detected, ground_truth : np.ndarray, shape (n, 2)
        Arrays of [start, end] times (same units, e.g. seconds).

    Returns
    -------
    tuple[int, int, int]
        (true_positives, false_positives, false_negatives).
    """
    detected = np.atleast_2d(detected)
    ground_truth = np.atleast_2d(ground_truth)

    matched_gt: set[int] = set()
    true_positives = 0

    for det in detected:
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            if _overlaps(det, gt):
                matched_gt.add(j)
                true_positives += 1
                break

    false_positives = len(detected) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)
    return true_positives, false_positives, false_negatives


def precision_recall_f1(
    detected: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for detected vs ground-truth events.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``.
    """
    tp, fp, fn = match_events(detected, ground_truth)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
