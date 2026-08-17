"""rippledetect: sharp-wave ripple detection in hippocampal LFP."""
__version__ = "0.1.0"

from .detection import detect_ripples, detect_ripples_array
from .plotting import plot_detection, plot_average_ripple, plot_event_grid
from .metrics import precision_recall_f1
from .nrem import classify_nrem, gate_by_nrem

__all__ = [
    "detect_ripples",
    "detect_ripples_array",
    "plot_detection",
    "plot_average_ripple",
    "plot_event_grid",
    "precision_recall_f1",
    "classify_nrem",
    "gate_by_nrem",
]
