"""Explosive-move and compression detection."""

from signalforge.anomaly.detector import (
    AnomalyReport,
    anomaly_features,
    coiling_series,
    detect,
    ignition_series,
    scan,
)

__all__ = [
    "AnomalyReport",
    "detect",
    "scan",
    "anomaly_features",
    "ignition_series",
    "coiling_series",
]
