"""Prediction models, validation, persistence and drift detection."""

from signalforge.models.drift import DriftReport, assess, population_stability_index
from signalforge.models.ensemble import CLASSES, ModelReport, SignalModel
from signalforge.models.registry import ModelEntry, ModelRegistry
from signalforge.models.validation import (
    Fold,
    PurgedWalkForward,
    check_no_leakage,
    walk_forward_predict,
)

__all__ = [
    "SignalModel",
    "ModelReport",
    "CLASSES",
    "ModelRegistry",
    "ModelEntry",
    "PurgedWalkForward",
    "Fold",
    "walk_forward_predict",
    "check_no_leakage",
    "DriftReport",
    "assess",
    "population_stability_index",
]
